#!/usr/bin/env python3
"""ROS adapter node for SAM dynamic obstacle segmentation."""

import logging
import threading

import message_filters
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from umap.msg import Obstacles, RematchStatistics

from config_manager import ExtractorConfig
from model_manager import ModelManager
from sam_adapters import SamInputAdapter
from sam_backend import SamBackend
from sam_publishers import SamRosPublisher
from utils import ColorGenerator, ImageProcessor


class SamRosNode:
    def __init__(self):
        self.config = ExtractorConfig()

        self.bridge = CvBridge()
        self.model_manager = ModelManager(self.config)
        self.image_processor = ImageProcessor(self.config)
        self.color_generator = ColorGenerator(self.config.MAX_COLORS)

        self.backend = SamBackend(
            config=self.config,
            model_manager=self.model_manager,
            image_processor=self.image_processor,
            color_generator=self.color_generator,
            logger=logging.getLogger(__name__),
        )
        self.input_adapter = SamInputAdapter(self.bridge)

        self.sync_queue = rospy.get_param("~sync_queue", 2)
        self.sync_slop = rospy.get_param("~sync_slop", 0.2)

        # Background worker: callback only updates latest job, worker processes it.
        self._job_lock = threading.Lock()
        self._latest_job = None
        self._worker_event = threading.Event()
        self._running = True

        self._setup_ros_interface()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        rospy.on_shutdown(self._shutdown)
        rospy.loginfo("SamRosNode initialized successfully")

    def _setup_ros_interface(self):
        image_topic = rospy.get_param("~image_topic", "/camera/rgb/image_raw")
        self.image_sub = message_filters.Subscriber(image_topic, Image)
        self.obstacles_sub = message_filters.Subscriber("tracker/dynamic_obstacles", Obstacles)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.obstacles_sub],
            queue_size=self.sync_queue,
            slop=self.sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self._synced_callback)

        self.vis_pub = rospy.Publisher("/umap/visualization/segmented_image", Image, queue_size=10)
        self.obstacles_pub = rospy.Publisher("/model_output/obstacles_sam", Obstacles, queue_size=10)
        self.roi_vis_pub = rospy.Publisher("/model_output/visualization/roi_image", Image, queue_size=10)
        self.mask_vis_pub = rospy.Publisher("/model_output/visualization/mask_debug", Image, queue_size=10)
        self.segmented_image_pub = rospy.Publisher("/model_output/segmented_image", Image, queue_size=10)
        self.statistics_pub = rospy.Publisher("/rematch_statistics_sam", RematchStatistics, queue_size=10)

        self.publisher = SamRosPublisher(
            bridge=self.bridge,
            config=self.config,
            vis_pub=self.vis_pub,
            obstacles_pub=self.obstacles_pub,
            roi_vis_pub=self.roi_vis_pub,
            mask_vis_pub=self.mask_vis_pub,
            segmented_image_pub=self.segmented_image_pub,
            statistics_pub=self.statistics_pub,
        )

    def _synced_callback(self, image_msg, obstacles_msg):
        if not obstacles_msg.box_ids:
            return

        with self._job_lock:
            # Keep latest synchronized pair and drop stale pending ones.
            self._latest_job = (image_msg, obstacles_msg)
        self._worker_event.set()

    def _worker_loop(self):
        while self._running and not rospy.is_shutdown():
            self._worker_event.wait()
            self._worker_event.clear()

            job = None
            with self._job_lock:
                job = self._latest_job
                self._latest_job = None

            if job is None:
                continue

            image_msg, obstacles_msg = job
            self._process_job(image_msg, obstacles_msg)

    def _process_job(self, image_msg, obstacles_msg):
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            request = self.input_adapter.build_request(image, obstacles_msg)

            # Skip expensive visual generation when nobody subscribes.
            request.generate_overlay = self.segmented_image_pub.get_num_connections() > 0
            request.generate_vis_image = self.vis_pub.get_num_connections() > 0
            request.generate_mask_debug = (
                self.config.ENABLE_MASK_DEBUG and self.mask_vis_pub.get_num_connections() > 0
            )
            request.generate_roi_grid = self.roi_vis_pub.get_num_connections() > 0

            result = self.backend.process(request)
            self.publisher.publish(obstacles_msg, result)
        except Exception as e:
            rospy.logerr("Worker processing failed: %s", str(e))

    def _shutdown(self):
        self._running = False
        self._worker_event.set()
        if hasattr(self, "_worker") and self._worker.is_alive():
            self._worker.join(timeout=1.0)


def update_params(config_cls):
    config_cls.PARALLEL = rospy.get_param("~parallel", config_cls.PARALLEL)
    config_cls.FOREGROUND_ONLY = rospy.get_param("~foreground_only", config_cls.FOREGROUND_ONLY)
    config_cls.BOOST_CLIP = rospy.get_param("~boost_clip", config_cls.BOOST_CLIP)
    config_cls.RUN_SAM = rospy.get_param("~run_sam", config_cls.RUN_SAM)
    config_cls.SAM_CONFIDENCE = rospy.get_param("~sam_confidence", config_cls.SAM_CONFIDENCE)
    config_cls.SAM_IOU = rospy.get_param("~sam_iou", config_cls.SAM_IOU)
    config_cls.SAM_IMAGE_SIZE = rospy.get_param("~sam_image_size", config_cls.SAM_IMAGE_SIZE)
    config_cls.SAM_BATCH_SIZE = rospy.get_param("~sam_batch_size", config_cls.SAM_BATCH_SIZE)
    config_cls.SAM_ENABLE_BATCH = rospy.get_param("~sam_enable_batch", config_cls.SAM_ENABLE_BATCH)
    config_cls.MAX_ROI_SIZE = rospy.get_param("~max_roi_size", config_cls.MAX_ROI_SIZE)
    config_cls.MIN_ROI_WIDTH = rospy.get_param("~min_roi_width", config_cls.MIN_ROI_WIDTH)
    config_cls.MIN_ROI_HEIGHT = rospy.get_param("~min_roi_height", config_cls.MIN_ROI_HEIGHT)
    config_cls.ENABLE_MASK_DEBUG = rospy.get_param("~enable_mask_debug", config_cls.ENABLE_MASK_DEBUG)


def main():
    try:
        rospy.init_node("sam_segmentation_node", anonymous=True)
        update_params(ExtractorConfig)
        node = SamRosNode()

        rospy.loginfo("SamRosNode is running...")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("SamRosNode shutdown")
    except Exception as e:
        rospy.logerr("SamRosNode failed: %s", str(e))


if __name__ == "__main__":
    main()
