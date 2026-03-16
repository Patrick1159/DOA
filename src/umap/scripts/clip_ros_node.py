#!/home/patrick/miniconda3/envs/fastsam/bin/python3
"""ROS adapter node for CLIP feature extraction on dynamic obstacles."""

import logging

import message_filters
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from umap.msg import Obstacles, RematchStatistics

from clip_backend import ClipBackend
from clip_types import ClipBackendRequest
from config_manager import ExtractorConfig
from model_manager import ModelManager
from utils import ImageProcessor


class ClipRosNode:
    def __init__(self):
        self.config = ExtractorConfig()

        self.bridge = CvBridge()
        self.model_manager = ModelManager(self.config)
        self.image_processor = ImageProcessor(self.config)

        self.backend = ClipBackend(
            config=self.config,
            model_manager=self.model_manager,
            image_processor=self.image_processor,
            logger=logging.getLogger(__name__),
        )

        self.sync_queue = rospy.get_param("~sync_queue", 15)
        self.sync_slop = rospy.get_param("~sync_slop", 0.2)

        self._setup_ros_interface()
        rospy.loginfo("ClipRosNode initialized successfully")

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

        self.obstacles_pub = rospy.Publisher("/model_output/obstacles_clip", Obstacles, queue_size=10)
        self.statistics_pub = rospy.Publisher("/rematch_statistics_clip", RematchStatistics, queue_size=10)

    def _synced_callback(self, image_msg, obstacles_msg):
        if not obstacles_msg.box_ids:
            return

        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            request = self._build_request(image, obstacles_msg)
            result = self.backend.process(request)
            self._publish_results(obstacles_msg, result)
        except Exception as e:
            rospy.logerr("Synchronized CLIP processing failed: %s", str(e))

    def _build_request(self, image, obstacles_msg) -> ClipBackendRequest:
        lengths = [
            len(obstacles_msg.box_ids),
            len(obstacles_msg.box_x),
            len(obstacles_msg.box_y),
            len(obstacles_msg.box_width),
            len(obstacles_msg.box_height),
            len(obstacles_msg.mask),
        ]
        obstacle_count = min(lengths) if lengths else 0
        if obstacle_count < len(obstacles_msg.box_ids):
            rospy.logwarn_throttle(
                2.0,
                "Obstacle fields length mismatch detected, truncating to aligned length",
            )

        box_ids = [int(v) for v in obstacles_msg.box_ids[:obstacle_count]]
        boxes = [
            (
                int(obstacles_msg.box_x[i]),
                int(obstacles_msg.box_y[i]),
                int(obstacles_msg.box_width[i]),
                int(obstacles_msg.box_height[i]),
            )
            for i in range(obstacle_count)
        ]
        masks = self._convert_ros_masks(obstacles_msg.mask, obstacle_count)

        try:
            timestamp = obstacles_msg.header.stamp.to_sec()
        except Exception:
            timestamp = None

        return ClipBackendRequest(
            image=image,
            box_ids=box_ids,
            boxes=boxes,
            masks=masks,
            timestamp=timestamp,
        )

    def _convert_ros_masks(self, mask_msgs, expected_count):
        converted_masks = []
        for i in range(expected_count):
            if i >= len(mask_msgs):
                converted_masks.append(None)
                continue
            try:
                converted_masks.append(self.bridge.imgmsg_to_cv2(mask_msgs[i], "passthrough"))
            except Exception as e:
                rospy.logerr("Failed to convert mask %d: %s", i, str(e))
                converted_masks.append(None)
        return converted_masks

    def _publish_results(self, obstacles_msg, result):
        try:
            processed_obstacles = self._create_obstacles_message(
                obstacles_msg,
                result.features_aligned,
                result.binary_mask,
            )
            self.obstacles_pub.publish(processed_obstacles)

            stats_msg = self._create_statistics_message(obstacles_msg, result.stats)
            self.statistics_pub.publish(stats_msg)

            rospy.loginfo(
                "CLIP: input=%d valid=%d processed=%d total=%.3fs model=%.3fs pre=%.3fs post=%.3fs obstacles=%d",
                result.stats.input_obstacle_count,
                result.stats.valid_roi_count,
                result.stats.processed_count,
                stats_msg.total_time,
                stats_msg.model_time,
                stats_msg.preprocess_time,
                stats_msg.postprocess_time,
                stats_msg.obstacle_count,
            )
        except Exception as e:
            rospy.logerr("Result publishing failed: %s", str(e))

    def _create_obstacles_message(self, original_msg, features_aligned, binary_mask):
        processed_msg = Obstacles()
        processed_msg.header = original_msg.header

        for attr in [
            "box_ids",
            "centroid_x",
            "centroid_y",
            "centroid_z",
            "thickness",
            "height",
            "width",
            "box_x",
            "box_y",
            "box_width",
            "box_height",
            "observed",
        ]:
            if hasattr(original_msg, attr):
                setattr(processed_msg, attr, getattr(original_msg, attr))

        processed_msg.features = []
        for feature in features_aligned:
            feature_msg = Float32MultiArray()
            feature_msg.data = feature.flatten().tolist() if feature is not None else []
            processed_msg.features.append(feature_msg)

        processed_msg.binary_mask.header = original_msg.header
        processed_msg.binary_mask.data = binary_mask.flatten().tolist()
        processed_msg.binary_mask.height = binary_mask.shape[0]
        processed_msg.binary_mask.width = binary_mask.shape[1]
        processed_msg.binary_mask.encoding = "mono8"
        processed_msg.binary_mask.is_bigendian = 0
        processed_msg.binary_mask.step = binary_mask.shape[1]

        return processed_msg

    def _create_statistics_message(self, original_msg, stats):
        stats_msg = RematchStatistics()
        stats_msg.header = original_msg.header
        stats_msg.node_name = "clip"
        stats_msg.obstacle_count = stats.processed_count
        stats_msg.total_time = stats.total_time
        stats_msg.model_time = stats.clip_time
        stats_msg.preprocess_time = stats.preprocess_time
        stats_msg.postprocess_time = stats.postprocess_time
        stats_msg.avg_model_time_per_obstacle = (
            stats_msg.model_time / stats_msg.obstacle_count if stats_msg.obstacle_count > 0 else 0.0
        )
        stats_msg.sync_wait_time = 0.0
        try:
            stats_msg.e2e_latency = rospy.Time.now().to_sec() - original_msg.header.stamp.to_sec()
        except Exception:
            stats_msg.e2e_latency = 0.0
        return stats_msg


def update_params(config_cls):
    config_cls.PARALLEL = rospy.get_param("~parallel", config_cls.PARALLEL)
    config_cls.FOREGROUND_ONLY = rospy.get_param("~foreground_only", config_cls.FOREGROUND_ONLY)
    config_cls.BOOST_CLIP = rospy.get_param("~boost_clip", config_cls.BOOST_CLIP)
    config_cls.RUN_SAM = rospy.get_param("~run_sam", config_cls.RUN_SAM)
    config_cls.RUN_CLIP = rospy.get_param("~run_clip", config_cls.RUN_CLIP)
    config_cls.SAM_CONFIDENCE = rospy.get_param("~sam_confidence", config_cls.SAM_CONFIDENCE)
    config_cls.SAM_IOU = rospy.get_param("~sam_iou", config_cls.SAM_IOU)
    config_cls.SAM_IMAGE_SIZE = rospy.get_param("~sam_image_size", config_cls.SAM_IMAGE_SIZE)
    config_cls.MAX_ROI_SIZE = rospy.get_param("~max_roi_size", config_cls.MAX_ROI_SIZE)
    config_cls.MIN_ROI_WIDTH = rospy.get_param("~min_roi_width", config_cls.MIN_ROI_WIDTH)
    config_cls.MIN_ROI_HEIGHT = rospy.get_param("~min_roi_height", config_cls.MIN_ROI_HEIGHT)
    config_cls.ENABLE_MASK_DEBUG = rospy.get_param("~enable_mask_debug", config_cls.ENABLE_MASK_DEBUG)


def main():
    try:
        rospy.init_node("clip_feature_node", anonymous=True)
        update_params(ExtractorConfig)
        node = ClipRosNode()

        rospy.loginfo("ClipRosNode is running...")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ClipRosNode shutdown")
    except Exception as e:
        rospy.logerr("ClipRosNode failed: %s", str(e))


if __name__ == "__main__":
    main()
