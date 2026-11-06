#!/usr/bin/env python3

"""
Subscribe CLIP and SAM messages, publish full /model_output/obstacles message.

This version uses ROS message_filters to time-synchronize SAM and CLIP messages
so each pair is aggregated and published exactly once, avoiding races and
mismatched frames.
"""
import time
import rospy
import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from umap.msg import Obstacles, RematchStatistics
import message_filters
from scipy.spatial.transform import Rotation as R

from config_manager import ExtractorConfig
from model_manager import ModelManager
from utils import ImageProcessor, ColorGenerator

class ParaMsgAggregator:
    """Aggregate synchronized SAM and CLIP messages for ParaCLIP processing.

    Two modes:
      - passthrough_clip=false (default): ApproximateTimeSynchronizer on SAM+CLIP.
        Both must arrive within sync_slop for a frame to be published.
      - passthrough_clip=true: SAM-triggered mode. CLIP messages are cached
        on arrival; each SAM message immediately triggers publication with the
        most recently cached CLIP features.  Use when CLIP is in dummy mode
        (run_clip=false) so the aggregator does not wait on a stream whose
        features are not consumed by detailed_match anyway.
      - passthrough_sam=true: CLIP-triggered mode. Each CLIP message is
        published with its own geometry/features, using an empty CLIP mask.
        Use when SAM is disabled so no cached dummy-SAM frame can become the
        geometry source.
    """

    def __init__(self):
        # Tunable synchronization parameters
        sync_slop = rospy.get_param("~sync_slop", 0.5)  # seconds tolerance for approximate sync
        sync_queue = rospy.get_param("~sync_queue", 20)  # queue size for message filters
        self.passthrough_clip = rospy.get_param("~passthrough_clip", False)
        self.passthrough_sam = rospy.get_param("~passthrough_sam", False)
        if self.passthrough_clip and self.passthrough_sam:
            rospy.logwarn(
                "Both passthrough_clip and passthrough_sam requested; "
                "using passthrough_clip and ignoring passthrough_sam."
            )
            self.passthrough_sam = False

        # Initialize transformation matrix and pose status
        self.cam_to_world = np.eye(4)
        self.has_pose = False

        # Publisher
        self.obstacle_pub = rospy.Publisher('/model_output/obstacles', Obstacles, queue_size=10)
        self.statistics_pub = rospy.Publisher('/rematch_statistics_aggregator', RematchStatistics, queue_size=10)

        # Subscribers
        cam_pose_sub = rospy.Subscriber('/camera/current_pose', PoseStamped, self.pose_callback)

        if self.passthrough_clip:
            # Passthrough mode: SAM triggers publication; CLIP is only cached.
            self._cached_clip_msg = None
            self._sam_sub = rospy.Subscriber('/model_output/obstacles_sam', Obstacles,
                                             self._sam_passthrough_callback)
            self._clip_sub = rospy.Subscriber('/model_output/obstacles_clip', Obstacles,
                                              self._clip_cache_callback)
            rospy.loginfo("ParaMsgAggregator initialized in passthrough_clip mode "
                          "(SAM-triggered, CLIP cached).")
        elif self.passthrough_sam:
            # Passthrough mode: CLIP triggers publication and supplies both
            # geometry and appearance features. No SAM stream is consumed.
            self._clip_sub = rospy.Subscriber('/model_output/obstacles_clip', Obstacles,
                                              self._clip_passthrough_callback)
            rospy.loginfo("ParaMsgAggregator initialized in passthrough_sam mode "
                          "(CLIP-triggered, CLIP geometry).")
        else:
            # Time-synchronized mode (existing behaviour)
            sam_sub = message_filters.Subscriber('/model_output/obstacles_sam', Obstacles)
            clip_sub = message_filters.Subscriber('/model_output/obstacles_clip', Obstacles)
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [sam_sub, clip_sub], queue_size=sync_queue, slop=sync_slop, allow_headerless=False
            )
            self.ts.registerCallback(self._synced_callback)
            rospy.loginfo(
                f"ParaMsgAggregator initialized with ApproximateTimeSynchronizer "
                f"(queue={sync_queue}, slop={sync_slop}s)."
            )

    def pose_callback(self, msg: PoseStamped):
        """Subcribe camera pose topic and compute cam_to_world transformation matrix"""
        self.cam_to_world = np.eye(4)
        rotx = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])

        quat = [msg.pose.orientation.x, msg.pose.orientation.y, 
                msg.pose.orientation.z, msg.pose.orientation.w]
        rotation_matrix = R.from_quat(quat).as_matrix()

        final_rotation = rotation_matrix @ rotx
        
        # Set rotation part (3x3)
        self.cam_to_world[:3, :3] = final_rotation
        
        # Set translation part (3x1)
        self.cam_to_world[0, 3] = msg.pose.position.x
        self.cam_to_world[1, 3] = msg.pose.position.y
        self.cam_to_world[2, 3] = msg.pose.position.z

        self.has_pose = True

        # rospy.loginfo_throttle(1.0, "Updated camera pose transformation matrix")
        # rospy.loginfo(f"Camera to World Transformation Matrix:\n{self.cam_to_world}")

    def _clip_cache_callback(self, clip_msg: Obstacles):
        """Passthrough mode: cache the most recent CLIP message."""
        self._cached_clip_msg = clip_msg

    def _sam_cache_callback(self, sam_msg: Obstacles):
        """Passthrough mode: cache the most recent SAM message."""
        self._cached_sam_msg = sam_msg

    def _sam_passthrough_callback(self, sam_msg: Obstacles):
        """Passthrough mode: publish on SAM arrival using cached CLIP features.

        If no CLIP message has been cached yet, synthesise a minimal dummy
        CLIP message so publication can proceed immediately.
        """
        if self._cached_clip_msg is None:
            # Build a minimal CLIP message with empty features.  In
            # passthrough_clip mode the caller guarantees that CLIP features
            # are not consumed by detailed_match (SIMILARITY_THRESHOLD=-1,
            # SIMILARITY_WEIGHT=0), so dummy features are harmless.
            dummy_clip = Obstacles()
            dummy_clip.header = sam_msg.header
            dummy_clip.features = []
            self._cached_clip_msg = dummy_clip
            rospy.logwarn_throttle(5.0,
                "[Aggregator] Passthrough mode: no CLIP message cached yet, "
                "using empty features for this frame."
            )
        self._synced_callback(sam_msg, self._cached_clip_msg)

    def _clip_passthrough_callback(self, clip_msg: Obstacles):
        """Passthrough mode: publish on CLIP arrival using CLIP geometry."""
        clip_as_geometry = Obstacles()
        clip_as_geometry.header = clip_msg.header
        for attr in [
            'box_ids', 'centroid_x', 'centroid_y', 'centroid_z',
            'thickness', 'height', 'width', 'box_x', 'box_y',
            'box_width', 'box_height', 'observed'
        ]:
            setattr(clip_as_geometry, attr, getattr(clip_msg, attr))
        clip_as_geometry.binary_mask = clip_msg.binary_mask
        self._synced_callback(clip_as_geometry, clip_msg)

    def _synced_callback(self, sam_msg: Obstacles, clip_msg: Obstacles):
        """Callback when SAM and CLIP messages are time-synchronized"""
        agg_start = time.time()
        try:
            aggregated_msg = Obstacles()
            aggregated_msg.header = sam_msg.header  # use SAM header as base

            for attr in [
                'box_ids', 'centroid_x', 'centroid_y', 'centroid_z',
                'thickness', 'height', 'width', 'box_x', 'box_y',
                'box_width', 'box_height', 'observed'
            ]:
                setattr(aggregated_msg, attr, getattr(sam_msg, attr))
            
            # Transform centroids from camera frame to world frame
            # aggregated_msg.centroid_x = []
            # aggregated_msg.centroid_y = []
            # aggregated_msg.centroid_z = []
            # for _ in range(len(sam_msg.centroid_x)):
            #     obstacle_xyz_camera_frame = np.array(
            #         [sam_msg.centroid_x[_], 
            #         sam_msg.centroid_y[_], 
            #         sam_msg.centroid_z[_]]).reshape(-1, 3)
            #     obstacle_xyz_world_frame = self.cam_to_world[:3, :3] @ obstacle_xyz_camera_frame.T + self.cam_to_world[:3, 3:4]
            #     rospy.loginfo(f"[Aggregator] Transformed obstacle centroid to world frame: {obstacle_xyz_world_frame.flatten()} from camera frame: {obstacle_xyz_camera_frame.flatten()}")
            #     rospy.loginfo(f"[AGG DEBUG] msg.centroid: {aggregated_msg.centroid_x}, type: {type(aggregated_msg.centroid_x)}")
            #     aggregated_msg.centroid_x.append(obstacle_xyz_world_frame[0, 0])
            #     aggregated_msg.centroid_y.append(obstacle_xyz_world_frame[1, 0])
            #     aggregated_msg.centroid_z.append(obstacle_xyz_world_frame[2, 0])
            # aggregated_msg.centroid_x = tuple(aggregated_msg.centroid_x)
            # aggregated_msg.centroid_y = tuple(aggregated_msg.centroid_y)
            # aggregated_msg.centroid_z = tuple(aggregated_msg.centroid_z)

            # NOTE: do NOT overwrite header.frame_id here. Upstream
            # coarse_match writes data + frame_id consistently (typically
            # `odom` after the frame-honest refactor). Overwriting it
            # without transforming centroid_* would produce the
            # "frame_id-says-odom-but-data-is-in-d435_link" lie that broke
            # downstream TF logic.
            aggregated_msg.binary_mask = sam_msg.binary_mask

            # Copy CLIP features
            aggregated_msg.features = clip_msg.features

            # Publish once per synchronized pair
            self.obstacle_pub.publish(aggregated_msg)
            rospy.loginfo("Published aggregated obstacles message (synced SAM+CLIP)")

            # Publish rematch statistics
            agg_total = time.time() - agg_start
            # Sync wait time: difference between the two message stamps
            sam_t = sam_msg.header.stamp.to_sec()
            clip_t = clip_msg.header.stamp.to_sec()
            sync_wait = abs(sam_t - clip_t)

            stats_msg = RematchStatistics()
            stats_msg.header = aggregated_msg.header
            stats_msg.node_name = 'aggregator'
            stats_msg.obstacle_count = len(list(sam_msg.box_ids))
            stats_msg.total_time = agg_total
            stats_msg.model_time = 0.0
            stats_msg.preprocess_time = 0.0
            stats_msg.postprocess_time = agg_total
            stats_msg.avg_model_time_per_obstacle = 0.0
            stats_msg.sync_wait_time = sync_wait
            # end-to-end latency: now - aggregated message stamp
            try:
                stats_msg.e2e_latency = rospy.Time.now().to_sec() - aggregated_msg.header.stamp.to_sec()
            except Exception:
                stats_msg.e2e_latency = 0.0
            self.statistics_pub.publish(stats_msg)
            rospy.loginfo(
                f"Aggregator: total={agg_total*1000:.1f}ms  sync_wait={sync_wait*1000:.1f}ms  obstacles={stats_msg.obstacle_count}"
            )
        except Exception as e:
            rospy.logerr(f"[Aggregator] Error in synchronized callback: {e}")

def main():
    rospy.init_node('para_msg_aggregator', anonymous=True)
    aggregator = ParaMsgAggregator()
    rospy.loginfo("ParaMsgAggregator node started, listening for SAM and CLIP messages...")
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException as e:
        rospy.logerr(f"ROS Interrupt Exception: {e}")
        
