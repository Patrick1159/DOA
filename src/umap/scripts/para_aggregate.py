#!/home/patrick/miniconda3/envs/fastsam/bin/python3
"""
Subscribe CLIP and SAM messages, publish full /model_output/obstacles message.

This version uses ROS message_filters to time-synchronize SAM and CLIP messages
so each pair is aggregated and published exactly once, avoiding races and
mismatched frames.
"""
import time
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from umap.msg import Obstacles, RematchStatistics
import message_filters
from scipy.spatial.transform import Rotation as R

class ParaMsgAggregator:
    """Aggregate synchronized SAM and CLIP messages for ParaCLIP processing"""

    def __init__(self):
        # Tunable synchronization parameters
        sync_slop = rospy.get_param("~sync_slop", 0.5)  # seconds tolerance for approximate sync
        sync_queue = rospy.get_param("~sync_queue", 20)  # queue size for message filters

        # Initialize transformation matrix and pose status
        self.cam_to_world = np.eye(4)
        self.has_pose = False

        # Publisher
        self.obstacle_pub = rospy.Publisher('/model_output/obstacles', Obstacles, queue_size=10)
        self.statistics_pub = rospy.Publisher('/rematch_statistics_aggregator', RematchStatistics, queue_size=10)

        # Latest upstream stats for aggregation
        self.latest_sam_stats = None
        self.latest_clip_stats = None
        self.sam_stats_sub = rospy.Subscriber('/rematch_statistics_sam', RematchStatistics, self._sam_stats_callback)
        self.clip_stats_sub = rospy.Subscriber('/rematch_statistics_clip', RematchStatistics, self._clip_stats_callback)

        # Subscribers (message_filters for time sync)
        cam_pose_sub = rospy.Subscriber('/camera/current_pose', PoseStamped, self.pose_callback)

        sam_sub = message_filters.Subscriber('/model_output/obstacles_sam', Obstacles)
        clip_sub = message_filters.Subscriber('/model_output/obstacles_clip', Obstacles)

        # Approximate time synchronizer to handle rate differences and small delays
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sam_sub, clip_sub], queue_size=sync_queue, slop=sync_slop, allow_headerless=False
        )
        self.ts.registerCallback(self._synced_callback)

        rospy.loginfo(
            f"ParaMsgAggregator initialized with ApproximateTimeSynchronizer (queue={sync_queue}, slop={sync_slop}s)."
        )

    def _sam_stats_callback(self, msg: RematchStatistics):
        self.latest_sam_stats = msg

    def _clip_stats_callback(self, msg: RematchStatistics):
        self.latest_clip_stats = msg

    def _build_clip_feature_map(self, clip_msg: Obstacles):
        """Build box_id -> Float32MultiArray map from CLIP output with alignment safeguards."""
        feature_map = {}
        if len(clip_msg.features) != len(clip_msg.box_ids):
            rospy.logwarn_throttle(
                2.0,
                "CLIP features length mismatch with box_ids, truncating by min length",
            )

        n = min(len(clip_msg.features), len(clip_msg.box_ids))
        for i in range(n):
            feature_map[int(clip_msg.box_ids[i])] = clip_msg.features[i]
        return feature_map

    @staticmethod
    def _safe_to_sec(stamp):
        try:
            return stamp.to_sec()
        except Exception:
            return None

    def _combine_upstream_stats(self, sam_msg: Obstacles, clip_msg: Obstacles):
        """Fetch latest SAM/CLIP stats if close in time to current synced pair."""
        max_delta = 0.5
        sam_ref = self._safe_to_sec(sam_msg.header.stamp)
        clip_ref = self._safe_to_sec(clip_msg.header.stamp)

        sam_stats = None
        if self.latest_sam_stats is not None and sam_ref is not None:
            sam_stats_t = self._safe_to_sec(self.latest_sam_stats.header.stamp)
            if sam_stats_t is not None and abs(sam_stats_t - sam_ref) <= max_delta:
                sam_stats = self.latest_sam_stats

        clip_stats = None
        if self.latest_clip_stats is not None and clip_ref is not None:
            clip_stats_t = self._safe_to_sec(self.latest_clip_stats.header.stamp)
            if clip_stats_t is not None and abs(clip_stats_t - clip_ref) <= max_delta:
                clip_stats = self.latest_clip_stats

        return sam_stats, clip_stats

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

            aggregated_msg.header.frame_id = "odom"

            aggregated_msg.binary_mask = sam_msg.binary_mask

            # Merge CLIP features by box_id, aligned with SAM obstacle ordering
            clip_feature_map = self._build_clip_feature_map(clip_msg)
            aggregated_msg.features = []
            for box_id in sam_msg.box_ids:
                feature_msg = clip_feature_map.get(int(box_id))
                if feature_msg is None:
                    feature_msg = Float32MultiArray()
                    feature_msg.data = []
                aggregated_msg.features.append(feature_msg)

            # Publish once per synchronized pair
            self.obstacle_pub.publish(aggregated_msg)
            rospy.loginfo("Published aggregated obstacles message (synced SAM+CLIP)")

            # Publish rematch statistics
            agg_total = time.time() - agg_start
            # Sync wait time: difference between the two message stamps
            sam_t = sam_msg.header.stamp.to_sec()
            clip_t = clip_msg.header.stamp.to_sec()
            sync_wait = abs(sam_t - clip_t)

            sam_stats, clip_stats = self._combine_upstream_stats(sam_msg, clip_msg)
            sam_model = sam_stats.model_time if sam_stats is not None else 0.0
            sam_pre = sam_stats.preprocess_time if sam_stats is not None else 0.0
            sam_post = sam_stats.postprocess_time if sam_stats is not None else 0.0
            sam_total = sam_stats.total_time if sam_stats is not None else 0.0

            clip_model = clip_stats.model_time if clip_stats is not None else 0.0
            clip_pre = clip_stats.preprocess_time if clip_stats is not None else 0.0
            clip_post = clip_stats.postprocess_time if clip_stats is not None else 0.0
            clip_total = clip_stats.total_time if clip_stats is not None else 0.0

            # Aggregator overhead is accounted as postprocess in merged pipeline stats
            merged_model = sam_model + clip_model
            merged_pre = sam_pre + clip_pre
            merged_post = sam_post + clip_post + agg_total
            merged_total = sam_total + clip_total + agg_total

            stats_msg = RematchStatistics()
            stats_msg.header = aggregated_msg.header
            stats_msg.node_name = 'aggregator'
            stats_msg.obstacle_count = len(list(sam_msg.box_ids))
            stats_msg.total_time = merged_total
            stats_msg.model_time = merged_model
            stats_msg.preprocess_time = merged_pre
            stats_msg.postprocess_time = merged_post
            stats_msg.avg_model_time_per_obstacle = (
                merged_model / stats_msg.obstacle_count if stats_msg.obstacle_count > 0 else 0.0
            )
            stats_msg.sync_wait_time = sync_wait
            # end-to-end latency: now - aggregated message stamp
            try:
                stats_msg.e2e_latency = rospy.Time.now().to_sec() - aggregated_msg.header.stamp.to_sec()
            except Exception:
                stats_msg.e2e_latency = 0.0
            self.statistics_pub.publish(stats_msg)
            rospy.loginfo(
                f"Aggregator: agg_only={agg_total*1000:.1f}ms  merged_total={merged_total*1000:.1f}ms  "
                f"sync_wait={sync_wait*1000:.1f}ms  obstacles={stats_msg.obstacle_count}"
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
        