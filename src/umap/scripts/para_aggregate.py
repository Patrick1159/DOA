#!/home/cius/miniconda3/envs/DOA/bin/python3
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
from umap.msg import Obstacles
import message_filters
from scipy.spatial.transform import Rotation as R

from config_manager import ExtractorConfig
from model_manager import ModelManager
from utils import ImageProcessor, ColorGenerator

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
        try:
            aggregated_msg = Obstacles()
            aggregated_msg.header = sam_msg.header  # use SAM header as base

            for attr in [
                'box_ids', 'centroid_x', 'centroid_y', 'centroid_z',
                'thickness', 'height', 'width', 'box_x', 'box_y',
                'box_width', 'box_height'
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

            # Copy CLIP features
            aggregated_msg.features = clip_msg.features

            # Publish once per synchronized pair
            self.obstacle_pub.publish(aggregated_msg)
            rospy.loginfo("Published aggregated obstacles message (synced SAM+CLIP)")
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
        