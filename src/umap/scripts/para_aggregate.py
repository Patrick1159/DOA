#!/home/patrick/miniconda3/envs/ros-noetic/bin/python3
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
from umap.msg import Obstacles
import message_filters

from config_manager import ExtractorConfig
from model_manager import ModelManager
from utils import ImageProcessor, ColorGenerator

class ParaMsgAggregator:
    """Aggregate synchronized SAM and CLIP messages for ParaCLIP processing"""

    def __init__(self):
        # Tunable synchronization parameters
        sync_slop = rospy.get_param("~sync_slop", 0.5)  # seconds tolerance for approximate sync
        sync_queue = rospy.get_param("~sync_queue", 20)  # queue size for message filters

        # Publisher
        self.obstacle_pub = rospy.Publisher('/model_output/obstacles', Obstacles, queue_size=10)

        # Subscribers (message_filters for time sync)
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

    def _synced_callback(self, sam_msg: Obstacles, clip_msg: Obstacles):
        """Callback when SAM and CLIP messages are time-synchronized"""
        aggregated_msg = Obstacles()
        aggregated_msg.header = sam_msg.header  # use SAM header as base

        for attr in [
            'box_ids', 'centroid_x', 'centroid_y', 'centroid_z',
            'thickness', 'height', 'width', 'box_x', 'box_y',
            'box_width', 'box_height'
        ]:
            setattr(aggregated_msg, attr, getattr(sam_msg, attr))

        aggregated_msg.binary_mask = sam_msg.binary_mask

        # Copy CLIP features
        aggregated_msg.features = clip_msg.features

        # Publish once per synchronized pair
        self.obstacle_pub.publish(aggregated_msg)
        rospy.loginfo("Published aggregated obstacles message (synced SAM+CLIP)")

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
        