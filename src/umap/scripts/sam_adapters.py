from typing import List, Optional

import numpy as np
import rospy

from sam_types import BackendRequest


class SamInputAdapter:
    """Convert ROS obstacle/image data into backend-native request types."""

    def __init__(self, bridge):
        self.bridge = bridge

    def build_request(self, image, obstacles_msg) -> BackendRequest:
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

        return BackendRequest(
            image=image,
            box_ids=box_ids,
            boxes=boxes,
            masks=masks,
            timestamp=timestamp,
        )

    def _convert_ros_masks(self, mask_msgs, expected_count: int) -> List[Optional[np.ndarray]]:
        converted_masks: List[Optional[np.ndarray]] = []
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
