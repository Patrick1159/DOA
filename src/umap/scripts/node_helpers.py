"""Shared helpers for the umap feature extraction nodes
(feature_extract.py, para_clipnode.py, para_samnode.py).

Only functions that were *literally duplicated* across all three nodes are
factored here. Per-node concerns (custom param lists, model-specific batch
logic, distinct publisher topology) stay in each node.
"""

import cv2
import rospy
from std_msgs.msg import Float32MultiArray
from umap.msg import Obstacles


# Fields that are copied verbatim from the input Obstacles message into the
# downstream Obstacles message. Identical in all three nodes.
_OBSTACLE_FIELDS = (
    'box_ids', 'centroid_x', 'centroid_y', 'centroid_z',
    'thickness', 'height', 'width',
    'box_x', 'box_y', 'box_width', 'box_height',
)


def convert_image_message(bridge, image_msg):
    """Decode a sensor_msgs/Image into a BGR numpy array.

    Returns the image, or None on failure (and logs the error). Callers should
    assign the result to their `current_image` attribute.
    """
    try:
        return bridge.imgmsg_to_cv2(image_msg, "bgr8")
    except Exception as e:
        rospy.logerr(f"Image conversion failed: {e}")
        return None


def make_visualization_image(current_image, obstacles_msg, valid_rois,
                             color_generator, config):
    """Draw bounding boxes + ID labels onto a copy of `current_image`.

    Returns the annotated image; on failure returns a clean copy of the input.
    """
    vis_image = current_image.copy()
    try:
        for i, box_id in enumerate(obstacles_msg.box_ids[:len(valid_rois)]):
            box_x = obstacles_msg.box_x[i]
            box_y = obstacles_msg.box_y[i]
            box_width = obstacles_msg.box_width[i]
            box_height = obstacles_msg.box_height[i]

            color = color_generator.get_color(i)
            cv2.rectangle(vis_image,
                          (box_x, box_y),
                          (box_x + box_width, box_y + box_height),
                          color, 2)
            cv2.putText(vis_image, f"ID:{box_id}",
                        (box_x, box_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        config.FONT_SCALE,
                        color,
                        config.FONT_THICKNESS)
        return vis_image
    except Exception as e:
        rospy.logerr(f"Visualization creation failed: {e}")
        return current_image.copy()


def make_mask_debug_visualization(image_processor, current_image,
                                  obstacles_msg, masks):
    """Render the mask-debug visualization using the shared ImageProcessor.

    Returns None on failure (caller should skip publishing in that case).
    """
    try:
        boxes = []
        for i in range(len(obstacles_msg.box_ids)):
            boxes.append((obstacles_msg.box_x[i], obstacles_msg.box_y[i],
                          obstacles_msg.box_width[i], obstacles_msg.box_height[i]))
        return image_processor.create_mask_visualization(
            current_image, masks, obstacles_msg.box_ids, boxes)
    except Exception as e:
        rospy.logerr(f"Mask debug visualization creation failed: {e}")
        return None


def build_obstacles_message(original_msg, features, binary_mask=None):
    """Construct an Obstacles message: copies the standard fields, attaches
    CLIP/SAM features as Float32MultiArray, and (if `binary_mask` is provided)
    fills the binary_mask sub-message.

    `binary_mask` is None for `feature_extract.py`, which does not publish it.
    """
    processed_msg = Obstacles()
    processed_msg.header = original_msg.header

    for attr in _OBSTACLE_FIELDS:
        setattr(processed_msg, attr, getattr(original_msg, attr))

    processed_msg.features = []
    for feature in features:
        feature_msg = Float32MultiArray()
        feature_msg.data = feature.flatten().tolist()
        processed_msg.features.append(feature_msg)

    if binary_mask is not None:
        processed_msg.binary_mask.header = original_msg.header
        processed_msg.binary_mask.data = binary_mask.flatten().tolist()
        processed_msg.binary_mask.height = binary_mask.shape[0]
        processed_msg.binary_mask.width = binary_mask.shape[1]
        processed_msg.binary_mask.encoding = "mono8"
        processed_msg.binary_mask.is_bigendian = 0
        processed_msg.binary_mask.step = binary_mask.shape[1]

    return processed_msg
