import numpy as np
import rospy

from umap.msg import Obstacles, RematchStatistics

from sam_types import BackendResult, ProcessingStats


class SamRosPublisher:
    """ROS message building and publishing for SAM backend results."""

    def __init__(
        self,
        bridge,
        config,
        vis_pub,
        obstacles_pub,
        roi_vis_pub,
        mask_vis_pub,
        segmented_image_pub,
        statistics_pub,
    ):
        self.bridge = bridge
        self.config = config
        self.vis_pub = vis_pub
        self.obstacles_pub = obstacles_pub
        self.roi_vis_pub = roi_vis_pub
        self.mask_vis_pub = mask_vis_pub
        self.segmented_image_pub = segmented_image_pub
        self.statistics_pub = statistics_pub

    def publish(self, original_msg, result: BackendResult):
        try:
            overlay_image = result.overlay_image
            vis_image = result.vis_image
            mask_debug_vis = result.mask_debug_vis
            roi_grid = result.roi_grid
            binary_mask = result.binary_mask
            stats = result.stats

            if overlay_image is not None:
                segmented_msg = self.bridge.cv2_to_imgmsg(overlay_image, "bgr8")
                segmented_msg.header.stamp = rospy.Time.now()
                self.segmented_image_pub.publish(segmented_msg)

            if vis_image is None:
                vis_image = overlay_image
            if vis_image is not None:
                vis_msg = self.bridge.cv2_to_imgmsg(vis_image, "bgr8")
                vis_msg.header = original_msg.header
                self.vis_pub.publish(vis_msg)

            if mask_debug_vis is not None and self.config.ENABLE_MASK_DEBUG:
                mask_debug_msg = self.bridge.cv2_to_imgmsg(mask_debug_vis, "bgr8")
                mask_debug_msg.header = original_msg.header
                self.mask_vis_pub.publish(mask_debug_msg)

            if roi_grid is not None:
                roi_msg = self.bridge.cv2_to_imgmsg(roi_grid, "bgr8")
                roi_msg.header = original_msg.header
                self.roi_vis_pub.publish(roi_msg)

            if binary_mask is None:
                if overlay_image is not None:
                    binary_mask = np.zeros(overlay_image.shape[:2], dtype=np.uint8)
                else:
                    rospy.logwarn_throttle(2.0, "Binary mask is missing and no image fallback available")
                    return

            processed_obstacles = self.build_obstacles_message(original_msg, binary_mask)
            self.obstacles_pub.publish(processed_obstacles)

            stats_msg = self.build_statistics_message(original_msg, stats)
            self.statistics_pub.publish(stats_msg)

            rospy.loginfo(
                "SAM: input=%d valid=%d segmented=%d masks=%d total=%.3fs model=%.3fs pre=%.3fs post=%.3fs obstacles=%d",
                stats.input_obstacle_count,
                stats.valid_roi_count,
                stats.segmented_roi_count,
                stats.total_mask_count,
                stats_msg.total_time,
                stats_msg.model_time,
                stats_msg.preprocess_time,
                stats_msg.postprocess_time,
                stats_msg.obstacle_count,
            )

        except Exception as e:
            rospy.logerr("Result publishing failed: %s", str(e))

    def build_obstacles_message(self, original_msg, binary_mask):
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

        processed_msg.binary_mask.header = original_msg.header
        processed_msg.binary_mask.data = binary_mask.flatten().tolist()
        processed_msg.binary_mask.height = binary_mask.shape[0]
        processed_msg.binary_mask.width = binary_mask.shape[1]
        processed_msg.binary_mask.encoding = "mono8"
        processed_msg.binary_mask.is_bigendian = 0
        processed_msg.binary_mask.step = binary_mask.shape[1]

        return processed_msg

    def build_statistics_message(self, original_msg, stats: ProcessingStats):
        stats_msg = RematchStatistics()
        stats_msg.header = original_msg.header
        stats_msg.node_name = "sam"
        stats_msg.obstacle_count = stats.segmented_roi_count
        stats_msg.total_time = stats.total_time
        stats_msg.model_time = stats.sam_time
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
