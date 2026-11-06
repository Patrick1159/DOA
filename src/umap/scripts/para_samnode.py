#!/home/patrick/miniconda3/envs/fastsam/bin/python3
"""
Subscribes to dynamic obstacles and image topics, processes ROIs, and publishes results.
"""

import time
import rospy
import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from umap.msg import BoundingBoxes2D, BoundingBoxes3D, SegmentedImage, MultiImageFeatures, Obstacles, RematchStatistics

from config_manager import ExtractorConfig
from model_manager import ModelManager
from utils import ImageProcessor, ColorGenerator
from node_helpers import (
    convert_image_message,
    make_visualization_image,
    make_mask_debug_visualization,
    build_obstacles_message,
)


class PrarSamNode:
    def __init__(self):
        # Initialize configuration
        self.config = ExtractorConfig()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        rospy.loginfo(f"Using device: {self.device}")

        # Initialize components
        self.bridge = CvBridge()
        self.model_manager = ModelManager(self.config)
        self.image_processor = ImageProcessor(self.config)
        self.color_generator = ColorGenerator(self.config.MAX_COLORS)
        
        # ROS interface
        self._setup_ros_interface()
        
        # State
        self.current_image = None
        
        # Parallel results
        self.to_publish = None 

        self.current_time = time.time()
        rospy.loginfo("PrarSamNode initialized successfully")
    
    def _setup_ros_interface(self):
        """Setup ROS subscribers and publishers"""
        # Subscribers
        image_topic = rospy.get_param('~image_topic', '/camera/rgb/image_raw')
        # Visualization publishing (RViz-only) is skipped in headless experiments
        # to keep this single-threaded node from blocking on per-frame image encode.
        self.enable_visualization = rospy.get_param('~enable_visualization', True)
        # SAM consumes coarse tracks (~21Hz) slower than they are produced (~26Hz),
        # so without dropping, frames queue and e2e latency grows without bound
        # (+0.15s/s, reaching 1-2s). Skip any coarse frame already older than this
        # so SAM always works on the freshest track the planner needs. 0 disables.
        self.max_stale = rospy.get_param('~max_stale', 0.15)
        # buff_size must exceed one serialized frame (640x480x3 ~= 900KB), else
        # rospy's default 64KB socket buffer backs up and image frames accumulate
        # in the TCP buffer, starving the obstacle callback on this single thread.
        # This caused SAM e2e latency to grow ~0.1s/s without bound (stale tracks
        # ~1s behind reality). 16MB holds ~18 frames so the receiver drains faster
        # than frames arrive.
        self.image_sub = rospy.Subscriber(image_topic, Image, self._handle_image_message,
                                          queue_size=1, buff_size=2**24)
        
        self.obstacles_sub = rospy.Subscriber(
                'tracker/dynamic_obstacles', Obstacles, self._process_dynamic_obstacles, queue_size=1)
        
        # Publishers
        self.vis_pub = rospy.Publisher(
            "/umap/visualization/segmented_image", Image, queue_size=10)
        self.obstacles_pub = rospy.Publisher(
            '/model_output/obstacles_sam', Obstacles, queue_size=10)
        self.roi_vis_pub = rospy.Publisher(
            "/model_output/visualization/roi_image", Image, queue_size=10)
        self.mask_vis_pub = rospy.Publisher(
            "/model_output/visualization/mask_debug", Image, queue_size=10)
        self.segmented_image_pub = rospy.Publisher(
            '/model_output/segmented_image', Image, queue_size=10)
        self.statistics_pub = rospy.Publisher(
            '/rematch_statistics_sam', RematchStatistics, queue_size=10)
        
    def _handle_image_message(self, image_msg):
        """Handle incoming image messages"""
        img = convert_image_message(self.bridge, image_msg)
        if img is not None:
            self.current_image = img
            rospy.logdebug("Received image message")

    def _process_dynamic_obstacles(self, obstacles_msg):
        """Process dynamic obstacles and extract features"""
        # rospy.loginfo("Processing dynamic obstacles")
        
        if self.current_image is None:
            rospy.logwarn("No current image available")
            return
        
        if not obstacles_msg.box_ids:
            # rospy.logwarn("No bounding boxes received")
            return

        # Drop stale frames: if a newer coarse frame is already queued behind this
        # one, processing this one only adds latency. Keeps SAM on the freshest track.
        if self.max_stale > 0.0:
            age = (rospy.Time.now() - obstacles_msg.header.stamp).to_sec()
            if age > self.max_stale:
                rospy.logdebug(f"Dropping stale coarse frame (age={age:.3f}s > {self.max_stale}s)")
                return
        
        try:
            cb_start = time.time()
            # Extract features from all ROIs (returns overlay_image)
            extracted_features, valid_rois, masks, processing_stats, binary_mask, overlay_image = self._extract_features_from_obstacles(
                obstacles_msg)

            # Visualization images cost per-frame cv2_to_imgmsg on 900KB frames but
            # are only consumed by RViz. In headless experiments nobody subscribes,
            # so building/publishing them only blocks this single-threaded node and
            # adds latency to the tracker the planner actually consumes.
            t_vis = time.time()
            if self.enable_visualization:
                vis_image = make_visualization_image(
                    self.current_image, obstacles_msg, valid_rois,
                    self.color_generator, self.config)
                mask_debug_vis = make_mask_debug_visualization(
                    self.image_processor, self.current_image, obstacles_msg, masks)
            else:
                vis_image = None
                mask_debug_vis = None
                overlay_image = None
            t_vis = time.time() - t_vis

            # Publish all results
            t_pub = time.time()
            self._publish_results(obstacles_msg=obstacles_msg, features=extracted_features,
                                valid_rois=valid_rois, vis_image=vis_image, mask_debug_vis=mask_debug_vis,
                                processing_stats=processing_stats, binary_mask=binary_mask, overlay_image=overlay_image)
            t_pub = time.time() - t_pub
            rospy.loginfo(f"SAM cb: extract+vis+pub total={time.time()-cb_start:.3f}s "
                          f"(sam={processing_stats.get('total_time', 0):.3f} vis={t_vis:.3f} pub={t_pub:.3f})")

        except Exception as e:
            rospy.logerr(f"Obstacle processing failed: {e}")
    
    def _extract_features_from_obstacles(self, obstacles_msg):
        """Extract features from all obstacle ROIs (batch FastSAM processing)"""
        rospy.loginfo(f"Time from last processing: {time.time() - self.current_time:.3f}s")
        self.current_time = time.time()
        
        features = []
        valid_rois = []
        masks = []  # Collect masks for debug visualization
        processing_stats = {
            'clip_time': 0.0,
            'sam_time': 0.0,
            'processed_count': 0,
            'preprocess_time': 0.0,
            'postprocess_time': 0.0,
        }
        
        rospy.loginfo(f"Processing {len(obstacles_msg.box_ids)} obstacles")
        time_start = time.time()

        # Pre-allocate overlay image and binary mask once for all obstacles
        overlay_image = self.current_image.copy()
        binary_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
        
        # Collect all SAM results for batch overlay processing
        all_transformed_masks = []  # List of (color, transformed_mask)
        
        # Batch convert ROS image messages to numpy arrays (efficiency gain)
        mask_conversion_start = time.time()
        converted_masks = []
        for i in range(len(obstacles_msg.box_ids)):
            try:
                mask = self.bridge.imgmsg_to_cv2(obstacles_msg.mask[i], "passthrough")
                converted_masks.append(mask)
            except Exception as e:
                rospy.logerr(f"Failed to convert mask {i}: {e}")
                converted_masks.append(None)
        rospy.logdebug(f"Batch mask conversion: {time.time() - mask_conversion_start:.3f}s")

        # Phase 1: Extract and validate ROIs
        roi_extraction_data = []  # List of (index, box_id, roi, mask)
        for i, box_id in enumerate(obstacles_msg.box_ids):
            try:
                mask = converted_masks[i]
                if mask is None:
                    masks.append(np.zeros((1, 1), dtype=np.uint8))
                    continue
                    
                masks.append(mask)  # Collect mask for visualization
                
                roi = self.image_processor.extract_roi_from_mask(
                    self.current_image, mask, self.config.FOREGROUND_ONLY)
                
                # Validate ROI
                if not self.image_processor.is_valid_roi(roi):
                    rospy.logdebug(f"Invalid ROI for box_id {box_id}")
                    continue
                
                roi_extraction_data.append((i, box_id, roi, mask))
                valid_rois.append(roi)
                
            except Exception as e:
                rospy.logerr(f"ROI extraction failed for box_id {box_id}: {e}")
                continue

        processing_stats['preprocess_time'] = time.time() - time_start

        # Phase 2: Batch FastSAM processing (GPU parallel)
        sam_results_dict = {}  # box_id -> results mapping
        
        if roi_extraction_data and self.config.RUN_SAM:
            batch_rois = [data[2] for data in roi_extraction_data]
            batch_box_ids = [data[1] for data in roi_extraction_data]
            
            # Try batch processing first
            sam_infer_start = time.time()
            batch_results = self.model_manager.run_fastsam_batch(batch_rois, batch_box_ids)
            processing_stats['sam_time'] = time.time() - sam_infer_start
            
            if batch_results is not None:
                # Batch processing succeeded
                for box_id, results in batch_results:
                    if results is not None and len(results) > 0 and results[0] is not None:
                        sam_results_dict[box_id] = results
                        if hasattr(results[0], 'masks') and results[0].masks is not None:
                            processing_stats['processed_count'] += 1
                            rospy.logdebug(f"Batch SAM result for box_id {box_id}: {results[0].masks.data.shape[0]} masks")
            else:
                # Fall back to sequential processing
                rospy.loginfo("Batch processing disabled or failed, using sequential processing")
                for i, box_id, roi, mask in roi_extraction_data:
                    sam_masks = self._process_with_sam(roi, box_id, processing_stats)
                    if sam_masks.shape[0] > 0:
                        sam_results_dict[box_id] = (type('Results', (), {'masks': type('Masks', (), {'data': sam_masks})()})(),)
        elif not self.config.RUN_SAM:
            # Create dummy results for all valid ROIs
            for i, box_id, roi, mask in roi_extraction_data:
                sam_masks = torch.ones((1, roi.shape[0], roi.shape[1]), dtype=torch.uint8)
                sam_results_dict[box_id] = (type('Results', (), {'masks': type('Masks', (), {'data': sam_masks})()})(),)

        # Phase 3: Transform masks and create overlays
        # **OPTIMIZATION**: Pre-allocate image_mask once and reuse
        image_mask_template = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
        
        for i, box_id, roi, mask in roi_extraction_data:
            try:
                # Get SAM results
                if box_id not in sam_results_dict:
                    continue
                
                sam_result = sam_results_dict[box_id]
                if sam_result is None:
                    continue
                
                sam_masks = sam_result[0].masks.data
                
                # Transform SAM masks from ROI coordinate system to original image coordinate system
                if sam_masks.shape[0] > 0:
                    # Get ROI bounding box in original image coordinates
                    box_x = obstacles_msg.box_x[i]
                    box_y = obstacles_msg.box_y[i]
                    box_width = obstacles_msg.box_width[i]
                    box_height = obstacles_msg.box_height[i]
                    
                    # Transform each mask from ROI to image coordinates
                    for mask_idx in range(sam_masks.shape[0]):
                        roi_mask = sam_masks[mask_idx].cpu().numpy()
                        resized_mask = cv2.resize(roi_mask.astype(np.uint8), 
                                                    (box_width, box_height),
                                                    interpolation=cv2.INTER_NEAREST)
                        
                        # **OPTIMIZATION**: Reuse template, clear only used region
                        image_mask_template.fill(0)  # Fast clear
                        image_mask_template[box_y:box_y + box_height, 
                                          box_x:box_x + box_width] = resized_mask
                        
                        # Store color and mask for batch processing
                        color = self.color_generator.get_color(mask_idx)
                        # **MUST COPY**: We reuse image_mask_template in next iteration
                        all_transformed_masks.append((color, image_mask_template.copy()))
                        
                        # Update binary mask
                        binary_mask = np.maximum(binary_mask, image_mask_template)
                    
                    rospy.logdebug(f"Processed {sam_masks.shape[0]} masks for box_id {box_id}")

            except Exception as e:
                rospy.logerr(f"Mask transformation failed for box_id {box_id}: {e}")
                continue

        # Batch blend all masks onto the image (vectorized operation)
        if all_transformed_masks:
            overlay_image = self._blend_masks_batch(overlay_image, all_transformed_masks)
        
        time_end = time.time()
        total_time = time_end - time_start
        processing_stats['postprocess_time'] = total_time - processing_stats['preprocess_time'] - processing_stats['sam_time']
        processing_stats['total_time'] = total_time
        rospy.loginfo(f"Successfully processed {processing_stats['processed_count']} obstacles in {total_time:.3f}s")

        return features, valid_rois, masks, processing_stats, binary_mask, overlay_image
    
    def _process_with_sam(self, roi, box_id, processing_stats) -> torch.Tensor:
        """Process ROI with FastSAM segmentation"""
        try:
            sam_start_time = time.time()
            results = self.model_manager.run_fastsam(roi)
            sam_time = time.time() - sam_start_time
            processing_stats['sam_time'] += sam_time
            rospy.loginfo(f"SAM processing for box_id {box_id}: {sam_time:.3f}s")

            if results is None or results[0] is None or results[0].masks is None:
                rospy.logwarn(f"SAM returned no results for box_id {box_id}")
                return torch.empty((0, 0, 0))

            # Process SAM results
            masks = results[0].masks.data  # (N, H, W) tensor
            num_masks = masks.shape[0]
            rospy.loginfo(f"SAM found {num_masks} masks for box_id {box_id}")
            return masks

        except Exception as e:
            rospy.logerr(f"SAM processing failed for box_id {box_id}: {e}")
            return torch.empty((0, 0, 0))
        
    def _blend_masks_batch(self, image, mask_color_pairs):
        """Batch blend multiple masks onto image using vectorized operations
        
        Args:
            image: Base image to blend onto
            mask_color_pairs: List of (color, mask) tuples
            
        Returns:
            Blended image
        """
        result = image.copy()
        
        # **OPTIMIZATION**: Pre-allocate colored_mask once
        alpha = 0.3
        colored_mask = np.zeros_like(image)
        
        for color, mask in mask_color_pairs:
            # **OPTIMIZATION**: Reuse colored_mask, only clear non-zero regions
            colored_mask.fill(0)  # Fast clear
            colored_mask[mask > 0] = color
            
            # Blend using cv2.addWeighted (optimized C++ backend)
            result = cv2.addWeighted(result, 1.0 - alpha, colored_mask, alpha, 0)
        
        return result
    def _create_visualization_image(self, obstacles_msg, valid_rois):
        """Backwards-compatible wrapper around node_helpers.make_visualization_image."""
        return make_visualization_image(
            self.current_image, obstacles_msg, valid_rois,
            self.color_generator, self.config)

    def _create_mask_debug_visualization(self, obstacles_msg, masks):
        """Backwards-compatible wrapper around node_helpers.make_mask_debug_visualization."""
        return make_mask_debug_visualization(
            self.image_processor, self.current_image, obstacles_msg, masks)
    
    def _publish_results(self, obstacles_msg, features, valid_rois, 
                        vis_image, mask_debug_vis, processing_stats, binary_mask, overlay_image=None):
        """Publish all processing results (merged visualization publishing)"""
        try:
            # Publish all visualizations in optimized order
            # 1. Segmented image (once, after all obstacles processed)
            if overlay_image is not None:
                segmented_msg = self.bridge.cv2_to_imgmsg(overlay_image, "bgr8")
                segmented_msg.header.stamp = rospy.Time.now()
                self.segmented_image_pub.publish(segmented_msg)
                rospy.logdebug("Published segmented image")
            
            # 2. Visualization image
            if vis_image is not None:
                vis_msg = self.bridge.cv2_to_imgmsg(vis_image, "bgr8")
                vis_msg.header = obstacles_msg.header
                self.vis_pub.publish(vis_msg)
            
            # 3. Mask debug visualization (vectorized)
            if mask_debug_vis is not None and self.config.ENABLE_MASK_DEBUG:
                mask_debug_msg = self.bridge.cv2_to_imgmsg(mask_debug_vis, "bgr8")
                mask_debug_msg.header = obstacles_msg.header
                self.mask_vis_pub.publish(mask_debug_msg)
                rospy.logdebug("Published mask debug visualization")
            
            # 4. ROI grid visualization
            if valid_rois:
                roi_grid = self.image_processor.create_roi_grid(
                    valid_rois, obstacles_msg.box_ids[:len(valid_rois)])
                if roi_grid is not None:
                    roi_msg = self.bridge.cv2_to_imgmsg(roi_grid, "bgr8")
                    roi_msg.header = obstacles_msg.header
                    self.roi_vis_pub.publish(roi_msg)
            
            # 5. Process and publish obstacles message
            processed_obstacles = self._create_obstacles_message(
                obstacles_msg, features, binary_mask)
            self.obstacles_pub.publish(processed_obstacles)
            
            # 6. Publish rematch statistics
            stats_msg = RematchStatistics()
            stats_msg.header = obstacles_msg.header
            stats_msg.node_name = 'sam'
            n = processing_stats.get('processed_count', 0)
            stats_msg.obstacle_count = n
            stats_msg.total_time = processing_stats.get('total_time', 0.0)
            stats_msg.model_time = processing_stats.get('sam_time', 0.0)
            stats_msg.preprocess_time = processing_stats.get('preprocess_time', 0.0)
            stats_msg.postprocess_time = processing_stats.get('postprocess_time', 0.0)
            stats_msg.avg_model_time_per_obstacle = (
                stats_msg.model_time / n if n > 0 else 0.0)
            stats_msg.sync_wait_time = 0.0
            # end-to-end latency: now - message stamp
            try:
                stats_msg.e2e_latency = rospy.Time.now().to_sec() - obstacles_msg.header.stamp.to_sec()
            except Exception:
                stats_msg.e2e_latency = 0.0
            self.statistics_pub.publish(stats_msg)

            # Log performance metrics
            rospy.loginfo(
                f"SAM: total={stats_msg.total_time:.3f}s  "
                f"model={stats_msg.model_time:.3f}s  "
                f"pre={stats_msg.preprocess_time:.3f}s  "
                f"post={stats_msg.postprocess_time:.3f}s  "
                f"obstacles={n}"
            )
            
        except Exception as e:
            rospy.logerr(f"Result publishing failed: {e}")
    
    def _create_obstacles_message(self, original_msg, features, binary_mask):
        """Create obstacles message with extracted features and binary mask"""
        return build_obstacles_message(original_msg, features, binary_mask=binary_mask)

def update_params(ExtractorConfig):
    """Update parameters from ROS parameter server"""
    ExtractorConfig.PARALLEL = rospy.get_param('~parallel', ExtractorConfig.PARALLEL)
    ExtractorConfig.FOREGROUND_ONLY = rospy.get_param('~foreground_only', ExtractorConfig.FOREGROUND_ONLY)
    ExtractorConfig.BOOST_CLIP = rospy.get_param('~boost_clip', ExtractorConfig.BOOST_CLIP)
    ExtractorConfig.RUN_SAM = rospy.get_param('~run_sam', ExtractorConfig.RUN_SAM)
    ExtractorConfig.SAM_CONFIDENCE = rospy.get_param('~sam_confidence', ExtractorConfig.SAM_CONFIDENCE)
    ExtractorConfig.SAM_IOU = rospy.get_param('~sam_iou', ExtractorConfig.SAM_IOU)
    ExtractorConfig.SAM_IMAGE_SIZE = rospy.get_param('~sam_image_size', ExtractorConfig.SAM_IMAGE_SIZE)
    ExtractorConfig.SAM_BATCH_SIZE = rospy.get_param('~sam_batch_size', ExtractorConfig.SAM_BATCH_SIZE)
    ExtractorConfig.SAM_ENABLE_BATCH = rospy.get_param('~sam_enable_batch', ExtractorConfig.SAM_ENABLE_BATCH)
    ExtractorConfig.MAX_ROI_SIZE = rospy.get_param('~max_roi_size', ExtractorConfig.MAX_ROI_SIZE)
    ExtractorConfig.MIN_ROI_WIDTH = rospy.get_param('~min_roi_width', ExtractorConfig.MIN_ROI_WIDTH)
    ExtractorConfig.MIN_ROI_HEIGHT = rospy.get_param('~min_roi_height', ExtractorConfig.MIN_ROI_HEIGHT)
    ExtractorConfig.ENABLE_MASK_DEBUG = rospy.get_param('~enable_mask_debug', ExtractorConfig.ENABLE_MASK_DEBUG)

def main():
    """Main function"""
    try:
        rospy.init_node("feature_extractor", anonymous=True)
        update_params(ExtractorConfig)
        extractor = PrarSamNode()
        
        rospy.loginfo("PrarSamNode node is running...")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("PrarSamNode node shutdown")
    except Exception as e:
        rospy.logerr(f"FeatureExtPrarSamNoderactor node failed: {e}")


if __name__ == '__main__':
    main()