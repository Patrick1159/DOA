#!/home/cius/miniconda3/envs/DOA/bin/python3
"""
Feature extraction node for dynamic obstacle tracking.
Subscribes to dynamic obstacles and RGB images, extracts CLIP features and SAM segmentation results.
"""

import time
import rospy
import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray
from umap.msg import BoundingBoxes2D, BoundingBoxes3D, SegmentedImage, MultiImageFeatures, Obstacles

from config_manager import ExtractorConfig
from model_manager import ModelManager
from utils import ImageProcessor, ColorGenerator


class FeatureExtractor:
    """Main ROS node for feature extraction and image segmentation"""
    
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
        
        rospy.loginfo("FeatureExtractor initialized successfully")
    
    def _setup_ros_interface(self):
        """Setup ROS subscribers and publishers"""
        # Subscribers
        image_topic = rospy.get_param('~image_topic', '/camera/rgb/image_raw')
        self.image_sub = rospy.Subscriber(image_topic, Image, self._handle_image_message)
        self.obstacles_sub = rospy.Subscriber(
            'tracker/dynamic_obstacles', Obstacles, self._process_dynamic_obstacles)
        
        # Publishers
        self.seg_pub = rospy.Publisher(
            "/umap/segmented_boxes", BoundingBoxes2D, queue_size=10)
        self.vis_pub = rospy.Publisher(
            "/umap/visualization/segmented_image", Image, queue_size=10)
        self.feature_pub = rospy.Publisher(
            '/model_output/clip_features', MultiImageFeatures, queue_size=10)
        self.box3d_repub = rospy.Publisher(
            '/model_output/3D_bounding_boxes', BoundingBoxes3D, queue_size=10)
        self.obstacles_pub = rospy.Publisher(
            '/model_output/obstacles', Obstacles, queue_size=10)
        self.roi_vis_pub = rospy.Publisher(
            "/model_output/visualization/roi_image", Image, queue_size=10)
        self.mask_vis_pub = rospy.Publisher(
            "/model_output/visualization/mask_debug", Image, queue_size=10)
    
    def _handle_image_message(self, image_msg):
        """Handle incoming image messages"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            rospy.logdebug("Received image message")
        except Exception as e:
            rospy.logerr(f"Image conversion failed: {e}")
    
    def _process_dynamic_obstacles(self, obstacles_msg):
        """Process dynamic obstacles and extract features"""
        rospy.loginfo("Processing dynamic obstacles")
        
        if self.current_image is None:
            rospy.logwarn("No current image available")
            return
        
        if not obstacles_msg.box_ids:
            rospy.logwarn("No bounding boxes received")
            return
        
        try:
            # Extract features from all ROIs
            extracted_features, valid_rois, masks, processing_stats = self._extract_features_from_obstacles(
                obstacles_msg)
            
            # Create visualizations
            vis_image = self._create_visualization_image(
                obstacles_msg, valid_rois)
            
            # Create mask debug visualization
            mask_debug_vis = self._create_mask_debug_visualization(
                obstacles_msg, masks)
            
            # Publish all results
            self._publish_results(obstacles_msg, extracted_features, 
                                valid_rois, vis_image, mask_debug_vis, processing_stats)
            
        except Exception as e:
            rospy.logerr(f"Obstacle processing failed: {e}")
    
    def _extract_features_from_obstacles(self, obstacles_msg):
        """Extract features from all obstacle ROIs"""
        features = []
        valid_rois = []
        masks = []  # Collect masks for debug visualization
        processing_stats = {
            'clip_time': 0.0,
            'sam_time': 0.0,
            'processed_count': 0
        }
        
        rospy.loginfo(f"Processing {len(obstacles_msg.box_ids)} obstacles")
        
        for i, box_id in enumerate(obstacles_msg.box_ids):
            try:
                # Convert mask and extract ROI
                mask = self.bridge.imgmsg_to_cv2(obstacles_msg.mask[i], "passthrough")
                masks.append(mask)  # Collect mask for visualization
                
                roi = self.image_processor.extract_roi_from_mask(
                    self.current_image, mask, self.config.FOREGROUND_ONLY)
                
                # Validate ROI
                if not self.image_processor.is_valid_roi(roi):
                    rospy.logdebug(f"Invalid ROI for box_id {box_id}")
                    continue
                
                # Extract CLIP features
                clip_start_time = time.time()
                roi_features = self.model_manager.extract_clip_features(
                    roi, boost=self.config.BOOST_CLIP)
                clip_time = time.time() - clip_start_time
                
                if roi_features is not None:
                    features.append(roi_features)
                    valid_rois.append(roi)
                    processing_stats['clip_time'] += clip_time
                    processing_stats['processed_count'] += 1
                    
                    rospy.logdebug(f"Processed box_id {box_id} in {clip_time:.3f}s")
                
                # Optional SAM processing
                if self.config.RUN_SAM:
                    self._process_with_sam(roi, box_id, processing_stats)
                
            except Exception as e:
                rospy.logerr(f"Feature extraction failed for box_id {box_id}: {e}")
                # Add empty mask for failed processing to maintain alignment
                masks.append(np.zeros((1, 1), dtype=np.uint8))
                continue
        
        rospy.loginfo(f"Successfully processed {processing_stats['processed_count']} obstacles")
        return features, valid_rois, masks, processing_stats
    
    def _process_with_sam(self, roi, box_id, processing_stats):
        """Process ROI with FastSAM segmentation"""
        try:
            sam_start_time = time.time()
            results = self.model_manager.run_fastsam(roi)
            sam_time = time.time() - sam_start_time
            
            processing_stats['sam_time'] += sam_time
            rospy.logdebug(f"SAM processing for box_id {box_id}: {sam_time:.3f}s")
            
            # Process SAM results
            
                
        except Exception as e:
            rospy.logerr(f"SAM processing failed for box_id {box_id}: {e}")
    
    def _create_visualization_image(self, obstacles_msg, valid_rois):
        """Create visualization image with bounding boxes and labels"""
        vis_image = self.current_image.copy()
        
        try:
            for i, box_id in enumerate(obstacles_msg.box_ids[:len(valid_rois)]):
                # Get box coordinates
                box_x = obstacles_msg.box_x[i]
                box_y = obstacles_msg.box_y[i]
                box_width = obstacles_msg.box_width[i]
                box_height = obstacles_msg.box_height[i]
                
                # Draw bounding box
                color = self.color_generator.get_color(i)
                cv2.rectangle(vis_image, 
                             (box_x, box_y), 
                             (box_x + box_width, box_y + box_height), 
                             color, 2)
                
                # Add label
                cv2.putText(vis_image, f"ID:{box_id}", 
                           (box_x, box_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           self.config.FONT_SCALE, 
                           color, 
                           self.config.FONT_THICKNESS)
            
            return vis_image
            
        except Exception as e:
            rospy.logerr(f"Visualization creation failed: {e}")
            return self.current_image.copy()
    
    def _create_mask_debug_visualization(self, obstacles_msg, masks):
        """Create debug visualization for masks"""
        try:
            # Prepare box information for visualization
            boxes = []
            for i in range(len(obstacles_msg.box_ids)):
                box = (obstacles_msg.box_x[i], obstacles_msg.box_y[i], 
                      obstacles_msg.box_width[i], obstacles_msg.box_height[i])
                boxes.append(box)
            
            # Create mask visualization
            mask_debug_image = self.image_processor.create_mask_visualization(
                self.current_image, masks, obstacles_msg.box_ids, boxes)
            
            return mask_debug_image
            
        except Exception as e:
            rospy.logerr(f"Mask debug visualization creation failed: {e}")
            return None
    
    def _publish_results(self, obstacles_msg, features, valid_rois, 
                        vis_image, mask_debug_vis, processing_stats):
        """Publish all processing results"""
        try:
            # Publish visualization image
            vis_msg = self.bridge.cv2_to_imgmsg(vis_image, "bgr8")
            vis_msg.header = obstacles_msg.header
            self.vis_pub.publish(vis_msg)
            
            # Publish mask debug visualization
            if mask_debug_vis is not None and self.config.ENABLE_MASK_DEBUG:
                mask_debug_msg = self.bridge.cv2_to_imgmsg(mask_debug_vis, "bgr8")
                mask_debug_msg.header = obstacles_msg.header
                self.mask_vis_pub.publish(mask_debug_msg)
                rospy.logdebug("Published mask debug visualization")
            
            # Publish ROI grid visualization
            if valid_rois:
                roi_grid = self.image_processor.create_roi_grid(
                    valid_rois, obstacles_msg.box_ids[:len(valid_rois)])
                if roi_grid is not None:
                    roi_msg = self.bridge.cv2_to_imgmsg(roi_grid, "bgr8")
                    roi_msg.header = obstacles_msg.header
                    self.roi_vis_pub.publish(roi_msg)
            
            # Publish processed obstacles with features
            processed_obstacles = self._create_obstacles_message(
                obstacles_msg, features)
            self.obstacles_pub.publish(processed_obstacles)
            
            # Log processing statistics
            rospy.loginfo(
                f"Published results: {len(features)} features, "
                f"CLIP: {processing_stats['clip_time']:.3f}s, "
                f"SAM: {processing_stats['sam_time']:.3f}s"
            )
            
        except Exception as e:
            rospy.logerr(f"Result publishing failed: {e}")
    
    def _create_obstacles_message(self, original_msg, features):
        """Create obstacles message with extracted features"""
        processed_msg = Obstacles()
        processed_msg.header = original_msg.header
        
        # Copy obstacle data
        for attr in ['box_ids', 'centroid_x', 'centroid_y', 'centroid_z',
                    'thickness', 'height', 'width', 'box_x', 'box_y',
                    'box_width', 'box_height']:
            setattr(processed_msg, attr, getattr(original_msg, attr))
        
        # Add features
        processed_msg.features = []
        for feature in features:
            feature_msg = Float32MultiArray()
            feature_msg.data = feature.flatten().tolist()
            processed_msg.features.append(feature_msg)
        
        return processed_msg


def update_params(ExtractorConfig):
    """Update parameters from ROS parameter server"""
    ExtractorConfig.FOREGROUND_ONLY = rospy.get_param('~foreground_only', ExtractorConfig.FOREGROUND_ONLY)
    ExtractorConfig.BOOST_CLIP = rospy.get_param('~boost_clip', ExtractorConfig.BOOST_CLIP)
    ExtractorConfig.RUN_SAM = rospy.get_param('~run_sam', ExtractorConfig.RUN_SAM)
    ExtractorConfig.SAM_CONFIDENCE = rospy.get_param('~sam_confidence', ExtractorConfig.SAM_CONFIDENCE)
    ExtractorConfig.SAM_IOU = rospy.get_param('~sam_iou', ExtractorConfig.SAM_IOU)
    ExtractorConfig.SAM_IMAGE_SIZE = rospy.get_param('~sam_image_size', ExtractorConfig.SAM_IMAGE_SIZE)
    ExtractorConfig.MAX_ROI_SIZE = rospy.get_param('~max_roi_size', ExtractorConfig.MAX_ROI_SIZE)
    ExtractorConfig.MIN_ROI_WIDTH = rospy.get_param('~min_roi_width', ExtractorConfig.MIN_ROI_WIDTH)
    ExtractorConfig.MIN_ROI_HEIGHT = rospy.get_param('~min_roi_height', ExtractorConfig.MIN_ROI_HEIGHT)
    ExtractorConfig.ENABLE_MASK_DEBUG = rospy.get_param('~enable_mask_debug', ExtractorConfig.ENABLE_MASK_DEBUG)

def main():
    """Main function"""
    try:
        rospy.init_node("feature_extractor", anonymous=True)
        update_params(ExtractorConfig)
        extractor = FeatureExtractor()
        
        rospy.loginfo("FeatureExtractor node is running...")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("FeatureExtractor node shutdown")
    except Exception as e:
        rospy.logerr(f"FeatureExtractor node failed: {e}")


if __name__ == '__main__':
    main()