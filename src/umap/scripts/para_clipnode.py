#!/home/patrick/miniconda3/envs/ros-noetic/bin/python3
"""
CLIP process node, for parallel processing of images using FastSAM.
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

class ParaClipNode:
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
        rospy.loginfo("ParaClipNode initialized successfully")

    def _setup_ros_interface(self):
        """Setup ROS subscribers and publishers"""
        # Subscribers
        image_topic = rospy.get_param('~image_topic', '/camera/rgb/image_raw')
        self.image_sub = rospy.Subscriber(image_topic, Image, self._handle_image_message)
        self.obstacles_sub = rospy.Subscriber(
            'tracker/dynamic_obstacles', Obstacles, self._process_dynamic_obstacles)
        
        # Publishers
        self.obstacles_pub = rospy.Publisher(
            '/model_output/obstacles_clip', Obstacles, queue_size=10)

        
    def _handle_image_message(self, image_msg):
        """Handle incoming image messages"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            rospy.logdebug("Received image message")
        except Exception as e:
            rospy.logerr
    
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
            extracted_features, valid_rois, masks, processing_stats, binary_mask = self._extract_features_from_obstacles(
                obstacles_msg)

            # Publish all results
            self._publish_results(obstacles_msg=obstacles_msg, features=extracted_features, 
                                processing_stats=processing_stats, binary_mask=binary_mask)
            
        except Exception as e:
            rospy.logerr(f"Obstacle processing failed: {e}")

    def _extract_features_from_obstacles(self, obstacles_msg):
        """Extract features from all obstacle ROIs"""
        rospy.loginfo(f"Time from last processing: {time.time() - self.current_time:.3f}s")
        self.current_time = time.time()
        
        features = []
        valid_rois = []
        masks = []  # Collect masks for debug visualization
        processing_stats = {
            'clip_time': 0.0,
            'sam_time': 0.0,
            'processed_count': 0
        }
        
        rospy.loginfo(f"Processing {len(obstacles_msg.box_ids)} obstacles")
        time_start = time.time()

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
        
            except Exception as e:
                rospy.logerr(f"Feature extraction failed for box_id {box_id}: {e}")
                # Add empty mask for failed processing to maintain alignment
                masks.append(np.zeros((1, 1), dtype=np.uint8))
                continue

        binary_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
        time_end = time.time()
        total_time = time_end - time_start
        rospy.loginfo(f"Successfully processed {processing_stats['processed_count']} obstacles in {total_time:.3f}s")

        return features, valid_rois, masks, processing_stats, binary_mask
    
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
        
    def _publish_results(self, obstacles_msg, features, processing_stats, binary_mask):
        """Publish all processing results"""
        try:
            # Publish processed obstacles with features
            processed_obstacles = self._create_obstacles_message(
                obstacles_msg, features, binary_mask)
            # rospy.loginfo(f"raw binary mask non zero zones: {np.count_nonzero(binary_mask.data)}")
            # rospy.loginfo(f"binary mask in processed obstacles non zero zones: {np.count_nonzero(processed_obstacles.binary_mask.data)}")
            self.obstacles_pub.publish(processed_obstacles)
            
            # Log processing statistics
            rospy.loginfo(
                f"Published results: {len(features)} features, "
                f"CLIP: {processing_stats['clip_time']:.3f}s, "
                f"SAM: {processing_stats['sam_time']:.3f}s"
            )
            
        except Exception as e:
            rospy.logerr(f"Result publishing failed: {e}")

    def _create_obstacles_message(self, original_msg, features, binary_mask):
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
        
        # Add masks
        processed_msg.binary_mask.header = original_msg.header
        processed_msg.binary_mask.data = binary_mask.flatten().tolist()
        processed_msg.binary_mask.height = binary_mask.shape[0]
        processed_msg.binary_mask.width = binary_mask.shape[1]
        processed_msg.binary_mask.encoding = "mono8"
        processed_msg.binary_mask.is_bigendian = 0
        processed_msg.binary_mask.step = binary_mask.shape[1]
        
        return processed_msg
    



def update_params(ExtractorConfig):
    """Update parameters from ROS parameter server"""
    ExtractorConfig.PARALLEL = rospy.get_param('~parallel', ExtractorConfig.PARALLEL)
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
        extractor = ParaClipNode()

        rospy.loginfo("ParaClipNode node is running...")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ParaClipNode node shutdown")
    except Exception as e:
        rospy.logerr(f"ParaClipNode node failed: {e}")


if __name__ == '__main__':
    main()