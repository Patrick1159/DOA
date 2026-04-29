#!/home/user/miniconda3/envs/oa/bin/python3
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
from umap.msg import BoundingBoxes2D, BoundingBoxes3D, SegmentedImage, MultiImageFeatures, Obstacles, RematchStatistics

from config_manager import ExtractorConfig
from model_manager import ModelManager
from utils import ImageProcessor, ColorGenerator
from node_helpers import (
    convert_image_message,
    build_obstacles_message,
)

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
        self.statistics_pub = rospy.Publisher(
            '/rematch_statistics_clip', RematchStatistics, queue_size=10)

        
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
            'processed_count': 0,
            'preprocess_time': 0.0,
            'postprocess_time': 0.0,
            'total_time': 0.0,
        }
        
        rospy.loginfo(f"Processing {len(obstacles_msg.box_ids)} obstacles")
        time_start = time.time()

        for i, box_id in enumerate(obstacles_msg.box_ids):
            try:
                # Convert mask and extract ROI  (preprocess)
                roi_start = time.time()
                mask = self.bridge.imgmsg_to_cv2(obstacles_msg.mask[i], "passthrough")
                masks.append(mask)  # Collect mask for visualization
                
                roi = self.image_processor.extract_roi_from_mask(
                    self.current_image, mask, self.config.FOREGROUND_ONLY)
                processing_stats['preprocess_time'] += time.time() - roi_start
                
                # Validate ROI
                if not self.image_processor.is_valid_roi(roi):
                    rospy.logdebug(f"Invalid ROI for box_id {box_id}")
                    continue
                
                # Extract CLIP features  (model inference)
                if self.config.RUN_CLIP:
                    clip_start_time = time.time()
                    roi_features = self.model_manager.extract_clip_features(
                        roi, boost=self.config.BOOST_CLIP)
                    clip_time = time.time() - clip_start_time
                else:
                    roi_features = np.array([1, 1 ,1])  # Dummy feature
                    clip_time = 0.0
                
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
        processing_stats['total_time'] = total_time
        processing_stats['postprocess_time'] = max(
            0.0, total_time - processing_stats['preprocess_time'] - processing_stats['clip_time'])
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
            
            # Publish rematch statistics
            stats_msg = RematchStatistics()
            stats_msg.header = obstacles_msg.header
            stats_msg.node_name = 'clip'
            n = processing_stats.get('processed_count', 0)
            stats_msg.obstacle_count = n
            stats_msg.total_time = processing_stats.get('total_time', 0.0)
            stats_msg.model_time = processing_stats.get('clip_time', 0.0)
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

            # Log processing statistics
            rospy.loginfo(
                f"CLIP: total={stats_msg.total_time:.3f}s  "
                f"model={stats_msg.model_time:.3f}s  "
                f"pre={stats_msg.preprocess_time:.3f}s  "
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
    ExtractorConfig.RUN_CLIP = rospy.get_param('~run_clip', ExtractorConfig.RUN_CLIP)
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