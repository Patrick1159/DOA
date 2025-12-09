import numpy as np
import cv2
import rospy

class ImageProcessor:
    """Handles image processing operations"""
    
    def __init__(self, config):
        self.config = config
    
    def extract_roi_from_mask(self, image, mask, remove_background=True):
        """Extract region of interest from image using mask"""
        try:
            non_zero_points = np.argwhere(mask > 0)
            
            if len(non_zero_points) == 0:
                rospy.logdebug("Mask contains no non-zero points")
                return np.zeros((0, 0, 3), dtype=image.dtype)
            
            # Calculate bounding rectangle
            min_coords = non_zero_points.min(axis=0)
            max_coords = non_zero_points.max(axis=0)
            min_y, min_x = min_coords
            max_y, max_x = max_coords
            
            # Extract ROI
            roi = image[min_y:max_y+1, min_x:max_x+1]
            mask_roi = mask[min_y:max_y+1, min_x:max_x+1]
            
            if remove_background:
                # **OPTIMIZATION**: Direct masked copy, avoid full zeros array
                result_roi = roi.copy()  # Copy first to avoid modifying original
                result_roi[mask_roi == 0] = 0  # Clear background
            else:
                result_roi = roi
            
            rospy.logdebug(f"ROI extracted successfully: size {result_roi.shape}")
            return result_roi
            
        except Exception as e:
            rospy.logerr(f"ROI extraction failed: {e}")
            return np.zeros((0, 0, 3), dtype=image.dtype)
    
    def is_valid_roi(self, roi):
        """Check if ROI meets minimum size requirements"""
        if roi.size == 0:
            return False
        
        height, width = roi.shape[:2]
        return (height >= self.config.MIN_ROI_HEIGHT and 
                width >= self.config.MIN_ROI_WIDTH)
    
    def create_roi_grid(self, rois, box_ids):
        """Create a grid visualization of multiple ROIs"""
        if not rois:
            return None
        
        try:
            # Calculate grid layout
            num_rois = len(rois)
            cols = int(np.ceil(np.sqrt(num_rois)))
            rows = int(np.ceil(num_rois / cols))
            
            # Calculate target size
            max_height = max(roi.shape[0] for roi in rois)
            max_width = max(roi.shape[1] for roi in rois)
            
            scale = min(self.config.MAX_ROI_SIZE / max_height, 
                       self.config.MAX_ROI_SIZE / max_width, 1.0)
            
            target_height = int(max_height * scale)
            target_width = int(max_width * scale)
            
            # Create grid
            grid_height = rows * target_height
            grid_width = cols * target_width
            grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            # Place ROIs in grid
            for i, (roi, box_id) in enumerate(zip(rois, box_ids)):
                row, col = divmod(i, cols)
                
                # Resize and place ROI
                resized_roi = cv2.resize(roi, (target_width, target_height))
                y_start = row * target_height
                x_start = col * target_width
                
                grid_image[y_start:y_start+target_height, 
                          x_start:x_start+target_width] = resized_roi
                
                # Add label
                cv2.putText(grid_image, f"ID:{box_id}", 
                           (x_start + 5, y_start + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           self.config.FONT_SCALE, 
                           (0, 255, 0), 
                           self.config.FONT_THICKNESS)
            
            return grid_image
            
        except Exception as e:
            rospy.logerr(f"ROI grid creation failed: {e}")
            return None
    
    def create_mask_visualization(self, original_image, masks, box_ids, boxes):
        """Create a debug visualization showing all masks (vectorized)"""
        try:
            height, width = original_image.shape[:2]
            
            # Pre-allocate output images
            mask_vis = original_image.copy()
            mask_only = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Color palette for different masks
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                     (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
                     (0, 128, 128), (128, 128, 0)]
            
            alpha = 0.3
            
            # Process all masks in single pass (vectorized)
            for i, (mask, box_id) in enumerate(zip(masks, box_ids)):
                if mask is None or mask.size == 0:
                    continue
                
                color = colors[i % len(colors)]
                mask_bool = mask > 0
                
                # Vectorized blending: only apply to masked regions
                mask_vis[mask_bool] = cv2.addWeighted(
                    mask_vis[mask_bool].astype(np.float32),
                    1.0 - alpha,
                    np.full_like(mask_vis[mask_bool], color, dtype=np.float32),
                    alpha, 0
                ).astype(np.uint8)
                
                mask_only[mask_bool] = color
                
                # Draw box and labels if available
                if i < len(boxes):
                    box_x, box_y, box_width, box_height = boxes[i]
                    cv2.rectangle(mask_vis, (box_x, box_y),
                                 (box_x + box_width, box_y + box_height), color, 2)
                    cv2.rectangle(mask_only, (box_x, box_y),
                                 (box_x + box_width, box_y + box_height), color, 2)
                    
                    cv2.putText(mask_vis, f"ID:{box_id}", (box_x, box_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE,
                               color, self.config.FONT_THICKNESS)
                    cv2.putText(mask_only, f"ID:{box_id}", (box_x, box_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE,
                               color, self.config.FONT_THICKNESS)
            
            # Combine visualizations (single hstack)
            combined_vis = np.hstack([mask_vis, mask_only])
            
            # Add title
            text_height = 30
            title_img = np.zeros((text_height, combined_vis.shape[1], 3), dtype=np.uint8)
            cv2.putText(title_img, "Mask Debug: Original+Overlay | Masks Only",
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            final_vis = np.vstack([title_img, combined_vis])
            rospy.logdebug(f"Created mask visualization with {len(masks)} masks")
            return final_vis
            
        except Exception as e:
            rospy.logerr(f"Mask visualization creation failed: {e}")
            return None


class ColorGenerator:
    """Generates unique colors for visualization"""
    
    def __init__(self, max_colors=100):
        self.rgb_map = [self._id_to_color(i) for i in range(max_colors)]
    
    def _id_to_color(self, index):
        """Generate unique color for visualization"""
        import hashlib
        hash_hex = hashlib.md5(str(index).encode()).hexdigest()[:6]
        return tuple(int(hash_hex[i:i+2], 16) for i in (0, 2, 4))
    
    def get_color(self, index):
        """Get color for given index"""
        return self.rgb_map[index % len(self.rgb_map)]