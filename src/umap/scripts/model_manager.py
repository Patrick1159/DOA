#!/home/patrick/miniconda3/envs/fastsam/bin/python3

import rospy
import torch
import clip
import cv2
import PIL
import sys
import time
import os
from config_manager import ExtractorConfig
import numpy as np

sys.path.append("/home/patrick/FastSAM/")
from fastsam import FastSAM, FastSAMPrompt

class ModelManager:
    """Manages AI models for feature extraction"""
    
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize CLIP and FastSAM models"""
        try:
            rospy.loginfo(f"Initializing models on device: {self.device}")

            # PyTorch 2.6 changed torch.load default to weights_only=True, which
            # can break Ultralytics/FastSAM checkpoint loading.
            self._configure_torch_checkpoint_loading()
            
            # Initialize FastSAM
            self.fastsam_model = FastSAM(self.config.FASTSAM_WEIGHTS_PATH)
            
            # Initialize CLIP
            self.clip_model, self.clip_preprocess = clip.load(
                self.config.CLIP_MODEL_NAME, device=self.device)
            
            rospy.loginfo("Models initialized successfully")
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize models: {e}")
            raise

    def _configure_torch_checkpoint_loading(self):
        """Configure compatibility for trusted FastSAM checkpoints on PyTorch >=2.6."""
        try:
            # If FastSAM uses torch.load without explicit weights_only, force old behavior.
            # Use this only for trusted local checkpoints.
            os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

            # Allowlist common Ultralytics model classes for weights_only loading path.
            from ultralytics.nn.tasks import (
                BaseModel,
                ClassificationModel,
                DetectionModel,
                SegmentationModel,
                PoseModel,
                OBBModel,
            )
            torch.serialization.add_safe_globals([
                BaseModel,
                ClassificationModel,
                DetectionModel,
                SegmentationModel,
                PoseModel,
                OBBModel,
            ])
        except Exception as e:
            rospy.logwarn(f"Torch checkpoint compatibility setup skipped: {e}")
    
    def extract_clip_features(self, image, boost=True):
        """Extract CLIP features from image"""
        try:
            if boost:
                height, width = image.shape[:2]
                resize_shape = (width // 2, height // 2)
                image = cv2.resize(image, resize_shape)
            
            # Convert and preprocess
            pil_image = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            processed_image = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.clip_model.encode_image(processed_image)
                features /= features.norm(dim=-1, keepdim=True)
            
            return features.cpu().numpy()
            
        except Exception as e:
            rospy.logerr(f"CLIP feature extraction failed: {e}")
            return None
    
    def run_fastsam(self, image):
        """Run FastSAM segmentation on image"""
        try:
            results = self.fastsam_model(
                image, 
                device=self.device, 
                retina_masks=True,
                conf=self.config.SAM_CONFIDENCE,
                iou=self.config.SAM_IOU,
                imgsz=self.config.SAM_IMAGE_SIZE
            )
            rospy.loginfo("FastSAM processing completed")
            return results
            
        except Exception as e:
            rospy.logerr(f"FastSAM processing failed: {e}")
            return None
    
    def run_fastsam_batch(self, rois, box_ids):
        """Run FastSAM segmentation on multiple ROIs with batch-first strategy.
        
        Args:
            rois: List of ROI images (numpy arrays)
            box_ids: List of corresponding box IDs
            
        Returns:
            List of (box_id, results) tuples, or None if batch processing not enabled
        """
        if not self.config.SAM_ENABLE_BATCH or len(rois) <= 1:
            # Fall back to sequential processing
            return None
        
        try:
            batch_results = []
            batch_size = self.config.SAM_BATCH_SIZE
            
            rospy.loginfo(f"TRUE Batch FastSAM: Processing {len(rois)} ROIs with batch_size={batch_size}")
            batch_start_time = time.time()
            
            # Process ROIs in chunks. For each chunk, try one FastSAM batch call.
            for batch_idx in range(0, len(rois), batch_size):
                batch_end = min(batch_idx + batch_size, len(rois))
                batch_rois = rois[batch_idx:batch_end]
                batch_box_ids = box_ids[batch_idx:batch_end]
                
                rospy.loginfo(f"Processing batch {batch_idx // batch_size + 1}: ROIs {batch_idx}-{batch_end}")
                
                # Resize ROIs to same size for batched inference.
                target_size = self.config.SAM_IMAGE_SIZE
                resized_rois = []
                
                for roi in batch_rois:
                    resized_roi = cv2.resize(roi, (target_size, target_size))
                    resized_rois.append(resized_roi)
                
                batch_start = time.time()
                try:
                    # True batch attempt: single model invocation for the chunk.
                    batched_results = self.fastsam_model(
                        resized_rois,
                        device=self.device,
                        retina_masks=True,
                        conf=self.config.SAM_CONFIDENCE,
                        iou=self.config.SAM_IOU,
                        imgsz=target_size,
                    )

                    if not isinstance(batched_results, list):
                        batched_results = [batched_results]

                    if len(batched_results) != len(batch_box_ids):
                        raise RuntimeError(
                            f"Unexpected batch output size: got {len(batched_results)}, "
                            f"expected {len(batch_box_ids)}"
                        )

                    batch_sam_results = [
                        (box_id, [results])
                        for box_id, results in zip(batch_box_ids, batched_results)
                    ]
                    
                    batch_time = time.time() - batch_start
                    rospy.loginfo(
                        f"Batch {batch_idx // batch_size + 1} true-batched in {batch_time:.3f}s "
                        f"({batch_time / len(batch_rois):.3f}s per ROI)"
                    )
                    
                    batch_results.extend(batch_sam_results)
                    
                except Exception as e:
                    rospy.logwarn(f"True batch processing failed: {e}, falling back to sequential")
                    # Fallback: process individually
                    for roi, box_id in zip(batch_rois, batch_box_ids):
                        try:
                            results = self.fastsam_model(
                                roi,
                                device=self.device,
                                retina_masks=True,
                                conf=self.config.SAM_CONFIDENCE,
                                iou=self.config.SAM_IOU,
                                imgsz=self.config.SAM_IMAGE_SIZE
                            )
                            batch_results.append((box_id, results))
                        except Exception as e2:
                            rospy.logerr(f"Individual SAM failed for box_id {box_id}: {e2}")
                            batch_results.append((box_id, None))
                
                # Clear GPU cache between batches to avoid OOM
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    rospy.logdebug(f"Cleared GPU cache after batch")
            
            total_batch_time = time.time() - batch_start_time
            avg_time_per_roi = total_batch_time / len(rois) if rois else 0
            rospy.loginfo(f"Batch FastSAM completed: {len(rois)} ROIs in {total_batch_time:.3f}s "
                         f"({avg_time_per_roi:.3f}s per ROI)")
            return batch_results
            
        except Exception as e:
            rospy.logerr(f"Batch FastSAM processing failed: {e}")
            return None
    
    def run_fastsam_box_prompts(self, image_path, box_prompts, visualize_boxes=False):
        """Run FastSAM with box prompts
        
        Args:
            image_path: Path to the input image
            box_prompts: List of bounding boxes [[x, y, w, h], ...]
            visualize_boxes: If True, draw box prompts on the result image
        """
        try:
            # Run FastSAM on the image
            everything_results = self.fastsam_model(
            image_path,
            device=self.device,
            retina_masks=True,
            imgsz=self.config.SAM_IMAGE_SIZE,
            conf=self.config.SAM_CONFIDENCE,
            iou=self.config.SAM_IOU
            )
            
            # Initialize prompt processor
            prompt_process = FastSAMPrompt(image_path, everything_results, device=self.device)
            
            # Apply box prompts
            annotations = prompt_process.box_prompt(bboxes=box_prompts)
            
            # Visualize box prompts if requested
            if visualize_boxes and hasattr(prompt_process, 'img'):
                result_img = prompt_process.img.copy() if isinstance(prompt_process.img, np.ndarray) else cv2.imread(image_path)
                
                # Draw each box prompt
                for box in box_prompts:
                    x, y, w, h = box
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box
                
                prompt_process.img = result_img
            
            rospy.loginfo("FastSAM box prompt processing completed")
            return annotations, prompt_process
            
        except Exception as e:
            rospy.logerr(f"FastSAM box prompt processing failed: {e}")
            return None, None

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    
    models = ModelManager(ExtractorConfig)
    # Example usage (replace with actual image data)
    image_path = 'src/umap/scripts/test_img/dogs.jpg'
    dummy_image = cv2.imread(image_path)
    clip_features = models.extract_clip_features(dummy_image)
    
    # Run FastSAM with box prompts
    box_prompts = [[420, 250, 180, 200]]
    annotations, prompt_process = models.run_fastsam_box_prompts(image_path, box_prompts=box_prompts, visualize_boxes=True)
    
    # Visualize segmentation results
    if annotations is not None and prompt_process is not None:
        try:
            prompt_process.plot(
                annotations=annotations,
                output_path='src/umap/scripts/test_img/fastsam_result.jpg',
                withContours=True
            )
            rospy.loginfo("Segmentation visualization saved")
        except Exception as e:
            rospy.logwarn(f"Plot failed with Qt error, trying alternative method: {e}")
            # Alternative: just save the masks directly
            if hasattr(annotations, 'masks') and annotations.masks is not None:
                # Draw masks on original image
                result_img = dummy_image.copy()
                import numpy as np
                for mask in annotations.masks:
                    mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
                    colored_mask = np.zeros_like(result_img)
                    colored_mask[mask_np > 0] = [0, 255, 0]  # Green mask
                    result_img = cv2.addWeighted(result_img, 0.7, colored_mask, 0.3, 0)
                cv2.imwrite('src/umap/scripts/test_img/fastsam_result.jpg', result_img)
                rospy.loginfo("Segmentation saved using OpenCV fallback")
    
    print("CLIP Features shape:", clip_features.shape if clip_features is not None else None)
    print("FastSAM annotations:", type(annotations))