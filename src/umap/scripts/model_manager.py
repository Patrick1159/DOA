import rospy
import torch
import clip
import cv2
import PIL
import sys
import time
from config_manager import ExtractorConfig
import numpy as np

# PyTorch 2.6+ defaults to weights_only=True, which rejects third-party
# checkpoint classes (e.g. ultralytics.nn.tasks.SegmentationModel).
# Revert to the pre-2.6 behaviour for trusted local model files.
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

sys.path.append(ExtractorConfig.FASTSAM_LIB_PATH)
from fastsam import FastSAM, FastSAMPrompt, FastSAMPredictor

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
            
            # Initialize FastSAM
            self.fastsam_model = FastSAM(self.config.FASTSAM_WEIGHTS_PATH)
            
            # Initialize CLIP
            self.clip_model, self.clip_preprocess = clip.load(
                self.config.CLIP_MODEL_NAME, device=self.device)
            
            rospy.loginfo("Models initialized successfully")
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize models: {e}")
            raise
    
    def extract_clip_features(self, image, boost=True):
        """Extract CLIP features from a single image."""
        try:
            if boost:
                height, width = image.shape[:2]
                resize_shape = (width // 2, height // 2)
                image = cv2.resize(image, resize_shape)

            pil_image = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            processed_image = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.clip_model.encode_image(processed_image)
                features /= features.norm(dim=-1, keepdim=True)

            return features.cpu().numpy()

        except Exception as e:
            rospy.logerr(f"CLIP feature extraction failed: {e}")
            return None

    def extract_clip_features_batch(self, images, boost=True):
        """Extract CLIP features from multiple images in one GPU forward pass.

        Args:
            images: List of BGR numpy arrays (may have different sizes).
            boost: If True, halve each image dimension before encoding.

        Returns:
            (N, D) numpy array of L2-normalised features, or None on failure.
        """
        if not images:
            return np.empty((0, 512), dtype=np.float32)

        try:
            processed = []
            for image in images:
                if boost:
                    h, w = image.shape[:2]
                    image = cv2.resize(image, (w // 2, h // 2))
                pil = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                processed.append(self.clip_preprocess(pil))

            batch = torch.stack(processed).to(self.device)

            with torch.no_grad():
                features = self.clip_model.encode_image(batch)
                features /= features.norm(dim=-1, keepdim=True)

            return features.cpu().numpy()

        except Exception as e:
            rospy.logerr(f"Batch CLIP feature extraction failed: {e}")
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
    
    def _get_sam_predictor(self, conf, iou, imgsz):
        """Return a cached FastSAMPredictor, avoiding the per-call overhead
        (model warmup, predictor re-creation) that FastSAM.predict() incurs."""
        key = (conf, iou, imgsz)
        if not hasattr(self, '_sam_predictor_cache'):
            self._sam_predictor_cache = {}
        if key not in self._sam_predictor_cache:
            overrides = self.fastsam_model.overrides.copy()
            overrides['conf'] = 0.25
            overrides.update(conf=conf, iou=iou, imgsz=imgsz,
                             mode='predict', save=False, verbose=False)
            pred = FastSAMPredictor(overrides=overrides)
            pred.setup_model(model=self.fastsam_model.model, verbose=False)
            self._sam_predictor_cache[key] = pred
        return self._sam_predictor_cache[key]

    def run_fastsam_batch(self, rois, box_ids):
        """Run FastSAM on *all* ROIs in one true GPU batch via a cached predictor."""
        if not self.config.SAM_ENABLE_BATCH or len(rois) <= 1:
            return None

        try:
            target_size = self.config.SAM_IMAGE_SIZE

            # Resize to uniform size required for single-tensor stacking
            resized_rois = [cv2.resize(r, (target_size, target_size)) for r in rois]

            rospy.loginfo(f"Batch FastSAM: {len(rois)} ROIs (imgsz={target_size})")
            t0 = time.time()

            predictor = self._get_sam_predictor(
                conf=self.config.SAM_CONFIDENCE,
                iou=self.config.SAM_IOU,
                imgsz=target_size)

            raw_results = list(predictor.stream_inference(
                source=resized_rois, model=self.fastsam_model.model))

            elapsed = time.time() - t0
            rospy.loginfo(f"Batch FastSAM completed: {len(rois)} ROIs in {elapsed:.3f}s "
                          f"({elapsed / len(rois):.3f}s per ROI)")

            batch_results = []
            for box_id, result in zip(box_ids, raw_results):
                batch_results.append((box_id, [result]))
            return batch_results

        except Exception as e:
            rospy.logerr(f"Batch SAM inference failed: {e}, falling back to sequential")
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