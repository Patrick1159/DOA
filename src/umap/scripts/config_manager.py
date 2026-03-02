import torch
import rospy

class ExtractorConfig:
    """Configuration constants for the feature extractor"""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Model paths
    FASTSAM_WEIGHTS_PATH = '/home/patrick/FastSAM/weights/FastSAM.pt'
    CLIP_MODEL_NAME = "ViT-B/32"
    
    # Processing parameters
    FOREGROUND_ONLY = False
    BOOST_CLIP = True
    RUN_SAM =True
    RUN_CLIP = True
    ENABLE_MASK_DEBUG = True  # Enable mask debug visualization
    PARALLEL = True
    
    # SAM parameters
    SAM_CONFIDENCE = 0.5
    SAM_IOU = 0.8
    SAM_IMAGE_SIZE = 64
    SAM_BATCH_SIZE = 12  # Batch inference - adjust based on GPU memory (2-8 recommended)
    SAM_ENABLE_BATCH = True  # Enable batch processing for FastSAM
    
    # ROI parameters
    MAX_ROI_SIZE = 200
    MIN_ROI_WIDTH = 10
    MIN_ROI_HEIGHT = 10
    
    # Visualization parameters
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    MAX_COLORS = 100