from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class BackendRequest:
    image: np.ndarray
    box_ids: List[int]
    boxes: List[Tuple[int, int, int, int]]
    masks: List[Optional[np.ndarray]]
    timestamp: Optional[float] = None
    generate_overlay: bool = True
    generate_vis_image: bool = True
    generate_mask_debug: bool = True
    generate_roi_grid: bool = True


@dataclass
class RoiItem:
    index: int
    box_id: int
    box: Tuple[int, int, int, int]
    roi: np.ndarray
    mask: np.ndarray


@dataclass
class ProcessingStats:
    input_obstacle_count: int = 0
    valid_roi_count: int = 0
    segmented_roi_count: int = 0
    total_mask_count: int = 0
    sam_time: float = 0.0
    preprocess_time: float = 0.0
    postprocess_time: float = 0.0
    total_time: float = 0.0


@dataclass
class BackendResult:
    valid_rois: List[np.ndarray]
    valid_box_ids: List[int]
    binary_mask: np.ndarray
    overlay_image: np.ndarray
    vis_image: np.ndarray
    mask_debug_vis: Optional[np.ndarray]
    roi_grid: Optional[np.ndarray]
    stats: ProcessingStats
