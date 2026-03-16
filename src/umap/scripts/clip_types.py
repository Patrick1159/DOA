from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ClipBackendRequest:
    image: np.ndarray
    box_ids: List[int]
    boxes: List[Tuple[int, int, int, int]]
    masks: List[Optional[np.ndarray]]
    timestamp: Optional[float] = None


@dataclass
class ClipProcessingStats:
    input_obstacle_count: int = 0
    valid_roi_count: int = 0
    processed_count: int = 0
    clip_time: float = 0.0
    preprocess_time: float = 0.0
    postprocess_time: float = 0.0
    total_time: float = 0.0


@dataclass
class ClipBackendResult:
    features_aligned: List[Optional[np.ndarray]]
    binary_mask: np.ndarray
    stats: ClipProcessingStats
