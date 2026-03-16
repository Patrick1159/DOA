import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from clip_types import ClipBackendRequest, ClipBackendResult, ClipProcessingStats


class ClipBackend:
    """Pure CLIP processing backend without ROS dependencies."""

    def __init__(self, config, model_manager, image_processor, logger=None):
        self.config = config
        self.model_manager = model_manager
        self.image_processor = image_processor
        self.logger = logger or logging.getLogger(__name__)
        self.last_process_time = time.perf_counter()

    def process(self, request: ClipBackendRequest) -> ClipBackendResult:
        self._validate_request(request)
        return self.process_fields(
            image=request.image,
            box_ids=request.box_ids,
            boxes=request.boxes,
            masks=request.masks,
        )

    def process_fields(
        self,
        image: np.ndarray,
        box_ids: List[int],
        boxes: List[Tuple[int, int, int, int]],
        masks: List[Optional[np.ndarray]],
    ) -> ClipBackendResult:
        now = time.perf_counter()
        self.logger.info("Time from last processing: %.3fs", now - self.last_process_time)
        self.last_process_time = now

        stats = ClipProcessingStats(input_obstacle_count=len(box_ids))
        self.logger.info("Processing %d obstacles", len(box_ids))
        time_start = time.perf_counter()

        features_aligned: List[Optional[np.ndarray]] = [None] * len(box_ids)

        for i, box_id in enumerate(box_ids):
            try:
                roi_start = time.perf_counter()
                mask = masks[i] if i < len(masks) else None
                if mask is None:
                    continue

                roi = self.image_processor.extract_roi_from_mask(
                    image, mask, self.config.FOREGROUND_ONLY
                )
                stats.preprocess_time += time.perf_counter() - roi_start

                if not self.image_processor.is_valid_roi(roi):
                    self.logger.debug("Invalid ROI for box_id %s", str(box_id))
                    continue

                stats.valid_roi_count += 1

                if self.config.RUN_CLIP:
                    clip_start_time = time.perf_counter()
                    roi_features = self.model_manager.extract_clip_features(
                        roi, boost=self.config.BOOST_CLIP
                    )
                    clip_time = time.perf_counter() - clip_start_time
                else:
                    roi_features = np.array([1, 1, 1], dtype=np.float32)
                    clip_time = 0.0

                if roi_features is not None:
                    features_aligned[i] = roi_features
                    stats.clip_time += clip_time
                    stats.processed_count += 1
                    self.logger.debug(
                        "Processed box_id %s in %.3fs", str(box_id), clip_time
                    )
            except Exception as e:
                self.logger.exception(
                    "Feature extraction failed for box_id %s: %s", str(box_id), str(e)
                )

        binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        stats.total_time = time.perf_counter() - time_start
        stats.postprocess_time = max(
            0.0, stats.total_time - stats.preprocess_time - stats.clip_time
        )

        self.logger.info(
            "CLIP pipeline done: input=%d valid=%d processed=%d total=%.3fs",
            stats.input_obstacle_count,
            stats.valid_roi_count,
            stats.processed_count,
            stats.total_time,
        )

        return ClipBackendResult(
            features_aligned=features_aligned,
            binary_mask=binary_mask,
            stats=stats,
        )

    def _validate_request(self, request: ClipBackendRequest):
        if request.image is None or not isinstance(request.image, np.ndarray):
            raise ValueError("ClipBackendRequest.image must be a numpy array")

        n_ids = len(request.box_ids)
        if n_ids != len(request.boxes) or n_ids != len(request.masks):
            raise ValueError(
                "ClipBackendRequest length mismatch: box_ids=%d boxes=%d masks=%d"
                % (n_ids, len(request.boxes), len(request.masks))
            )
