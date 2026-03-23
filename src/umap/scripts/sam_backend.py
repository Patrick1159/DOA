import logging
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from sam_types import BackendRequest, BackendResult, ProcessingStats, RoiItem


class SamBackend:
    """Pure algorithm backend. No ROS imports or ROS message dependencies."""

    def __init__(self, config, model_manager, image_processor, color_generator, logger=None):
        self.config = config
        self.model_manager = model_manager
        self.image_processor = image_processor
        self.color_generator = color_generator
        self.logger = logger or logging.getLogger(__name__)
        self.last_process_time = time.perf_counter()

    def process(self, request: BackendRequest) -> BackendResult:
        self._validate_request(request)
        return self.process_fields(
            image=request.image,
            box_ids=request.box_ids,
            boxes=request.boxes,
            masks=request.masks,
            generate_overlay=request.generate_overlay,
            generate_vis_image=request.generate_vis_image,
            generate_mask_debug=request.generate_mask_debug,
            generate_roi_grid=request.generate_roi_grid,
        )

    def process_fields(
        self,
        image: np.ndarray,
        box_ids: List[int],
        boxes: List[Tuple[int, int, int, int]],
        masks: List[Optional[np.ndarray]],
        generate_overlay: bool = True,
        generate_vis_image: bool = True,
        generate_mask_debug: bool = True,
        generate_roi_grid: bool = True,
    ) -> BackendResult:
        now = time.perf_counter()
        self.logger.info("Time from last processing: %.3fs", now - self.last_process_time)
        self.last_process_time = now

        stats = ProcessingStats(input_obstacle_count=len(box_ids))
        self.logger.info("Processing %d obstacles", len(box_ids))
        time_start = time.perf_counter()

        preprocess_start = time.perf_counter()
        roi_items, masks_debug = self._extract_valid_rois(image, box_ids, boxes, masks)
        stats.valid_roi_count = len(roi_items)
        stats.preprocess_time = time.perf_counter() - preprocess_start

        sam_masks_dict = self._run_sam_on_rois(roi_items, stats)

        postprocess_start = time.perf_counter()
        mask_result = self._transform_sam_masks_to_image(
            image,
            roi_items,
            sam_masks_dict,
            build_overlay=generate_overlay,
        )
        overlay_image = mask_result["overlay_image"]

        valid_rois = [item.roi for item in roi_items]
        valid_box_ids = [item.box_id for item in roi_items]
        vis_image = self._create_visualization_image(image, roi_items) if generate_vis_image else None
        mask_debug_vis = (
            self._create_mask_debug_visualization(image, box_ids, boxes, masks_debug)
            if generate_mask_debug
            else None
        )
        roi_grid = (
            self._create_roi_grid_visualization(valid_rois, valid_box_ids)
            if generate_roi_grid
            else None
        )

        stats.segmented_roi_count = mask_result["segmented_roi_count"]
        stats.total_mask_count = mask_result["total_mask_count"]
        stats.postprocess_time = time.perf_counter() - postprocess_start
        stats.total_time = time.perf_counter() - time_start
        self.logger.info(
            "SAM pipeline done: input=%d valid=%d segmented=%d masks=%d total=%.3fs",
            stats.input_obstacle_count,
            stats.valid_roi_count,
            stats.segmented_roi_count,
            stats.total_mask_count,
            stats.total_time,
        )

        return BackendResult(
            valid_rois=valid_rois,
            valid_box_ids=valid_box_ids,
            binary_mask=mask_result["binary_mask"],
            overlay_image=overlay_image,
            vis_image=vis_image,
            mask_debug_vis=mask_debug_vis,
            roi_grid=roi_grid,
            stats=stats,
        )

    def _validate_request(self, request: BackendRequest):
        if request.image is None or not isinstance(request.image, np.ndarray):
            raise ValueError("BackendRequest.image must be a numpy array")

        n_ids = len(request.box_ids)
        if n_ids != len(request.boxes) or n_ids != len(request.masks):
            raise ValueError(
                "BackendRequest length mismatch: box_ids=%d boxes=%d masks=%d"
                % (n_ids, len(request.boxes), len(request.masks))
            )

    def _extract_valid_rois(
        self,
        image: np.ndarray,
        box_ids: List[int],
        boxes: List[Tuple[int, int, int, int]],
        masks: List[Optional[np.ndarray]],
    ):
        roi_items: List[RoiItem] = []
        masks_debug: List[np.ndarray] = []
        for i, box_id in enumerate(box_ids):
            try:
                mask = masks[i] if i < len(masks) else None
                if mask is None:
                    masks_debug.append(np.zeros((1, 1), dtype=np.uint8))
                    continue

                masks_debug.append(mask)
                roi = self.image_processor.extract_roi_from_mask(
                    image, mask, self.config.FOREGROUND_ONLY
                )
                if not self.image_processor.is_valid_roi(roi):
                    self.logger.debug("Invalid ROI for box_id %s", str(box_id))
                    continue

                roi_items.append(
                    RoiItem(
                        index=i,
                        box_id=int(box_id),
                        box=boxes[i],
                        roi=roi,
                        mask=mask,
                    )
                )
            except Exception as e:
                self.logger.exception("ROI extraction failed for box_id %s: %s", str(box_id), str(e))

        return roi_items, masks_debug

    def _run_sam_on_rois(self, roi_items: List[RoiItem], stats: ProcessingStats):
        sam_masks_dict: Dict[int, torch.Tensor] = {}
        if not roi_items:
            return sam_masks_dict

        if self.config.RUN_SAM:
            batch_rois = [item.roi for item in roi_items]
            batch_box_ids = [item.box_id for item in roi_items]

            sam_infer_start = time.perf_counter()
            batch_results = self.model_manager.run_fastsam_batch(batch_rois, batch_box_ids)
            stats.sam_time += time.perf_counter() - sam_infer_start

            if batch_results is not None:
                for box_id, results in batch_results:
                    if results is None or not hasattr(results[0], "masks"):
                        continue
                    sam_masks = results[0].masks.data
                    sam_masks_dict[int(box_id)] = sam_masks
                    self.logger.debug(
                        "Batch SAM result for box_id %s: %d masks",
                        str(box_id),
                        sam_masks.shape[0],
                    )
            else:
                self.logger.info("Batch processing disabled or failed, using sequential processing")
                for item in roi_items:
                    sam_masks = self._process_with_sam(item.roi, item.box_id, stats)
                    if sam_masks.shape[0] > 0:
                        sam_masks_dict[item.box_id] = sam_masks
        else:
            for item in roi_items:
                sam_masks_dict[item.box_id] = torch.ones(
                    (1, item.roi.shape[0], item.roi.shape[1]), dtype=torch.uint8
                )

        return sam_masks_dict

    def _process_with_sam(self, roi: np.ndarray, box_id: int, stats: ProcessingStats) -> torch.Tensor:
        try:
            sam_start_time = time.perf_counter()
            results = self.model_manager.run_fastsam(roi)
            sam_time = time.perf_counter() - sam_start_time
            stats.sam_time += sam_time
            self.logger.info("SAM processing for box_id %s: %.3fs", str(box_id), sam_time)

            masks = results[0].masks.data
            self.logger.info("SAM found %d masks for box_id %s", masks.shape[0], str(box_id))
            return masks
        except Exception as e:
            self.logger.exception("SAM processing failed for box_id %s: %s", str(box_id), str(e))
            return torch.empty((0, 0, 0))

    def _transform_sam_masks_to_image(self, image, roi_items: List[RoiItem], sam_masks_dict, build_overlay=True):
        image_shape = image.shape
        binary_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        color_overlay = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8) if build_overlay else None
        segmented_roi_count = 0
        total_mask_count = 0

        img_h, img_w = image_shape[:2]
        for item in roi_items:
            if item.box_id not in sam_masks_dict:
                continue

            sam_masks = sam_masks_dict[item.box_id]
            if sam_masks is None or sam_masks.shape[0] == 0:
                continue

            segmented_roi_count += 1
            total_mask_count += int(sam_masks.shape[0])

            box_x, box_y, box_width, box_height = item.box
            if box_width <= 0 or box_height <= 0:
                continue

            x0 = max(0, box_x)
            y0 = max(0, box_y)
            x1 = min(img_w, box_x + box_width)
            y1 = min(img_h, box_y + box_height)
            if x0 >= x1 or y0 >= y1:
                continue

            dx0 = x0 - box_x
            dy0 = y0 - box_y
            dx1 = dx0 + (x1 - x0)
            dy1 = dy0 + (y1 - y0)

            for mask_idx in range(sam_masks.shape[0]):
                roi_mask = sam_masks[mask_idx].cpu().numpy().astype(np.uint8)
                resized_mask = cv2.resize(
                    roi_mask,
                    (box_width, box_height),
                    interpolation=cv2.INTER_NEAREST,
                )

                cropped_mask = resized_mask[dy0:dy1, dx0:dx1]
                active = cropped_mask > 0
                if not np.any(active):
                    continue

                region_binary = binary_mask[y0:y1, x0:x1]
                region_binary[active] = 1

                if build_overlay:
                    color = self.color_generator.get_color(mask_idx)
                    region_color = color_overlay[y0:y1, x0:x1]
                    region_color[active] = color

            self.logger.debug("Processed %d masks for box_id %s", sam_masks.shape[0], str(item.box_id))

        overlay_image = None
        if build_overlay:
            overlay_image = image.copy()
        if build_overlay and total_mask_count > 0:
            # Blend only once to avoid per-mask full-frame copies.
            overlay_image = cv2.addWeighted(
                image,
                0.7,
                color_overlay,
                0.3,
                0,
            )

        return {
            "binary_mask": binary_mask,
            "overlay_image": overlay_image,
            "segmented_roi_count": segmented_roi_count,
            "total_mask_count": total_mask_count,
        }

    def _blend_masks_batch(self, image: np.ndarray, mask_color_pairs):
        result = image.copy()
        alpha = 0.3
        colored_mask = np.zeros_like(image)

        for color, mask in mask_color_pairs:
            colored_mask.fill(0)
            colored_mask[mask > 0] = color
            result = cv2.addWeighted(result, 1.0 - alpha, colored_mask, alpha, 0)

        return result

    def _create_visualization_image(self, image: np.ndarray, roi_items: List[RoiItem]):
        vis_image = image.copy()
        try:
            for item in roi_items:
                box_x, box_y, box_width, box_height = item.box
                color = self.color_generator.get_color(item.index)
                cv2.rectangle(
                    vis_image,
                    (box_x, box_y),
                    (box_x + box_width, box_y + box_height),
                    color,
                    2,
                )
                cv2.putText(
                    vis_image,
                    "ID:%s" % str(item.box_id),
                    (box_x, box_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.FONT_SCALE,
                    color,
                    self.config.FONT_THICKNESS,
                )
            return vis_image
        except Exception as e:
            self.logger.exception("Visualization creation failed: %s", str(e))
            return image.copy()

    def _create_mask_debug_visualization(self, image, box_ids, boxes, masks):
        try:
            safe_masks = [
                m if m is not None else np.zeros((1, 1), dtype=np.uint8)
                for m in masks
            ]
            return self.image_processor.create_mask_visualization(
                image, safe_masks, box_ids, boxes
            )
        except Exception as e:
            self.logger.exception("Mask debug visualization creation failed: %s", str(e))
            return None

    def _create_roi_grid_visualization(self, valid_rois, valid_box_ids):
        if not valid_rois:
            return None
        return self.image_processor.create_roi_grid(valid_rois, valid_box_ids)
