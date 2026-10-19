# -*- coding: utf-8 -*-
"""图像分割对外门面（预览 / 批量 / 叠加）。"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from plant_phenotype.segmentation.runner import (
    overlay_mask_on_image as _overlay_mask_on_image,
    run_batch_prediction as _run_batch_prediction,
    run_single_preview as _run_single_preview,
)


class SegmentationFacade:
    def preview(
        self,
        config_path: str,
        checkpoint_path: str,
        image_path: str,
        no_cuda: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return _run_single_preview(config_path, checkpoint_path, image_path, no_cuda)

    def batch_predict(
        self,
        config_path: str,
        checkpoint_path: str,
        image_dir: str,
        output_dir: str,
        no_cuda: bool = False,
        line_callback: Optional[Callable[[str], None]] = None,
        image_files: Optional[List[str]] = None,
    ) -> Tuple[int, List]:
        return _run_batch_prediction(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            image_dir=image_dir,
            output_dir=output_dir,
            no_cuda=no_cuda,
            line_callback=line_callback,
            image_files=image_files,
        )

    def overlay_mask(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.45,
    ) -> np.ndarray:
        return _overlay_mask_on_image(image_rgb, mask, color=color, alpha=alpha)

    def default_config_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "resources" / "configs" / "unet.yaml"


_default_facade = SegmentationFacade()


def get_facade() -> SegmentationFacade:
    return _default_facade
