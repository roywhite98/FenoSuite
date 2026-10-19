# -*- coding: utf-8 -*-
"""数据标注对外门面（切块 / 还原 / SAM2 / 画布域类型再导出）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from plant_phenotype.annotation.inference.sam2_download import ensure_sam2_weights as _ensure_sam2_weights
from plant_phenotype.annotation.inference.sam2_onnx import (
    Sam2EmbeddingCache,
    SegmentAnything2ONNX,
    marks_need_positive as _marks_need_positive,
    predict_best_mask as _predict_best_mask,
)
from plant_phenotype.annotation.layer import AnnotationLayer, BrushShape, otsu_threshold_for_image
from plant_phenotype.annotation.processing import run_preprocessing_on_dir
from plant_phenotype.annotation.reassemble import reassemble_from_tiles
from plant_phenotype.annotation.undo import UndoStack
from plant_phenotype.annotation.config import (
    DEFAULT_MIN_FOREGROUND_RATIO,
    DEFAULT_TILE_SIZE,
    IMAGE_EXTENSIONS,
    MASK_SUFFIX,
)

# GUI 只应从本模块取这些类型 / 工厂，勿直连 annotation.layer / inference
__all__ = [
    "AnnotationFacade",
    "AnnotationLayer",
    "BrushShape",
    "Sam2EmbeddingCache",
    "SegmentAnything2ONNX",
    "UndoStack",
    "get_facade",
]


class AnnotationFacade:
    def tile_directory(
        self,
        annotated_dir: Path | str,
        tiles_base_dir: Path | str,
        tile_size: int = DEFAULT_TILE_SIZE,
        min_foreground_ratio: float = DEFAULT_MIN_FOREGROUND_RATIO,
    ) -> int:
        return run_preprocessing_on_dir(
            Path(annotated_dir),
            Path(tiles_base_dir),
            tile_size=tile_size,
            min_foreground_ratio=min_foreground_ratio,
        )

    def reassemble_tiles(
        self,
        tile_dir: Path | str,
        output_image_path: Optional[Path | str] = None,
        output_mask_path: Optional[Path | str] = None,
        use_predicted_mask: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return reassemble_from_tiles(
            Path(tile_dir),
            output_image_path=Path(output_image_path) if output_image_path else None,
            output_mask_path=Path(output_mask_path) if output_mask_path else None,
            use_predicted_mask=use_predicted_mask,
        )

    def compute_otsu_threshold(self, image: np.ndarray) -> int:
        return int(otsu_threshold_for_image(image))

    def ensure_sam2_weights(
        self,
        *,
        config_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        status: Optional[Callable[[str], None]] = None,
        pulse: Optional[Callable[[], None]] = None,
    ) -> Tuple[Path, Path]:
        return _ensure_sam2_weights(
            config_path=config_path,
            cache_dir=cache_dir,
            status=status,
            pulse=pulse,
        )

    def create_sam2_session(self, encoder_path: Path | str, decoder_path: Path | str) -> SegmentAnything2ONNX:
        return SegmentAnything2ONNX(str(encoder_path), str(decoder_path))

    def predict_sam2_mask(
        self,
        session: SegmentAnything2ONNX,
        image_rgb: np.ndarray,
        marks: List[dict],
        embedding: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return _predict_best_mask(session, image_rgb, marks, embedding=embedding)

    def marks_need_positive(self, marks: List[dict]) -> bool:
        return _marks_need_positive(marks)

    def new_annotation_layer(
        self,
        width: int,
        height: int,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> AnnotationLayer:
        return AnnotationLayer(width, height, color=color)

    def new_undo_stack(self, max_size: int = 50) -> UndoStack:
        return UndoStack(max_size=max_size)

    def new_sam2_embedding_cache(self, max_items: int = 2) -> Sam2EmbeddingCache:
        return Sam2EmbeddingCache(max_items=max_items)

    def default_tile_params(self) -> Dict[str, Any]:
        return {
            "tile_size": DEFAULT_TILE_SIZE,
            "min_foreground_ratio": DEFAULT_MIN_FOREGROUND_RATIO,
        }

    def image_extensions(self) -> Tuple[str, ...]:
        return IMAGE_EXTENSIONS

    def mask_suffix(self) -> str:
        return MASK_SUFFIX


_default_facade = AnnotationFacade()


def get_facade() -> AnnotationFacade:
    return _default_facade
