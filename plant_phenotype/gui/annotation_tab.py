# -*- coding: utf-8 -*-
"""兼容导出：数据标注 Tab 已迁移至 `plant_phenotype.app.plugins.annotation.tab`。"""
from plant_phenotype.app.plugins.annotation.tab import (
    DEFAULT_MIN_FOREGROUND_RATIO,
    DEFAULT_TILE_SIZE,
    IMAGE_EXTENSIONS,
    MASK_SUFFIX,
    AnnotationTab,
    Sam2Worker,
    build_annotation_tab,
)

__all__ = [
    "AnnotationTab",
    "DEFAULT_MIN_FOREGROUND_RATIO",
    "DEFAULT_TILE_SIZE",
    "IMAGE_EXTENSIONS",
    "MASK_SUFFIX",
    "Sam2Worker",
    "build_annotation_tab",
]
