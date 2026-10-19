# -*- coding: utf-8 -*-
"""兼容导出：模型训练 Tab 已迁移至 `plant_phenotype.app.plugins.training.tab`。"""
from plant_phenotype.app.plugins.training.tab import (
    DEFAULT_CONFIG,
    DEFAULT_REMOTE_URL,
    TrainingTab,
    build_training_tab,
)
from plant_phenotype.training.remote_client import poll_remote_job as _poll_remote_job

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_REMOTE_URL",
    "TrainingTab",
    "_poll_remote_job",
    "build_training_tab",
]
