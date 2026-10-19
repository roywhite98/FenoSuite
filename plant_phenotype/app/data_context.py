# -*- coding: utf-8 -*-
"""会话级数据上下文（工作区路径、最近掩码与表型结果）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KEY_LAST_MASK_PATH = "root_phenotype.last_mask_path"
KEY_PIPELINE_RESULT = "root_phenotype.pipeline_result"
KEY_VECTOR_DOCUMENT = "root_phenotype.vector_document"
KEY_LAST_MASK_DIR = "segmentation.last_mask_dir"
KEY_LAST_CHECKPOINT_PATH = "training.last_checkpoint_path"


@dataclass
class DataContext:
    """当前工作区路径、最近文件等；可扩展为项目级 project.yaml 镜像。"""

    project_root: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.extra[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.extra.get(key, default)

    def set_last_mask_path(self, path: str | Path | None) -> None:
        if path is None:
            self.extra.pop(KEY_LAST_MASK_PATH, None)
        else:
            self.set(KEY_LAST_MASK_PATH, str(path))

    def get_last_mask_path(self) -> str | None:
        v = self.get(KEY_LAST_MASK_PATH)
        return str(v) if v else None

    def set_pipeline_result(self, result: Any) -> None:
        self.set(KEY_PIPELINE_RESULT, result)

    def get_pipeline_result(self) -> Any:
        return self.get(KEY_PIPELINE_RESULT)

    def set_vector_document(self, document: Any) -> None:
        self.set(KEY_VECTOR_DOCUMENT, document)

    def get_vector_document(self) -> Any:
        return self.get(KEY_VECTOR_DOCUMENT)

    def set_last_mask_dir(self, path: str | Path | None) -> None:
        if path is None:
            self.extra.pop(KEY_LAST_MASK_DIR, None)
        else:
            self.set(KEY_LAST_MASK_DIR, str(path))

    def get_last_mask_dir(self) -> str | None:
        path = self.get(KEY_LAST_MASK_DIR)
        return str(path) if path else None

    def set_last_checkpoint_path(self, path: str | Path | None) -> None:
        if path is None:
            self.extra.pop(KEY_LAST_CHECKPOINT_PATH, None)
        else:
            self.set(KEY_LAST_CHECKPOINT_PATH, str(path))

    def get_last_checkpoint_path(self) -> str | None:
        path = self.get(KEY_LAST_CHECKPOINT_PATH)
        return str(path) if path else None
