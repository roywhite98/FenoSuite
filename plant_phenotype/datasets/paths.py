# -*- coding: utf-8 -*-
"""软件目录与默认数据集根目录。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_WORKSPACE = "PLANT_PHENOTYPE_WORKSPACE"


def software_root() -> Path:
    """打包后为可执行文件所在目录，源码运行时为仓库根目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    override = os.environ.get(ENV_WORKSPACE, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return software_root() / "workspace"


def default_datasets_root() -> Path:
    return workspace_root() / "datasets"
