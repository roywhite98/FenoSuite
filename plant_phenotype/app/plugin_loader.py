# -*- coding: utf-8 -*-
"""加载内置插件模块。"""
from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plant_phenotype.app.plugin_api import PlatformApp

_LOG = logging.getLogger(__name__)

_BUILTIN_PLUGIN_MODULES = (
    "plant_phenotype.app.plugins.datasets.plugin",
    "plant_phenotype.app.plugins.annotation.plugin",
    "plant_phenotype.app.plugins.training.plugin",
    "plant_phenotype.app.plugins.segmentation.plugin",
    "plant_phenotype.app.plugins.root_phenotype.plugin",
    "plant_phenotype.app.plugins.root_phenotype_beta.plugin",
)


def load_builtin_plugins(app: PlatformApp) -> None:
    """导入各内置插件并调用 register(app)。"""
    for mod_name in _BUILTIN_PLUGIN_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            get_plugin = getattr(mod, "get_plugin", None)
            if get_plugin is None:
                raise AttributeError(f"{mod_name} 缺少 get_plugin()")
            plugin = get_plugin()
            plugin.register(app)
        except Exception:
            _LOG.exception("加载插件失败: %s", mod_name)
            raise
