# -*- coding: utf-8 -*-
"""内置插件：数据集管理。"""
from __future__ import annotations

from plant_phenotype.app.plugin_api import PlatformApp, PluginManifest
from plant_phenotype.app.plugins.datasets.tab import build_datasets_tab
from plant_phenotype.app.plugins.datasets.tools import ALL_TOOLS
from plant_phenotype.app.tab_manager import TabContribution


def get_plugin():
    return BuiltinDatasetsPlugin()


class BuiltinDatasetsPlugin:
    manifest = PluginManifest(
        id="builtin.datasets",
        name="数据集",
        version="0.1.0",
        description="多数据集存储、配对、标签映射、校验与格式转换",
    )

    def register(self, app: PlatformApp) -> None:
        app.register_tab(
            TabContribution(
                id="builtin.datasets.workspace",
                title="数据集",
                factory=lambda app, parent=None: build_datasets_tab(parent, app=app),
                order=5,
            )
        )
        for tool_cls in ALL_TOOLS:
            app.register_tool(tool_cls())
