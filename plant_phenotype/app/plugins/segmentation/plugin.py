# -*- coding: utf-8 -*-
"""内置插件：图像分割。"""
from __future__ import annotations

from plant_phenotype.app.plugin_api import PlatformApp, PluginManifest
from plant_phenotype.app.plugins.segmentation.tab import build_segmentation_tab
from plant_phenotype.app.plugins.segmentation.tools import BatchPredictTool, PreviewTool
from plant_phenotype.app.tab_manager import TabContribution


def get_plugin():
    return BuiltinSegmentationPlugin()


class BuiltinSegmentationPlugin:
    manifest = PluginManifest(
        id="builtin.segmentation",
        name="图像分割",
        version="0.1.0",
        description="单张预览与批量分割",
    )

    def register(self, app: PlatformApp) -> None:
        app.register_tab(
            TabContribution(
                id="builtin.segmentation.workspace",
                title="图像分割",
                factory=lambda app, parent=None: build_segmentation_tab(parent, app=app),
                order=30,
            )
        )
        app.register_tool(BatchPredictTool())
        app.register_tool(PreviewTool())
