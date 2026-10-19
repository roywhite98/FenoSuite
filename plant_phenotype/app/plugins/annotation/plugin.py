# -*- coding: utf-8 -*-
"""内置插件：数据标注。"""
from __future__ import annotations

from plant_phenotype.app.plugin_api import PlatformApp, PluginManifest
from plant_phenotype.app.plugins.annotation.tab import build_annotation_tab
from plant_phenotype.app.plugins.annotation.tools import ReassembleTilesTool, TileDirectoryTool
from plant_phenotype.app.tab_manager import TabContribution


def get_plugin():
    return BuiltinAnnotationPlugin()


class BuiltinAnnotationPlugin:
    manifest = PluginManifest(
        id="builtin.annotation",
        name="数据标注",
        version="0.1.0",
        description="画布标注、切块与 SAM2 点提示",
    )

    def register(self, app: PlatformApp) -> None:
        app.register_tab(
            TabContribution(
                id="builtin.annotation.workspace",
                title="数据标注",
                factory=lambda app, parent=None: build_annotation_tab(parent, app=app),
                order=10,
            )
        )
        app.register_tool(TileDirectoryTool())
        app.register_tool(ReassembleTilesTool())
