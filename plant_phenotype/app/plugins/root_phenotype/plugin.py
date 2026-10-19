# -*- coding: utf-8 -*-
"""内置插件：根系表型。"""
from __future__ import annotations

from plant_phenotype.app.plugin_api import PlatformApp, PluginManifest
from plant_phenotype.app.plugins.root_phenotype.tab import build_root_phenotype_tab
from plant_phenotype.app.plugins.root_phenotype.tools import (
    RootAnalyzePipelineTool,
    RootAnalyzeSelectedRootsTool,
)
from plant_phenotype.app.tab_manager import TabContribution


def get_plugin():
    return BuiltinRootPhenotypePlugin()


class BuiltinRootPhenotypePlugin:
    manifest = PluginManifest(
        id="builtin.root_phenotype",
        name="根系表型",
        version="0.1.0",
        description="骨架、拓扑与统计",
    )

    def register(self, app: PlatformApp) -> None:
        app.register_tab(
            TabContribution(
                id="builtin.root_phenotype.workspace",
                title="根系表型",
                factory=lambda app, parent=None: build_root_phenotype_tab(app, parent),
                order=40,
            )
        )
        app.register_tool(RootAnalyzeSelectedRootsTool())
        app.register_tool(RootAnalyzePipelineTool())
