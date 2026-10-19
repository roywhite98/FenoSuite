# -*- coding: utf-8 -*-
"""内置插件：模型训练。"""
from __future__ import annotations

from plant_phenotype.app.plugin_api import PlatformApp, PluginManifest
from plant_phenotype.app.plugins.training.tab import build_training_tab
from plant_phenotype.app.plugins.training.tools import LocalTrainTool, RemoteTrainTool, SSHCheckTool, SSHTrainTool
from plant_phenotype.app.tab_manager import TabContribution


def get_plugin():
    return BuiltinTrainingPlugin()


class BuiltinTrainingPlugin:
    manifest = PluginManifest(
        id="builtin.training",
        name="模型训练",
        version="0.1.0",
        description="本地或远程模型训练",
    )

    def register(self, app: PlatformApp) -> None:
        app.register_tab(
            TabContribution(
                id="builtin.training.workspace",
                title="模型训练",
                factory=lambda app, parent=None: build_training_tab(parent, app=app),
                order=20,
            )
        )
        app.register_tool(LocalTrainTool())
        app.register_tool(RemoteTrainTool())
        app.register_tool(SSHTrainTool())
        app.register_tool(SSHCheckTool())
