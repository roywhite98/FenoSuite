# -*- coding: utf-8 -*-
"""应用宿主：注册 Tool、Tab、Viewer 与任务运行器。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from plant_phenotype.app.data_context import DataContext
from plant_phenotype.app.dataset_service import DatasetService
from plant_phenotype.app.registry import ToolRegistry
from plant_phenotype.app.result_viewer import ResultViewer, ResultViewerRegistry
from plant_phenotype.app.tab_manager import TabContribution, TabManager
from plant_phenotype.app.tool_api import Tool

if TYPE_CHECKING:
    from plant_phenotype.app.task_runner import QtTaskRunner


@dataclass(frozen=True)
class PluginManifest:
    id: str
    name: str
    version: str
    description: str
    dependencies: list[str] = field(default_factory=list)


class Plugin(Protocol):
    manifest: PluginManifest

    def register(self, app: PlatformApp) -> None: ...


class PlatformApp:
    """插件注册入口；GUI Shell 持有单例并传给各 Tab / Tool。"""

    def __init__(self, datasets_root: str | Path | None = None) -> None:
        self.context = DataContext()
        self.datasets = DatasetService(datasets_root)
        self.tools = ToolRegistry()
        self.tabs = TabManager()
        self.viewers = ResultViewerRegistry()
        self._tasks: QtTaskRunner | None = None

    def set_task_runner(self, runner: QtTaskRunner | None) -> None:
        self._tasks = runner

    @property
    def tasks(self) -> QtTaskRunner | None:
        return self._tasks

    def register_tool(self, tool: Tool) -> None:
        self.tools.register(tool)

    def register_tab(self, tab: TabContribution) -> None:
        self.tabs.register(tab)

    def register_viewer(self, viewer: ResultViewer) -> None:
        self.viewers.register(viewer)
