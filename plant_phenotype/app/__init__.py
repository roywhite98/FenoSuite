# -*- coding: utf-8 -*-
"""Application task execution and plugin support."""
from __future__ import annotations

from plant_phenotype.app.data_context import DataContext
from plant_phenotype.app.plugin_api import PlatformApp, Plugin, PluginManifest
from plant_phenotype.app.plugin_loader import load_builtin_plugins
from plant_phenotype.app.registry import ToolRegistry
from plant_phenotype.app.result_viewer import ResultViewer, ResultViewerRegistry
from plant_phenotype.app.tab_manager import TabContribution, TabManager
from plant_phenotype.app.task_models import (
    Artifact,
    CancelToken,
    TaskReporter,
    TaskState,
    ToolResult,
    ToolSpec,
)
from plant_phenotype.app.tool_api import Tool

__all__ = [
    "Artifact",
    "CancelToken",
    "DataContext",
    "PlatformApp",
    "Plugin",
    "PluginManifest",
    "QtTaskRunner",
    "ResultViewer",
    "ResultViewerRegistry",
    "TabContribution",
    "TabManager",
    "TaskReporter",
    "TaskState",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "load_builtin_plugins",
]


def __getattr__(name: str):
    if name == "QtTaskRunner":
        from plant_phenotype.app.task_runner import QtTaskRunner

        return QtTaskRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
