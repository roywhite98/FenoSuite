# -*- coding: utf-8 -*-
"""独立 Tool 协议。"""
from __future__ import annotations

from typing import Protocol

from plant_phenotype.app.data_context import DataContext
from plant_phenotype.app.task_models import CancelToken, TaskReporter, ToolResult, ToolSpec


class Tool(Protocol):
    spec: ToolSpec

    def validate(self, inputs: dict, context: DataContext) -> None: ...

    def run(
        self,
        inputs: dict,
        context: DataContext,
        reporter: TaskReporter,
        cancel: CancelToken,
    ) -> ToolResult: ...
