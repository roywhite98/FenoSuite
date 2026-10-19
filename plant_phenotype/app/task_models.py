# -*- coding: utf-8 -*-
"""平台任务与工具结果数据模型。"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

TaskStatus = Literal["queued", "running", "success", "failed", "cancelled"]
ToolRunStatus = Literal["success", "cancelled", "failed"]
ArtifactKind = Literal["image", "mask", "table", "model", "log", "directory", "json", "figure", "object"]


@dataclass
class Artifact:
    id: str
    kind: ArtifactKind
    path: Path | None = None
    data: Any = None
    mime: str | None = None


@dataclass
class ToolResult:
    status: ToolRunStatus
    message: str = ""
    artifacts: list[Artifact] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    id: str
    name: str
    category: str
    description: str
    input_schema: dict
    output_schema: dict
    ui: dict | None = None
    long_running: bool = False
    result_viewer: str | None = None


@dataclass
class TaskState:
    id: str
    tool_id: str
    title: str
    status: TaskStatus
    progress: float | None = None
    message: str = ""
    logs: list[str] = field(default_factory=list)
    result: ToolResult | None = None
    error: str | None = None


class CancelToken:
    """跨线程取消标记。"""

    def __init__(self) -> None:
        self._evt = threading.Event()

    def cancel(self) -> None:
        self._evt.set()

    def is_cancelled(self) -> bool:
        return self._evt.is_set()

    def raise_if_cancelled(self) -> None:
        if self._evt.is_set():
            raise RuntimeError("cancelled")


class TaskReporter(Protocol):
    def log(self, message: str) -> None: ...

    def progress(self, current: int, total: int | None = None, message: str = "") -> None: ...

    def artifact(self, artifact: Artifact) -> None: ...
