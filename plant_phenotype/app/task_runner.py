# -*- coding: utf-8 -*-
"""基于 QThread 的统一 Tool 任务执行器。"""
from __future__ import annotations

import traceback
import uuid
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from plant_phenotype.app.data_context import DataContext
from plant_phenotype.app.registry import ToolRegistry
from plant_phenotype.app.task_models import (
    Artifact,
    CancelToken,
    TaskReporter,
    TaskState,
    ToolResult,
)


class _SignalBridge(QObject):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, object, str)
    artifact = pyqtSignal(object)


class _BridgeReporter:
    def __init__(self, bridge: _SignalBridge) -> None:
        self._b = bridge

    def log(self, message: str) -> None:
        self._b.log.emit(message)

    def progress(self, current: int, total: int | None = None, message: str = "") -> None:
        self._b.progress.emit(current, total, message)

    def artifact(self, artifact: Artifact) -> None:
        self._b.artifact.emit(artifact)


class _ToolWorker(QThread):
    finished_result = pyqtSignal(str, object)  # task_id, ToolResult
    failed = pyqtSignal(str, str)  # task_id, error

    def __init__(
        self,
        task_id: str,
        tool_id: str,
        inputs: dict[str, Any],
        context: DataContext,
        tool_run: Any,
        bridge: _SignalBridge,
        cancel: CancelToken,
    ) -> None:
        super().__init__()
        self._task_id = task_id
        self._tool_id = tool_id
        self._inputs = inputs
        self._context = context
        self._tool = tool_run
        self._bridge = bridge
        self._cancel = cancel

    def run(self) -> None:
        reporter: TaskReporter = _BridgeReporter(self._bridge)
        try:
            self._tool.validate(self._inputs, self._context)
        except Exception as e:
            self.failed.emit(self._task_id, f"{type(e).__name__}: {e}")
            return
        try:
            self._cancel.raise_if_cancelled()
            result = self._tool.run(self._inputs, self._context, reporter, self._cancel)
            self.finished_result.emit(self._task_id, result)
        except RuntimeError as e:
            if "cancelled" in str(e).lower():
                self.finished_result.emit(
                    self._task_id,
                    ToolResult(status="cancelled", message="已取消"),
                )
            else:
                self.failed.emit(self._task_id, str(e))
        except Exception as e:
            self.failed.emit(self._task_id, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class QtTaskRunner(QObject):
    """在后台线程运行已注册 Tool；通过信号更新 UI。"""

    task_state_changed = pyqtSignal(object)  # TaskState

    def __init__(self, tools: ToolRegistry, context: DataContext, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._tools = tools
        self._context = context
        self._tasks: dict[str, TaskState] = {}
        self._workers: dict[str, _ToolWorker] = {}
        self._cancels: dict[str, CancelToken] = {}
        self._bridges: dict[str, _SignalBridge] = {}

    def cancel(self, task_id: str) -> None:
        c = self._cancels.get(task_id)
        if c is not None:
            c.cancel()
        st = self._tasks.get(task_id)
        if st is not None and st.status == "queued":
            st.status = "cancelled"
            st.message = "已取消（未开始）"
            self.task_state_changed.emit(st)

    def cancel_all(self) -> None:
        for tid in list(self._cancels.keys()):
            self.cancel(tid)

    def run_tool(
        self,
        tool_id: str,
        inputs: dict[str, Any],
        title: str | None = None,
    ) -> str:
        try:
            tool = self._tools.get(tool_id)
        except KeyError as e:
            raise ValueError(f"未注册的 Tool: {tool_id}") from e
        task_id = str(uuid.uuid4())
        display = title or tool.spec.name
        state = TaskState(
            id=task_id,
            tool_id=tool_id,
            title=display,
            status="queued",
            message="排队中",
        )
        self._tasks[task_id] = state
        self.task_state_changed.emit(state)

        cancel = CancelToken()
        self._cancels[task_id] = cancel
        bridge = _SignalBridge()
        self._bridges[task_id] = bridge

        def on_log(msg: str) -> None:
            st = self._tasks.get(task_id)
            if st is not None:
                st.logs.append(msg)
                st.message = msg[:200]
                self.task_state_changed.emit(st)

        def on_progress(cur: int, total: object, msg: str) -> None:
            st = self._tasks.get(task_id)
            if st is not None:
                if total is not None and int(total) > 0:
                    st.progress = min(1.0, max(0.0, cur / int(total)))
                st.message = msg or st.message
                self.task_state_changed.emit(st)

        bridge.log.connect(on_log)
        bridge.progress.connect(on_progress)

        worker = _ToolWorker(task_id, tool_id, inputs, self._context, tool, bridge, cancel)

        def on_done(tid: str, result: ToolResult) -> None:
            st = self._tasks.get(tid)
            if st is not None:
                st.status = "success" if result.status == "success" else ("cancelled" if result.status == "cancelled" else "failed")
                st.result = result
                st.error = result.error
                st.message = result.message or st.message
                self.task_state_changed.emit(st)
            self._cleanup(tid)

        def on_fail(tid: str, err: str) -> None:
            st = self._tasks.get(tid)
            if st is not None:
                st.status = "failed"
                st.error = err
                st.message = err[:200]
                self.task_state_changed.emit(st)
            self._cleanup(tid)

        worker.finished_result.connect(on_done)
        worker.failed.connect(on_fail)
        worker.finished.connect(worker.deleteLater)

        st = self._tasks[task_id]
        st.status = "running"
        st.message = "运行中…"
        self.task_state_changed.emit(st)

        self._workers[task_id] = worker
        worker.start()
        return task_id

    def _cleanup(self, task_id: str) -> None:
        self._workers.pop(task_id, None)
        self._cancels.pop(task_id, None)
        self._bridges.pop(task_id, None)

    def wait_task(self, task_id: str, timeout_ms: int = 600_000) -> None:
        w = self._workers.get(task_id)
        if w is not None and w.isRunning():
            w.wait(timeout_ms)

    def wait_all(self, timeout_ms: int = 600_000) -> None:
        for w in list(self._workers.values()):
            if w.isRunning():
                w.wait(timeout_ms)
