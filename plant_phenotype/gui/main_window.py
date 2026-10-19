# -*- coding: utf-8 -*-
"""统一主窗口：由内置插件注册工作区 Tab；任务状态在状态栏显示。"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from plant_phenotype.gui.theme import window_title
from plant_phenotype.app.plugin_api import PlatformApp
from plant_phenotype.app.plugin_loader import load_builtin_plugins
from plant_phenotype.app.task_models import TaskState
from plant_phenotype.app.task_runner import QtTaskRunner

ROOT = Path(__file__).resolve().parents[1]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(str(ROOT / "resources" / "icons" / "app.ico")))
        screen = QApplication.primaryScreen()
        if screen is not None:
            ag = screen.availableGeometry()
            w = max(960, min(int(ag.width() * 0.9), ag.width()))
            h = max(600, min(int(ag.height() * 0.9), ag.height()))
            self.resize(w, h)
        else:
            self.resize(1280, 750)
        self.setMinimumSize(800, 500)

        self._platform_app = PlatformApp()
        self._task_runner = QtTaskRunner(self._platform_app.tools, self._platform_app.context, self)
        self._platform_app.set_task_runner(self._task_runner)
        load_builtin_plugins(self._platform_app)
        self._platform_app.viewers.register_builtin_qt_viewers()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self.setWindowTitle(window_title())
        self._tab_by_id = self._platform_app.tabs.populate_workspace(self._tabs, self._platform_app, self)
        layout.addWidget(self._tabs)

        self._task_runner.task_state_changed.connect(self._on_task_state_changed)

    @property
    def platform_app(self) -> PlatformApp:
        return self._platform_app

    def _on_task_state_changed(self, state: TaskState) -> None:
        if state.status == "running":
            self.statusBar().showMessage(f"[{state.title}] {state.message}")
        elif state.status == "queued":
            self.statusBar().showMessage(f"[{state.title}] 排队中")
        elif state.status in ("success", "cancelled", "failed"):
            tail = state.error or state.message or ""
            self.statusBar().showMessage(f"[{state.title}] {state.status}: {tail[:300]}", 12000)

    def tab_annotation(self) -> QWidget | None:
        return self._tab_by_id.get("builtin.annotation.workspace")

    def tab_training(self) -> QWidget | None:
        return self._tab_by_id.get("builtin.training.workspace")

    def tab_segmentation(self) -> QWidget | None:
        return self._tab_by_id.get("builtin.segmentation.workspace")

    def tab_root_phenotype(self) -> QWidget | None:
        return self._tab_by_id.get("builtin.root_phenotype.workspace")

    def tab_root_phenotype_beta(self) -> QWidget | None:
        return self._tab_by_id.get("builtin.root_phenotype_beta.workspace")

    def closeEvent(self, event) -> None:
        if self._task_runner is not None:
            self._task_runner.wait_all()
        for i in range(self._tabs.count()):
            w = self._tabs.widget(i)
            if w is not None and hasattr(w, "wait_workers"):
                w.wait_workers()
        super().closeEvent(event)
