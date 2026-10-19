# -*- coding: utf-8 -*-
"""结果查看器注册表与占位实现（PyQt6 延迟导入，便于无 GUI 环境做单元测试）。"""
from __future__ import annotations

from typing import Protocol

from plant_phenotype.app.task_models import Artifact


class ResultViewer(Protocol):
    id: str
    kinds: tuple[str, ...]

    def can_view(self, artifact: Artifact) -> bool: ...

    def widget(self, parent: object | None) -> object: ...

    def show_artifact(self, artifact: Artifact) -> None: ...


class _LogArtifactViewer:
    id = "log/plain"
    kinds = ("log",)

    def __init__(self) -> None:
        from PyQt6.QtWidgets import QPlainTextEdit

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setPlaceholderText("日志类 Artifact 将显示在此")

    def can_view(self, artifact: Artifact) -> bool:
        return artifact.kind == "log"

    def widget(self, parent: object | None) -> object:
        return self._text

    def show_artifact(self, artifact: Artifact) -> None:
        if artifact.path and artifact.path.is_file():
            try:
                self._text.setPlainText(artifact.path.read_text(encoding="utf-8", errors="replace")[:50_000])
            except OSError:
                self._text.setPlainText(str(artifact.path))
        elif artifact.data is not None:
            self._text.setPlainText(str(artifact.data))
        else:
            self._text.clear()


class _TableArtifactViewer:
    id = "table/plain"
    kinds = ("table",)

    def __init__(self) -> None:
        from PyQt6.QtWidgets import QLabel

        self._label = QLabel("表格类结果请在各功能 Tab 中查看（如根系表型统计面板）。")
        self._label.setWordWrap(True)

    def can_view(self, artifact: Artifact) -> bool:
        return artifact.kind == "table"

    def widget(self, parent: object | None) -> object:
        return self._label

    def show_artifact(self, artifact: Artifact) -> None:
        hint = artifact.path.name if artifact.path else "内存表格"
        self._label.setText(f"表格 Artifact: {hint}（完整表格请在对应业务 Tab 查看）。")


class _ImageArtifactViewer:
    id = "image/preview"
    kinds = ("image", "mask")

    def __init__(self) -> None:
        from PyQt6.QtWidgets import QLabel

        self._label = QLabel("图像 / Mask 请在数据标注或分割预览 Tab 中打开对应文件。")
        self._label.setWordWrap(True)

    def can_view(self, artifact: Artifact) -> bool:
        return artifact.kind in ("image", "mask")

    def widget(self, parent: object | None) -> object:
        return self._label

    def show_artifact(self, artifact: Artifact) -> None:
        p = str(artifact.path) if artifact.path else "(无路径)"
        self._label.setText(f"路径: {p}")


class ResultViewerRegistry:
    def __init__(self) -> None:
        self._viewers: list[ResultViewer] = []

    def register_builtin_qt_viewers(self) -> None:
        """在 QApplication 已实例化后调用（内部使用 PyQt6 控件）。"""
        self.register(_LogArtifactViewer())
        self.register(_TableArtifactViewer())
        self.register(_ImageArtifactViewer())

    def register(self, viewer: ResultViewer) -> None:
        if any(v.id == viewer.id for v in self._viewers):
            raise ValueError(f"ResultViewer id 已注册: {viewer.id}")
        self._viewers.append(viewer)

    def pick(self, artifact: Artifact) -> ResultViewer | None:
        for v in self._viewers:
            if v.can_view(artifact):
                return v
        return None
