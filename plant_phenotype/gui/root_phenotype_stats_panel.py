# -*- coding: utf-8 -*-
"""统计面板：按根1/根2/… 分块展示主干、二级根（参考 SegmentTask2/ui/stats_panel）。"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
    QScrollArea,
)


class RootPhenotypeStatsPanel(QWidget):
    """展示 PipelineResult 的 summary_df：层级、名称、数量、总长度、平均长度、平均直径。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._root_widget = QWidget()
        self._root_layout = QVBoxLayout(self._root_widget)
        self._root_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidget(self._root_widget)
        scroll.setWidgetResizable(True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self._placeholder = QLabel("未分析")
        self._placeholder.setWordWrap(True)
        self._root_layout.addWidget(self._placeholder)
        self._group_boxes: list = []
        self._scale_factor = None
        self._scale_unit = "mm"

    def set_result(self, result, scale_factor=None, scale_unit: str = "mm") -> None:
        """result: PipelineResult 或 None。scale_factor: 像素/实际长度，实际长度 = 像素/scale_factor。"""
        for g in self._group_boxes:
            g.setParent(None)
            g.deleteLater()
        self._group_boxes.clear()
        self._placeholder.setVisible(True)
        self._scale_factor = scale_factor
        self._scale_unit = scale_unit or "mm"

        if result is None:
            self._placeholder.setText("未分析")
            return
        if getattr(result, "error", None):
            self._placeholder.setText("错误: " + (result.error or "")[:200])
            return

        def _fmt(v):
            if self._scale_factor is not None and self._scale_factor > 0:
                return f"{(v / self._scale_factor):.4f}"
            return f"{v:.2f}"

        headers = ["层级", "名称", "数量", "总长度", "平均长度", "平均直径"]
        if self._scale_factor is not None and self._scale_factor > 0:
            u = self._scale_unit
            headers = ["层级", "名称", "数量", f"总长度({u})", f"平均长度({u})", f"平均直径({u})"]

        roots = getattr(result, "roots", None) or []
        if roots:
            self._placeholder.setVisible(False)
            for k, root in enumerate(roots):
                summary_df = root.get("summary_df")
                if summary_df is None or summary_df.empty:
                    continue
                g = QGroupBox(f"根{k + 1}")
                self._root_layout.addWidget(g)
                self._group_boxes.append(g)
                ly = QVBoxLayout(g)
                table = QTableWidget()
                table.setColumnCount(6)
                table.setHorizontalHeaderLabels(headers)
                table.setRowCount(len(summary_df))
                for i, row in summary_df.iterrows():
                    table.setItem(i, 0, QTableWidgetItem(str(int(row.get("level", 0)))))
                    table.setItem(i, 1, QTableWidgetItem(str(row.get("name", ""))))
                    table.setItem(i, 2, QTableWidgetItem(str(int(row.get("count", 0)))))
                    table.setItem(i, 3, QTableWidgetItem(_fmt(row.get("total_length", 0))))
                    table.setItem(i, 4, QTableWidgetItem(_fmt(row.get("avg_length", 0))))
                    table.setItem(i, 5, QTableWidgetItem(_fmt(row.get("avg_diameter", 0))))
                ly.addWidget(table)
            return

        df = getattr(result, "summary_df", None)
        if df is None or df.empty:
            self._placeholder.setText("无统计")
            return
        self._placeholder.setVisible(False)
        g = QGroupBox("统计")
        self._root_layout.addWidget(g)
        self._group_boxes.append(g)
        ly = QVBoxLayout(g)
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(df))
        for i, row in df.iterrows():
            table.setItem(i, 0, QTableWidgetItem(str(int(row.get("level", 0)))))
            table.setItem(i, 1, QTableWidgetItem(str(row.get("name", ""))))
            table.setItem(i, 2, QTableWidgetItem(str(int(row.get("count", 0)))))
            table.setItem(i, 3, QTableWidgetItem(_fmt(row.get("total_length", 0))))
            table.setItem(i, 4, QTableWidgetItem(_fmt(row.get("avg_length", 0))))
            table.setItem(i, 5, QTableWidgetItem(_fmt(row.get("avg_diameter", 0))))
        ly.addWidget(table)
