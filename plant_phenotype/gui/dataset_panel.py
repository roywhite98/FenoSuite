# -*- coding: utf-8 -*-
"""当前数据集样本面板：切换共享的当前数据集，按状态筛选样本，选中时发出信号。"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from plant_phenotype.datasets.schema import STATUS_LABELS, TASK_TYPE_LABELS, Sample, split_label
from plant_phenotype.datasets.store import Dataset

STATUS_COLORS = {
    "unlabeled": QColor(140, 140, 140),
    "annotated": QColor(30, 140, 60),
    "predicted": QColor(200, 120, 0),
}
_FILTERS = (("all", "全部"), ("annotated", "已标注"), ("unlabeled", "未标注"), ("predicted", "预测/候选"))


class DatasetSamplePanel(QWidget):
    sampleSelected = pyqtSignal(str, str)  # dataset_id, sample_id
    displayModeChanged = pyqtSignal(str)  # "image" 或 "mask"

    def __init__(self, service, parent=None, sample_filter: Optional[Callable[[Sample], bool]] = None,
                 empty_hint: str = "", default_display: str = "image") -> None:
        super().__init__(parent)
        self._service = service
        self._sample_filter = sample_filter
        self._empty_hint = empty_hint
        self._dataset: Dataset | None = None
        self._selected_id: str | None = None
        self._display_mode = "mask" if default_display == "mask" else "image"
        self._pending = False
        self._pending_full = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(QLabel("当前数据集:"))
        self._combo = QComboBox()
        self._combo.setToolTip("在此切换后，标注、训练、分割、根系表型及根系表型 beta 等页面同步使用该数据集")
        self._combo.currentIndexChanged.connect(self._on_combo_changed)
        layout.addWidget(self._combo)

        self._info = QLabel("")
        self._info.setWordWrap(True)
        self._info.setProperty("mdRole", "caption")
        layout.addWidget(self._info)

        row = QHBoxLayout()
        row.addWidget(QLabel("筛选:"))
        self._filter = QComboBox()
        for key, label in _FILTERS:
            self._filter.addItem(label, key)
        self._filter.currentIndexChanged.connect(self._fill_samples)
        row.addWidget(self._filter, 1)
        layout.addLayout(row)

        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("显示:"))
        self._view_group = QButtonGroup(self)
        self._view_group.setExclusive(True)
        self._image_btn = QPushButton("原图")
        self._mask_btn = QPushButton("掩码")
        for key, button in (("image", self._image_btn), ("mask", self._mask_btn)):
            button.setCheckable(True)
            button.setToolTip("切换当前样本在画布上显示原图或对应掩码")
            self._view_group.addButton(button)
            button.clicked.connect(lambda _checked=False, mode=key: self.set_display_mode(mode))
            view_row.addWidget(button)
        self._image_btn.setChecked(self._display_mode == "image")
        self._mask_btn.setChecked(self._display_mode == "mask")
        layout.addLayout(view_row)

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list, 1)

        service.on_current_changed(self._on_service_current)
        service.on_dataset_changed(self._on_service_changed)
        self.destroyed.connect(lambda *_: self._detach(service))
        self.refresh()

    def _detach(self, service) -> None:
        service.remove_listener(self._on_service_current)
        service.remove_listener(self._on_service_changed)

    # ---------- 外部接口 ----------
    def dataset(self) -> Dataset | None:
        return self._dataset

    def selected_sample_id(self) -> str | None:
        return self._selected_id

    def display_mode(self) -> str:
        """当前画布应显示 ``image``（原图）或 ``mask``（掩码）。"""
        return self._display_mode

    def set_display_mode(self, mode: str) -> None:
        mode = "mask" if mode == "mask" else "image"
        button = self._mask_btn if mode == "mask" else self._image_btn
        button.setChecked(True)
        if mode == self._display_mode:
            return
        self._display_mode = mode
        self.displayModeChanged.emit(mode)

    def refresh(self) -> None:
        current = self._service.current_id
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItem("（未选择）", None)
        for s in self._service.store.list_summaries():
            self._combo.addItem(s.name, s.id)
        idx = self._combo.findData(current) if current else 0
        self._combo.setCurrentIndex(max(0, idx))
        self._combo.blockSignals(False)
        try:
            self._dataset = self._service.current()
        except Exception as e:
            self._dataset = None
            self._info.setText(f"加载失败: {e}")
        self._update_info()
        self._fill_samples()

    def select_sample(self, sample_id: str, emit: bool = True) -> bool:
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == sample_id:
                self._list.blockSignals(not emit)
                self._list.setCurrentItem(item)
                self._list.blockSignals(False)
                self._selected_id = sample_id
                return True
        return False

    def select_relative(self, delta: int) -> bool:
        row = self._list.currentRow() + delta
        if self._list.count() == 0 or row < 0 or row >= self._list.count():
            return False
        self._list.setCurrentRow(row)
        return True

    def clear_selection(self) -> None:
        self._list.blockSignals(True)
        self._list.setCurrentItem(None)
        self._list.clearSelection()
        self._list.blockSignals(False)
        self._selected_id = None

    # ---------- 内部 ----------
    def _update_info(self) -> None:
        ds = self._dataset
        if ds is None:
            self._info.setText("请先在「数据集」页新建或导入数据集，然后在此选择。")
            return
        s = ds.summary()
        text = (f"{TASK_TYPE_LABELS.get(s.task_type, s.task_type)} · {s.n_samples} 张 · "
                f"已标注 {s.n_annotated} · 预测 {s.n_predicted}")
        if s.has_splits:
            text += "\n" + s.split_text()
        reason = ds.protection_reason()
        if reason:
            text += "\n⚠ 只读：" + reason
        self._info.setText(text)

    def _fill_samples(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        ds = self._dataset
        key = self._filter.currentData()
        if ds is not None:
            for s in ds.samples:
                if self._sample_filter and not self._sample_filter(s):
                    continue
                if key == "annotated" and not (s.has_annotation and s.status == "annotated"):
                    continue
                if key == "unlabeled" and s.has_annotation:
                    continue
                if key == "predicted" and s.status != "predicted":
                    continue
                item = QListWidgetItem(f"● {s.image_name}")
                item.setData(Qt.ItemDataRole.UserRole, s.id)
                item.setForeground(STATUS_COLORS.get(s.status, QColor(0, 0, 0)))
                item.setToolTip(f"{s.id} · {STATUS_LABELS.get(s.status, s.status)} · {split_label(s.split)}")
                self._list.addItem(item)
        if ds is not None and self._list.count() == 0 and self._empty_hint:
            hint = QListWidgetItem(self._empty_hint)
            hint.setFlags(Qt.ItemFlag.NoItemFlags)
            self._list.addItem(hint)
        self._list.blockSignals(False)
        if self._selected_id:
            self.select_sample(self._selected_id, emit=False)

    def _on_combo_changed(self, _index: int) -> None:
        did = self._combo.currentData()
        try:
            self._service.set_current(did)
        except ValueError as e:
            QMessageBox.warning(self, "数据集", str(e))
            self.refresh()

    def _on_item_changed(self, item: QListWidgetItem | None, _prev) -> None:
        if item is None or self._dataset is None:
            return
        sid = item.data(Qt.ItemDataRole.UserRole)
        if not sid:
            return
        self._selected_id = sid
        self.sampleSelected.emit(self._dataset.id, sid)

    def _schedule(self, full: bool) -> None:
        # 回调可能来自本面板的选中信号处理过程，延迟到下一轮事件循环再重建列表
        self._pending_full = self._pending_full or full
        if not self._pending:
            self._pending = True
            QTimer.singleShot(0, self._run_pending)

    def _run_pending(self) -> None:
        full, self._pending, self._pending_full = self._pending_full, False, False
        if full:
            self.refresh()
        else:
            self._refresh_combo_only()

    def _on_service_current(self, _did) -> None:
        self._selected_id = None
        self._schedule(True)

    def _on_service_changed(self, did) -> None:
        self._schedule(did is None or self._dataset is None or did == self._dataset.id)

    def _refresh_combo_only(self) -> None:
        current = self._service.current_id
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItem("（未选择）", None)
        for s in self._service.store.list_summaries():
            self._combo.addItem(s.name, s.id)
        self._combo.setCurrentIndex(max(0, self._combo.findData(current) if current else 0))
        self._combo.blockSignals(False)
