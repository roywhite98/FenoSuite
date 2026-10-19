# -*- coding: utf-8 -*-
"""数据集管理页：数据集列表 → 样本列表 → 原图/掩码叠加预览，附 train/val/test 划分、标签管理、数据校验与格式转换。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QEvent, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from plant_phenotype.app.plugins.datasets.dialogs import (
    ImportDialog,
    MappingEditDialog,
    NewDatasetDialog,
    PairMasksDialog,
    SplitDialog,
    _progress_dialog,
)
from plant_phenotype.datasets import mask_io
from plant_phenotype.datasets.convert import CONVERSIONS, EXPORT_ENCODINGS, EXPORT_LAYOUTS
from plant_phenotype.datasets.schema import (
    BACKGROUND_ID,
    SPLITS,
    STATUS_LABELS,
    TASK_TYPE_LABELS,
    UNSPLIT_LABEL,
    split_label,
)
from plant_phenotype.datasets.store import Dataset, DatasetError
from plant_phenotype.datasets.validation import ISSUE_LABELS
from plant_phenotype.gui.dataset_panel import STATUS_COLORS
from plant_phenotype.gui.segmentation_preview import PreviewCanvasView, numpy_rgb_to_qimage
from plant_phenotype.image_io import DEFAULT_IMAGE_EXTENSIONS

_SAMPLE_FILTERS = (("all", "全部"), ("annotated", "已标注"), ("unlabeled", "未标注"),
                   ("predicted", "预测/候选"), ("issues", "有问题"))
_SPLIT_FILTERS = (("all", "全部划分"), *((s, split_label(s)) for s in SPLITS), ("", UNSPLIT_LABEL))
_SPLIT_COLORS = {"train": QColor(30, 110, 200), "val": QColor(190, 120, 0), "test": QColor(150, 60, 180)}


def _open_in_explorer(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def _ro_item(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


class DatasetsTab(QWidget):
    def __init__(self, parent=None, platform_app=None) -> None:
        super().__init__(parent)
        self._app = platform_app
        self._service = platform_app.datasets
        self._ds: Dataset | None = None
        self._issues: list[dict] = []
        self._usage: dict[int, dict[str, int]] | None = None
        self._task_id: str | None = None
        self._task_kind: str | None = None
        self._busy_widgets: list[QWidget] = []
        self._reload_pending = False

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_dataset_column())
        splitter.addWidget(self._build_sample_column())
        splitter.addWidget(self._build_right_tabs())
        splitter.setSizes([250, 470, 560])
        for i, factor in enumerate((0, 1, 2)):
            splitter.setStretchFactor(i, factor)
        root.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        self._status = QLabel("")
        self._status.setProperty("mdRole", "status")
        bottom.addWidget(self._status, 1)
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(260)
        self._progress.hide()
        bottom.addWidget(self._progress)
        self._cancel_btn = QPushButton("取消任务")
        self._cancel_btn.hide()
        self._cancel_btn.clicked.connect(self._cancel_task)
        bottom.addWidget(self._cancel_btn)
        root.addLayout(bottom)

        self._service.on_current_changed(self._on_service_current)
        self._service.on_dataset_changed(self._on_service_changed)
        if platform_app.tasks is not None:
            platform_app.tasks.task_state_changed.connect(self._on_task_state)
        self._reload_dataset_list()

    # ======================= 构建界面 =======================
    def _btn(self, text: str, slot, tip: str = "", busy: bool = True) -> QPushButton:
        b = QPushButton(text)
        b.clicked.connect(slot)
        if tip:
            b.setToolTip(tip)
        if busy:
            self._busy_widgets.append(b)
        return b

    def _build_dataset_column(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(0, 0, 0, 0)
        ly.addWidget(QLabel("数据集（选中即设为当前，各功能页共享）"))
        self._ds_list = QListWidget()
        self._ds_list.currentItemChanged.connect(self._on_ds_selected)
        ly.addWidget(self._ds_list, 1)
        grid = QGridLayout()
        grid.addWidget(self._btn("新建", self._new_dataset), 0, 0)
        grid.addWidget(self._btn("导入...", self._import_dataset, "从外部目录导入原图与掩码（复制到软件数据目录）"), 0, 1)
        grid.addWidget(self._btn("重命名", self._rename_dataset, "只改显示名称，数据集 ID 与文件关联不变"), 1, 0)
        grid.addWidget(self._btn("复制", self._copy_dataset), 1, 1)
        grid.addWidget(self._btn("删除", self._delete_dataset), 2, 0)
        grid.addWidget(self._btn("打开目录", self._open_dataset_dir, busy=False), 2, 1)
        ly.addLayout(grid)
        self._legacy_label = QLabel("")
        self._legacy_label.setWordWrap(True)
        self._legacy_label.setProperty("mdRole", "caption")
        ly.addWidget(self._legacy_label)
        self._legacy_btn = self._btn("导入旧数据目录...", self._import_legacy)
        ly.addWidget(self._legacy_btn)
        root_lbl = QLabel(f"数据目录: {self._service.root}")
        root_lbl.setWordWrap(True)
        root_lbl.setToolTip(str(self._service.root))
        # 长路径没有空格无法换行，忽略其宽度提示以免撑宽整列
        root_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        root_lbl.setProperty("mdRole", "caption")
        root_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        ly.addWidget(root_lbl)
        return w

    def _build_sample_column(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("筛选:"))
        self._sample_filter = QComboBox()
        for key, label in _SAMPLE_FILTERS:
            self._sample_filter.addItem(label, key)
        self._sample_filter.currentIndexChanged.connect(self._fill_samples)
        row.addWidget(self._sample_filter)
        self._split_filter = QComboBox()
        for key, label in _SPLIT_FILTERS:
            self._split_filter.addItem(label, key)
        self._split_filter.currentIndexChanged.connect(self._fill_samples)
        row.addWidget(self._split_filter)
        self._search = QLineEdit()
        self._search.setPlaceholderText("按文件名搜索")
        self._search.textChanged.connect(self._fill_samples)
        row.addWidget(self._search, 1)
        ly.addLayout(row)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(("原图", "掩码", "划分", "状态", "尺寸"))
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.sectionResized.connect(self._on_sample_section_resized)
        self._name_col_manual = False
        self._fitting_name_col = False
        self._fit_pending = False
        self._table.viewport().installEventFilter(self)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.itemSelectionChanged.connect(self._on_sample_selected)
        ly.addWidget(self._table, 1)
        grid = QGridLayout()
        grid.addWidget(self._btn("添加图像...", self._add_images), 0, 0)
        grid.addWidget(self._btn("指定掩码...", self._assign_mask, "为选中样本手动指定掩码文件（按数据集映射转换）"), 0, 1)
        grid.addWidget(self._btn("批量配对...", self._pair_masks, "按命名规则把外部掩码目录配对到样本"), 0, 2)
        grid.addWidget(self._btn("解除配对", self._unpair, "删除选中样本的掩码，原图保留"), 1, 0)
        grid.addWidget(self._btn("标为已标注", lambda: self._set_status("annotated"),
                                 "预测/候选结果复核后标为已标注，才会参与训练"), 1, 1)
        grid.addWidget(self._btn("删除样本", self._delete_samples), 1, 2)
        grid.addWidget(self._btn("设置划分...", self._assign_split,
                                 "把选中样本移到训练集 / 验证集 / 测试集，或移回未划分"), 2, 0)
        grid.addWidget(self._btn("随机划分...", self._random_split,
                                 "按比例随机划分 train / val / test（可复现）"), 2, 1)
        ly.addLayout(grid)
        return w

    def eventFilter(self, obj, event) -> bool:
        if obj is self._table.viewport() and event.type() == QEvent.Type.Resize:
            self._schedule_fit_name_columns()
        return super().eventFilter(obj, event)

    def _schedule_fit_name_columns(self) -> None:
        # 表头批量调整自适应列期间读到的列宽是中间值，推迟到本轮调整结束后再算
        if not self._name_col_manual and not self._fit_pending:
            self._fit_pending = True
            QTimer.singleShot(0, self._fit_name_columns)

    def _fit_name_columns(self) -> None:
        """未手动拖动过原图列时，原图与掩码两列平分其余列之外的宽度。"""
        self._fit_pending = False
        if self._name_col_manual:
            return
        header = self._table.horizontalHeader()
        fixed = sum(header.sectionSize(c) for c in range(2, self._table.columnCount()))
        free = self._table.viewport().width() - fixed
        self._fitting_name_col = True
        header.resizeSection(0, max(header.minimumSectionSize(), free // 2))
        self._fitting_name_col = False

    def _on_sample_section_resized(self, index: int, _old: int, _new: int) -> None:
        if index == 0 and not self._fitting_name_col:
            self._name_col_manual = True
        elif index >= 2:
            self._schedule_fit_name_columns()

    def _build_right_tabs(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_preview_page(), "预览")
        tabs.addTab(self._build_labels_page(), "标签")
        tabs.addTab(self._build_validation_page(), "校验")
        tabs.addTab(self._build_convert_page(), "转换/导出")
        self._right_tabs = tabs
        return tabs

    def _build_preview_page(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        row = QHBoxLayout()
        self._show_mask = QCheckBox("显示掩码")
        self._show_mask.setChecked(True)
        self._show_mask.toggled.connect(self._refresh_preview)
        row.addWidget(self._show_mask)
        row.addWidget(QLabel("透明度:"))
        self._alpha = QSlider(Qt.Orientation.Horizontal)
        self._alpha.setRange(0, 100)
        self._alpha.setValue(50)
        self._alpha.valueChanged.connect(self._refresh_preview)
        row.addWidget(self._alpha, 1)
        ly.addLayout(row)
        self._legend = QLabel("")
        self._legend.setTextFormat(Qt.TextFormat.RichText)
        self._legend.setWordWrap(True)
        ly.addWidget(self._legend)
        self._canvas = PreviewCanvasView()
        self._canvas.setMinimumSize(300, 240)
        ly.addWidget(self._canvas, 1)
        self._preview_info = QLabel("选择样本查看原图与掩码叠加；滚轮缩放，中键拖动")
        self._preview_info.setProperty("mdRole", "caption")
        ly.addWidget(self._preview_info)
        self._preview_cache: tuple[str, np.ndarray, np.ndarray | None] | None = None
        return w

    def _build_labels_page(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        tip = QLabel("类别编号即语义掩码中的像素值；名称和颜色只影响显示，修改后各功能页同步。")
        tip.setWordWrap(True)
        tip.setProperty("mdRole", "caption")
        ly.addWidget(tip)
        self._class_table = QTableWidget(0, 5)
        self._class_table.setHorizontalHeaderLabels(("编号", "名称", "颜色", "样本数", "像素数"))
        self._class_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._class_table.verticalHeader().setVisible(False)
        self._class_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._class_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        ly.addWidget(self._class_table, 1)
        grid = QGridLayout()
        grid.addWidget(self._btn("新增类别", self._add_class), 0, 0)
        grid.addWidget(self._btn("重命名", self._rename_class), 0, 1)
        grid.addWidget(self._btn("改颜色", self._recolor_class), 0, 2)
        grid.addWidget(self._btn("合并到...", self._merge_classes, "把选中类别合并到另一类别，先预览影响范围"), 1, 0)
        grid.addWidget(self._btn("删除类别", self._delete_class, "仅能删除未被使用的类别"), 1, 1)
        grid.addWidget(self._btn("统计用量", self._compute_usage), 1, 2)
        ly.addLayout(grid)
        self._mapping_label = QLabel("")
        self._mapping_label.setWordWrap(True)
        self._mapping_label.setProperty("mdRole", "caption")
        ly.addWidget(self._mapping_label)
        ly.addWidget(self._btn("编辑像素值 / 颜色映射...", self._edit_mapping,
                               "指定掩码、批量配对时，外部像素值/颜色按此映射到类别"))
        return w

    def _build_validation_page(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(self._btn("运行校验", self._run_validation))
        self._check_pixels = QCheckBox("检查掩码像素标签（较慢）")
        self._check_pixels.setChecked(True)
        row.addWidget(self._check_pixels)
        row.addStretch(1)
        ly.addLayout(row)
        self._issue_summary = QLabel("检查缺失文件、重复配对、尺寸不一致和未知标签；双击问题定位到样本。")
        self._issue_summary.setWordWrap(True)
        self._issue_summary.setProperty("mdRole", "caption")
        ly.addWidget(self._issue_summary)
        self._issue_table = QTableWidget(0, 4)
        self._issue_table.setHorizontalHeaderLabels(("级别", "类型", "样本", "说明"))
        self._issue_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._issue_table.verticalHeader().setVisible(False)
        self._issue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._issue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._issue_table.cellDoubleClicked.connect(self._locate_issue)
        ly.addWidget(self._issue_table, 1)
        return w

    def _build_convert_page(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        conv = QGroupBox("转换为新数据集（保留源数据与转换记录）")
        form = QFormLayout(conv)
        self._conv_kind = QComboBox()
        self._conv_kind.currentIndexChanged.connect(self._on_conv_kind_changed)
        form.addRow("转换类型:", self._conv_kind)
        self._conv_name = QLineEdit()
        form.addRow("新数据集名称:", self._conv_name)
        self._min_area = QSpinBox()
        self._min_area.setRange(1, 1_000_000)
        self._min_area.setValue(20)
        form.addRow("最小连通域面积:", self._min_area)
        self._connectivity = QComboBox()
        self._connectivity.addItem("8 邻域", 8)
        self._connectivity.addItem("4 邻域", 4)
        form.addRow("连通性:", self._connectivity)
        self._conv_note = QLabel("")
        self._conv_note.setWordWrap(True)
        self._conv_note.setProperty("mdRole", "caption")
        form.addRow(self._conv_note)
        form.addRow(self._btn("开始转换", self._run_convert))
        ly.addWidget(conv)

        exp = QGroupBox("导出到外部目录")
        form2 = QFormLayout(exp)
        self._exp_layout = QComboBox()
        for k, v in EXPORT_LAYOUTS.items():
            self._exp_layout.addItem(v, k)
        form2.addRow("目录布局:", self._exp_layout)
        self._exp_encoding = QComboBox()
        for k, v in EXPORT_ENCODINGS.items():
            self._exp_encoding.addItem(v, k)
        form2.addRow("掩码编码:", self._exp_encoding)
        self._exp_suffix = QLineEdit("_mask")
        form2.addRow("掩码后缀:", self._exp_suffix)
        self._exp_only = QCheckBox("仅导出有掩码的样本")
        form2.addRow(self._exp_only)
        self._exp_keep_splits = QCheckBox("保留划分（已划分样本写到 train/、val/、test/ 子目录）")
        self._exp_keep_splits.setChecked(True)
        form2.addRow(self._exp_keep_splits)
        form2.addRow(self._btn("导出...", self._run_export))
        ly.addWidget(exp)

        ly.addWidget(QLabel("最近任务报告:"))
        self._report = QPlainTextEdit()
        self._report.setReadOnly(True)
        ly.addWidget(self._report, 1)
        return w

    # ======================= 数据集列表 =======================
    def _reload_dataset_list(self) -> None:
        current = self._service.current_id
        self._ds_list.blockSignals(True)
        self._ds_list.clear()
        for s in self._service.store.list_summaries():
            text = f"{s.name}\n{TASK_TYPE_LABELS.get(s.task_type, s.task_type)} · {s.n_samples} 张 · 已标注 {s.n_annotated}"
            if s.has_splits:
                text += "\n" + s.split_text()
            item = QListWidgetItem(("★ " if s.id == current else "") + text)
            item.setData(Qt.ItemDataRole.UserRole, s.id)
            item.setToolTip(f"ID: {s.id}\n{s.path}")
            if s.id == current:
                f = QFont()
                f.setBold(True)
                item.setFont(f)
            self._ds_list.addItem(item)
            if s.id == current:
                self._ds_list.setCurrentItem(item)
        self._ds_list.blockSignals(False)
        legacy = self._service.store.find_legacy_dirs()
        self._legacy_label.setText(
            f"数据目录下有 {len(legacy)} 个未登记的旧数据目录（{', '.join(p.name for p in legacy[:3])}"
            f"{'…' if len(legacy) > 3 else ''}），可导入为数据集。" if legacy else ""
        )
        self._legacy_btn.setVisible(bool(legacy))
        self._load_current()

    def _load_current(self) -> None:
        try:
            self._ds = self._service.current()
        except Exception as e:
            self._ds = None
            self._status.setText(f"加载数据集失败: {e}")
        self._usage = None
        self._issues = []
        self._issue_table.setRowCount(0)
        self._preview_cache = None
        self._fill_samples()
        self._fill_classes()
        self._fill_conv_kinds()
        self._refresh_preview()

    def _on_ds_selected(self, item: QListWidgetItem | None, _prev) -> None:
        if item is None:
            return
        did = item.data(Qt.ItemDataRole.UserRole)
        if did != self._service.current_id:
            self._service.set_current(did)

    def _schedule_reload(self) -> None:
        # 回调可能来自列表自身的选中信号，延迟重建以免在信号处理中清空列表
        if not self._reload_pending:
            self._reload_pending = True
            QTimer.singleShot(0, self._run_reload)

    def _run_reload(self) -> None:
        self._reload_pending = False
        if self._task_id is None:
            self._reload_dataset_list()

    def _on_service_current(self, _did) -> None:
        self._schedule_reload()

    def _on_service_changed(self, _did) -> None:
        self._schedule_reload()

    def _notify(self, did: str | None = None) -> None:
        self._service.notify_changed(did if did is not None else (self._ds.id if self._ds else None))

    def _require_ds(self) -> Dataset | None:
        if self._ds is None:
            QMessageBox.information(self, "数据集", "请先选择或新建一个数据集。")
        return self._ds

    def _new_dataset(self) -> None:
        dlg = NewDatasetDialog(self)
        if dlg.exec() != NewDatasetDialog.DialogCode.Accepted:
            return
        name, task_type, classes = dlg.values()
        try:
            ds = self._service.store.create(name, task_type=task_type, classes=classes)
        except DatasetError as e:
            QMessageBox.warning(self, "新建数据集", str(e))
            return
        self._service.set_current(ds.id)
        self._notify(ds.id)

    def _import_dataset(self, source: str | None = None) -> None:
        dlg = ImportDialog(self, source=source if isinstance(source, str) else None)
        if dlg.exec() != ImportDialog.DialogCode.Accepted:
            return
        self._start_task("datasets.import", dlg.tool_inputs(), "import")

    def _import_legacy(self) -> None:
        legacy = self._service.store.find_legacy_dirs()
        if not legacy:
            return
        names = [p.name for p in legacy]
        name, ok = QInputDialog.getItem(self, "导入旧数据目录", "选择目录（可在导入对话框中改为其子目录）:", names, 0, False)
        if ok and name:
            self._import_dataset(str(legacy[names.index(name)]))

    def _rename_dataset(self) -> None:
        ds = self._require_ds()
        if ds is None:
            return
        name, ok = QInputDialog.getText(self, "重命名数据集", "新名称:", text=ds.name)
        if not ok or not name.strip() or name.strip() == ds.name:
            return
        try:
            self._service.store.rename(ds.id, name)
        except DatasetError as e:
            QMessageBox.warning(self, "重命名", str(e))
            return
        self._notify(ds.id)

    def _copy_dataset(self) -> None:
        ds = self._require_ds()
        if ds is None:
            return
        name, ok = QInputDialog.getText(self, "复制数据集", "副本名称:", text=f"{ds.name} 副本")
        if not ok or not name.strip():
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            cp = self._service.store.copy(ds.id, name)
        except Exception as e:
            QMessageBox.warning(self, "复制", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._service.set_current(cp.id)
        self._notify(cp.id)

    def _delete_dataset(self) -> None:
        ds = self._require_ds()
        if ds is None:
            return
        reply = QMessageBox.question(
            self, "删除数据集",
            f"确定删除数据集「{ds.name}」吗？\n将从磁盘删除 {len(ds.samples)} 个样本的原图、掩码与报告，无法恢复。\n{ds.path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            self._service.store.delete(ds.id)
        except Exception as e:
            QMessageBox.warning(self, "删除", str(e))
            return
        self._service.set_current(None)
        self._notify(None)

    def _open_dataset_dir(self) -> None:
        path = self._ds.path if self._ds else self._service.root
        path.mkdir(parents=True, exist_ok=True)
        _open_in_explorer(path)

    # ======================= 样本 =======================
    def _issue_samples(self) -> set[str]:
        return {i["sample_id"] for i in self._issues if i.get("sample_id")}

    def _fill_samples(self) -> None:
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        ds = self._ds
        if ds is not None:
            key = self._sample_filter.currentData()
            split_key = self._split_filter.currentData()
            text = self._search.text().strip().lower()
            issues = self._issue_samples()
            rows = []
            for s in ds.samples:
                if text and text not in s.image_name.lower():
                    continue
                if split_key != "all" and (s.split or "") != split_key:
                    continue
                if key == "annotated" and not (s.has_annotation and s.status == "annotated"):
                    continue
                if key == "unlabeled" and s.has_annotation:
                    continue
                if key == "predicted" and s.status != "predicted":
                    continue
                if key == "issues" and s.id not in issues:
                    continue
                rows.append(s)
            self._table.setRowCount(len(rows))
            for r, s in enumerate(rows):
                name = _ro_item(s.image_name)
                name.setData(Qt.ItemDataRole.UserRole, s.id)
                name.setToolTip(f"{s.id}\n来源: {s.source or ''}")
                self._table.setItem(r, 0, name)
                mask_text = s.mask.rsplit("/", 1)[-1] if s.mask else (f"{len(s.instances)} 个实例" if s.instances else "—")
                self._table.setItem(r, 1, _ro_item(mask_text))
                sp = _ro_item(s.split or "—")
                sp.setToolTip(split_label(s.split))
                if s.split:
                    sp.setForeground(_SPLIT_COLORS[s.split])
                self._table.setItem(r, 2, sp)
                st = _ro_item(STATUS_LABELS.get(s.status, s.status) + (" ⚠" if s.id in issues else ""))
                st.setForeground(STATUS_COLORS.get(s.status, QColor(0, 0, 0)))
                self._table.setItem(r, 3, st)
                self._table.setItem(r, 4, _ro_item(f"{s.width}×{s.height}" if s.width else ""))
        self._table.blockSignals(False)

    def _selected_samples(self):
        if self._ds is None:
            return []
        ids = []
        for r in sorted({i.row() for i in self._table.selectedIndexes()}):
            item = self._table.item(r, 0)
            if item is not None:
                ids.append(item.data(Qt.ItemDataRole.UserRole))
        return [s for s in (self._ds.sample(i) for i in ids) if s is not None]

    def _select_sample(self, sample_id: str) -> None:
        for r in range(self._table.rowCount()):
            if self._table.item(r, 0).data(Qt.ItemDataRole.UserRole) == sample_id:
                self._table.selectRow(r)
                self._table.scrollToItem(self._table.item(r, 0))
                return
        self._sample_filter.setCurrentIndex(0)
        self._split_filter.setCurrentIndex(0)
        self._search.clear()
        for r in range(self._table.rowCount()):
            if self._table.item(r, 0).data(Qt.ItemDataRole.UserRole) == sample_id:
                self._table.selectRow(r)
                self._table.scrollToItem(self._table.item(r, 0))
                return

    def _on_sample_selected(self) -> None:
        self._preview_cache = None
        self._refresh_preview()

    def _add_images(self) -> None:
        ds = self._require_ds()
        if ds is None:
            return
        exts = " ".join(f"*{e}" for e in DEFAULT_IMAGE_EXTENSIONS)
        files, _ = QFileDialog.getOpenFileNames(self, "选择要添加的图像", str(Path.home()), f"图像 ({exts})")
        if files:
            self._start_task("datasets.add_images", {"dataset_id": ds.id, "files": files}, "add_images")

    def _assign_mask(self) -> None:
        ds = self._require_ds()
        samples = self._selected_samples()
        if ds is None or len(samples) != 1:
            QMessageBox.information(self, "指定掩码", "请选中一个样本。")
            return
        if ds.meta.task_type != "semantic":
            QMessageBox.information(self, "指定掩码", "实例数据集不支持指定单张语义掩码。")
            return
        sample = samples[0]
        path, _ = QFileDialog.getOpenFileName(self, f"为 {sample.image_name} 指定掩码", str(Path.home()),
                                              "掩码 (*.png *.tif *.tiff *.bmp);;All (*)")
        if not path:
            return
        if sample.has_annotation and QMessageBox.question(
            self, "指定掩码", "该样本已有掩码，确定替换吗？"
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            unknown = ds.assign_mask_file(sample, path)
        except Exception as e:
            QMessageBox.warning(self, "指定掩码", f"读取掩码失败: {e}")
            return
        if unknown:
            QMessageBox.warning(
                self, "指定掩码",
                f"掩码含数据集映射未覆盖的值 {sorted(unknown)}，未写入。\n请在「标签」页编辑像素值/颜色映射后重试。",
            )
            return
        try:
            w, h = mask_io.read_image_size(path)
            if sample.width and (w, h) != (sample.width, sample.height):
                QMessageBox.warning(self, "指定掩码", f"注意：掩码尺寸 {w}×{h} 与原图 {sample.width}×{sample.height} 不一致。")
        except Exception:
            pass
        ds.add_history("assign_mask", f"{sample.id} <- {path}")
        ds.save()
        self._notify()

    def _pair_masks(self) -> None:
        ds = self._require_ds()
        if ds is None:
            return
        if ds.meta.task_type != "semantic":
            QMessageBox.information(self, "批量配对", "实例数据集不支持批量配对语义掩码。")
            return
        dlg = PairMasksDialog(ds, self)
        if dlg.exec() == PairMasksDialog.DialogCode.Accepted:
            self._start_task("datasets.pair_masks", {"dataset_id": ds.id, **dlg.tool_inputs()}, "pair_masks")

    def _unpair(self) -> None:
        ds = self._require_ds()
        samples = [s for s in self._selected_samples() if s.has_annotation]
        if ds is None or not samples:
            QMessageBox.information(self, "解除配对", "请选中有掩码的样本。")
            return
        if QMessageBox.question(self, "解除配对", f"删除 {len(samples)} 个样本的掩码文件？原图保留。") \
                != QMessageBox.StandardButton.Yes:
            return
        for s in samples:
            ds.clear_annotation(s)
        ds.add_history("unpair", f"{len(samples)} 个样本")
        ds.save()
        self._notify()

    def _set_status(self, status: str) -> None:
        ds = self._require_ds()
        samples = self._selected_samples()
        if ds is None or not samples:
            return
        skipped = 0
        for s in samples:
            try:
                ds.set_status(s, status)
            except DatasetError:
                skipped += 1
        ds.save()
        self._notify()
        if skipped:
            QMessageBox.information(self, "状态", f"{skipped} 个样本没有掩码，已跳过。")

    def _delete_samples(self) -> None:
        ds = self._require_ds()
        samples = self._selected_samples()
        if ds is None or not samples:
            return
        if QMessageBox.question(
            self, "删除样本", f"从数据集删除 {len(samples)} 个样本（含原图与掩码文件）？"
        ) != QMessageBox.StandardButton.Yes:
            return
        for s in samples:
            ds.remove_sample(s)
        ds.add_history("remove_samples", f"{len(samples)} 个")
        ds.save()
        self._notify()

    def _assign_split(self) -> None:
        ds = self._require_ds()
        samples = self._selected_samples()
        if ds is None or not samples:
            QMessageBox.information(self, "设置划分", "请先在样本列表中选中样本（可多选）。")
            return
        options = [(s, f"{split_label(s)} ({s})") for s in SPLITS] + [(None, UNSPLIT_LABEL)]
        labels = [label for _, label in options]
        current = {s.split for s in samples}
        idx = next((i for i, (k, _) in enumerate(options) if current == {k}), 0)
        choice, ok = QInputDialog.getItem(
            self, "设置划分", f"把选中的 {len(samples)} 个样本移到（原图与掩码会移动到对应子目录）:",
            labels, idx, False,
        )
        if not ok:
            return
        split = options[labels.index(choice)][0]
        self._start_task("datasets.split", {
            "dataset_id": ds.id, "mode": "assign", "split": split, "sample_ids": [s.id for s in samples],
        }, "split")

    def _random_split(self) -> None:
        ds = self._require_ds()
        if ds is None:
            return
        if not ds.samples:
            QMessageBox.information(self, "随机划分", "数据集中还没有样本。")
            return
        dlg = SplitDialog(ds, [s.id for s in self._selected_samples()], self)
        if dlg.exec() != SplitDialog.DialogCode.Accepted:
            return
        inputs = dlg.tool_inputs()
        if "sample_ids" not in inputs and ds.has_splits and QMessageBox.question(
            self, "随机划分", "将对全部样本重新划分，现有的 train / val / test 划分会被打乱。继续吗？"
        ) != QMessageBox.StandardButton.Yes:
            return
        self._start_task("datasets.split", {"dataset_id": ds.id, **inputs}, "split")

    # ======================= 预览 =======================
    def _refresh_preview(self) -> None:
        ds = self._ds
        samples = self._selected_samples()
        self._legend.setText(self._legend_html())
        if ds is None or not samples:
            self._canvas.clear()
            self._preview_info.setText("选择样本查看原图与掩码叠加；滚轮缩放，中键拖动")
            return
        s = samples[0]
        fit = False
        try:
            if self._preview_cache is None or self._preview_cache[0] != s.id:
                rgb = mask_io.load_rgb(ds.abs(s.image))
                index = ds.load_index_mask(s)
                self._preview_cache = (s.id, rgb, index)
                fit = True
            _, rgb, index = self._preview_cache
        except Exception as e:
            self._preview_info.setText(f"无法加载: {e}")
            return
        info = f"{s.image_name}　{STATUS_LABELS.get(s.status, s.status)}"
        display = rgb
        if index is not None:
            if index.shape[:2] != rgb.shape[:2]:
                info += f"　⚠ 尺寸不一致（原图 {rgb.shape[1]}×{rgb.shape[0]}，掩码 {index.shape[1]}×{index.shape[0]}）"
            elif self._show_mask.isChecked():
                display = mask_io.overlay(rgb, index, ds.meta.classes, self._alpha.value() / 100.0)
            vals = sorted(set(np.unique(index).tolist()))
            unknown = [v for v in vals if v not in ds.meta.class_ids]
            info += f"　标签值 {vals[:12]}" + (f"　⚠ 未定义 {unknown[:8]}（显示为品红）" if unknown else "")
        else:
            info += "　无掩码"
        self._canvas.set_pixmap(QPixmap.fromImage(numpy_rgb_to_qimage(display)))
        if fit:
            self._canvas.fitInView(self._canvas.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._preview_info.setText(info)

    def _legend_html(self) -> str:
        if self._ds is None:
            return ""
        parts = []
        for c in sorted(self._ds.meta.classes, key=lambda c: c.id):
            r, g, b = c.color
            parts.append(f"<span style='background:rgb({r},{g},{b});'>&nbsp;&nbsp;&nbsp;</span> {c.id} {c.name}")
        return "　".join(parts)

    # ======================= 标签 =======================
    def _fill_classes(self) -> None:
        self._class_table.setRowCount(0)
        ds = self._ds
        if ds is None:
            self._mapping_label.setText("")
            return
        rows = sorted(ds.meta.classes, key=lambda c: c.id)
        unknown = sorted(set(self._usage or {}) - ds.meta.class_ids)
        self._class_table.setRowCount(len(rows) + len(unknown))
        for r, c in enumerate(rows):
            self._class_table.setItem(r, 0, _ro_item(str(c.id)))
            self._class_table.setItem(r, 1, _ro_item(c.name))
            sw = _ro_item(mask_io.color_key(c.color))
            sw.setBackground(QColor(*c.color))
            sw.setForeground(QColor(255, 255, 255) if sum(c.color) < 380 else QColor(0, 0, 0))
            self._class_table.setItem(r, 2, sw)
            u = (self._usage or {}).get(c.id)
            self._class_table.setItem(r, 3, _ro_item(str(u["samples"]) if u else ("—" if self._usage is None else "0")))
            self._class_table.setItem(r, 4, _ro_item(str(u["pixels"]) if u else ("—" if self._usage is None else "0")))
        for k, v in enumerate(unknown):
            r = len(rows) + k
            self._class_table.setItem(r, 0, _ro_item(str(v)))
            warn = _ro_item("⚠ 未定义（掩码中存在）")
            warn.setForeground(QColor(200, 0, 120))
            self._class_table.setItem(r, 1, warn)
            self._class_table.setItem(r, 2, _ro_item(""))
            self._class_table.setItem(r, 3, _ro_item(str(self._usage[v]["samples"])))
            self._class_table.setItem(r, 4, _ro_item(str(self._usage[v]["pixels"])))
        mapping = ", ".join(f"{k}→{v}" for k, v in sorted(ds.meta.value_mapping.items()))
        self._mapping_label.setText(f"外部掩码映射: {mapping or '（空）'}")

    def _selected_class_ids(self) -> list[int]:
        ids = []
        for r in sorted({i.row() for i in self._class_table.selectedIndexes()}):
            item = self._class_table.item(r, 0)
            if item is not None and self._ds is not None and int(item.text()) in self._ds.meta.class_ids:
                ids.append(int(item.text()))
        return ids

    def _save_labels(self) -> None:
        self._ds.save()
        self._notify()

    def _add_class(self) -> None:
        ds = self._require_ds()
        if ds is None:
            return
        name, ok = QInputDialog.getText(self, "新增类别", f"类别名称（编号将为 {ds.meta.next_class_id()}）:")
        if not ok:
            return
        try:
            ds.add_class(name)
        except DatasetError as e:
            QMessageBox.warning(self, "新增类别", str(e))
            return
        self._save_labels()

    def _rename_class(self) -> None:
        ids = self._selected_class_ids()
        if self._ds is None or len(ids) != 1:
            QMessageBox.information(self, "重命名类别", "请选中一个类别。")
            return
        cls = self._ds.meta.class_by_id(ids[0])
        name, ok = QInputDialog.getText(self, "重命名类别", "新名称:", text=cls.name)
        if not ok:
            return
        try:
            self._ds.rename_class(cls.id, name)
        except DatasetError as e:
            QMessageBox.warning(self, "重命名类别", str(e))
            return
        self._save_labels()

    def _recolor_class(self) -> None:
        ids = self._selected_class_ids()
        if self._ds is None or len(ids) != 1:
            QMessageBox.information(self, "改颜色", "请选中一个类别。")
            return
        cls = self._ds.meta.class_by_id(ids[0])
        color = QColorDialog.getColor(QColor(*cls.color), self, f"类别 {cls.name} 的显示颜色")
        if not color.isValid():
            return
        self._ds.set_class_color(cls.id, (color.red(), color.green(), color.blue()))
        self._save_labels()

    def _compute_usage(self) -> bool:
        if self._require_ds() is None:
            return False
        dlg, progress = _progress_dialog(self, "统计类别用量…")
        try:
            self._usage = self._ds.class_usage(progress=progress, cancel=dlg.wasCanceled)
        except RuntimeError:
            return False
        except Exception as e:
            QMessageBox.warning(self, "统计用量", str(e))
            return False
        finally:
            dlg.close()
        self._fill_classes()
        return True

    def _merge_classes(self) -> None:
        ds = self._require_ds()
        src = [i for i in self._selected_class_ids() if i != BACKGROUND_ID]
        if ds is None or not src:
            QMessageBox.information(self, "合并类别", "请选中要合并的类别（背景不能作为源）。")
            return
        targets = [c for c in sorted(ds.meta.classes, key=lambda c: c.id) if c.id not in src]
        labels = [f"{c.id}  {c.name}" for c in targets]
        choice, ok = QInputDialog.getItem(self, "合并类别", "合并到:", labels, 0, False)
        if not ok:
            return
        target = targets[labels.index(choice)].id
        if self._usage is None and not self._compute_usage():
            return
        p = ds.merge_preview(src, target, self._usage)
        lines = [f"类别 {cid}（{ds.meta.class_by_id(cid).name}）：{v['samples']} 个样本，{v['pixels']} 像素"
                 + (f"，{v.get('instances', 0)} 个实例" if v.get("instances") else "")
                 for cid, v in p["per_class"].items()]
        msg = (f"将把以下类别合并到「{ds.meta.class_by_id(target).name}」并重写掩码：\n\n" + "\n".join(lines)
               + f"\n\n共影响约 {p['samples_upper_bound']} 个样本、{p['pixels']} 像素。源类别定义将被移除，"
                 "此操作直接修改当前数据集。\n建议先「复制」数据集作为备份。继续吗？")
        if QMessageBox.question(self, "合并类别 - 影响范围", msg) != QMessageBox.StandardButton.Yes:
            return
        self._start_task("datasets.merge_classes", {"dataset_id": ds.id, "sources": src, "target": target}, "merge")

    def _delete_class(self) -> None:
        ids = self._selected_class_ids()
        if self._ds is None or len(ids) != 1:
            QMessageBox.information(self, "删除类别", "请选中一个类别。")
            return
        if self._usage is None and not self._compute_usage():
            return
        try:
            self._ds.delete_class(ids[0], self._usage)
        except DatasetError as e:
            QMessageBox.warning(self, "删除类别", str(e))
            return
        self._save_labels()

    def _edit_mapping(self) -> None:
        ds = self._require_ds()
        if ds is None:
            return
        dlg = MappingEditDialog(ds, self)
        if dlg.exec() != MappingEditDialog.DialogCode.Accepted:
            return
        ds.meta.value_mapping = dlg.mapping()
        ds.add_history("edit_mapping", str(ds.meta.value_mapping))
        self._save_labels()

    # ======================= 校验 =======================
    def _run_validation(self) -> None:
        ds = self._require_ds()
        if ds is not None:
            self._start_task("datasets.validate",
                             {"dataset_id": ds.id, "check_pixels": self._check_pixels.isChecked()}, "validate")

    def _show_issues(self, issues: list[dict], report_path: str) -> None:
        self._issues = issues
        self._issue_table.setRowCount(len(issues))
        names = {s.id: s.image_name for s in self._ds.samples} if self._ds else {}
        for r, i in enumerate(issues):
            lvl = _ro_item("错误" if i["severity"] == "error" else "警告")
            lvl.setForeground(QColor(200, 30, 30) if i["severity"] == "error" else QColor(190, 120, 0))
            self._issue_table.setItem(r, 0, lvl)
            self._issue_table.setItem(r, 1, _ro_item(ISSUE_LABELS.get(i["kind"], i["kind"])))
            sid = i.get("sample_id") or ""
            it = _ro_item(names.get(sid, sid))
            it.setData(Qt.ItemDataRole.UserRole, sid)
            self._issue_table.setItem(r, 2, it)
            self._issue_table.setItem(r, 3, _ro_item(i["message"]))
        n_err = sum(1 for i in issues if i["severity"] == "error")
        self._issue_summary.setText(
            f"错误 {n_err} 项，警告 {len(issues) - n_err} 项。双击定位到样本；样本列表可筛选「有问题」。\n报告: {report_path}"
            if issues else f"未发现问题。报告: {report_path}"
        )
        self._fill_samples()

    def _locate_issue(self, row: int, _col: int) -> None:
        item = self._issue_table.item(row, 2)
        sid = item.data(Qt.ItemDataRole.UserRole) if item else None
        if sid:
            self._select_sample(sid)
            self._right_tabs.setCurrentIndex(0)

    # ======================= 转换 =======================
    def _fill_conv_kinds(self) -> None:
        self._conv_kind.blockSignals(True)
        self._conv_kind.clear()
        if self._ds is not None:
            allowed = "instance_to_semantic" if self._ds.meta.task_type == "instance" else "semantic_to_instance"
            self._conv_kind.addItem(CONVERSIONS[allowed], allowed)
            suffix = "语义" if allowed == "instance_to_semantic" else "实例候选"
            self._conv_name.setText(f"{self._ds.name}-{suffix}")
        self._conv_kind.blockSignals(False)
        self._on_conv_kind_changed()

    def _on_conv_kind_changed(self) -> None:
        kind = self._conv_kind.currentData()
        is_s2i = kind == "semantic_to_instance"
        self._min_area.setEnabled(is_s2i)
        self._connectivity.setEnabled(is_s2i)
        self._conv_note.setText({
            "semantic_to_instance": "语义转实例不能可靠还原对象：按每个类别的连通域生成候选实例，粘连的对象会被视为一个，"
                                    "结果标记为“预测/候选”，需人工复核。",
            "instance_to_semantic": "实例转语义会丢失个体区分：重叠像素取编号较大实例的类别。",
        }.get(kind, ""))

    def _run_convert(self) -> None:
        ds = self._require_ds()
        kind = self._conv_kind.currentData()
        if ds is None or not kind:
            return
        name = self._conv_name.text().strip()
        if not name:
            QMessageBox.warning(self, "转换", "请填写新数据集名称。")
            return
        self._start_task("datasets.convert", {
            "dataset_id": ds.id, "kind": kind, "name": name,
            "params": {"min_area": self._min_area.value(), "connectivity": self._connectivity.currentData()},
        }, "convert")

    def _run_export(self) -> None:
        ds = self._require_ds()
        if ds is None:
            return
        out = QFileDialog.getExistingDirectory(self, "选择导出目录（建议空目录）", str(Path.home()))
        if not out:
            return
        self._start_task("datasets.export", {
            "dataset_id": ds.id, "out_dir": out, "layout": self._exp_layout.currentData(),
            "encoding": self._exp_encoding.currentData(), "mask_suffix": self._exp_suffix.text().strip(),
            "only_annotated": self._exp_only.isChecked(), "keep_splits": self._exp_keep_splits.isChecked(),
        }, "export")

    # ======================= 任务 =======================
    def _start_task(self, tool_id: str, inputs: dict, kind: str) -> None:
        if self._task_id is not None or self._app.tasks is None:
            return
        self._task_kind = kind
        for w in self._busy_widgets:
            w.setEnabled(False)
        self._ds_list.setEnabled(False)
        self._progress.setRange(0, 0)
        self._progress.show()
        self._cancel_btn.show()
        self._status.setText("任务运行中…")
        self._task_id = self._app.tasks.run_tool(tool_id, {"datasets_root": str(self._service.root), **inputs})

    def _cancel_task(self) -> None:
        if self._task_id is not None:
            self._app.tasks.cancel(self._task_id)

    def _on_task_state(self, state) -> None:
        if state.id != self._task_id:
            return
        if state.status == "running":
            if state.progress is not None:
                self._progress.setRange(0, 1000)
                self._progress.setValue(int(state.progress * 1000))
            self._status.setText(state.message)
            return
        if state.status not in ("success", "failed", "cancelled"):
            return
        kind = self._task_kind
        self._task_id = None
        self._task_kind = None
        for w in self._busy_widgets:
            w.setEnabled(True)
        self._ds_list.setEnabled(True)
        self._progress.hide()
        self._cancel_btn.hide()
        if state.status != "success":
            self._status.setText(f"任务{'已取消' if state.status == 'cancelled' else '失败'}")
            if state.status == "failed":
                QMessageBox.critical(self, "任务失败", (state.error or state.message)[:2000])
            self._notify()
            return
        result = state.result
        self._status.setText(result.message)
        metrics = result.metrics or {}
        report = metrics.get("report")
        if report:
            self._report.setPlainText(self._format_report(report))
        new_id = metrics.get("dataset_id")
        if kind == "validate":
            self._show_issues(metrics.get("issues", []), metrics.get("report_path", ""))
            return
        if kind in ("import", "convert") and new_id:
            self._service.set_current(new_id)
        self._notify(new_id)
        if report and report.get("failed"):
            QMessageBox.warning(self, "完成（有失败项）", result.message + "\n\n" + "\n".join(
                f"{f.get('file') or f.get('sample')}: {f['error']}" for f in report["failed"][:15]
            ))
        elif kind in ("import", "convert", "export", "merge", "pair_masks", "split"):
            QMessageBox.information(self, "完成", result.message)

    @staticmethod
    def _format_report(report: dict) -> str:
        lines = []
        for k, v in report.items():
            if isinstance(v, list):
                lines.append(f"{k}: {len(v)} 项")
                lines += [f"  - {x}" for x in v[:20]]
                if len(v) > 20:
                    lines.append("  …")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)

    def wait_workers(self) -> None:
        if self._task_id is not None and self._app.tasks is not None:
            self._app.tasks.wait_task(self._task_id)


def build_datasets_tab(parent=None, app=None) -> QWidget:
    return DatasetsTab(parent, platform_app=app)
