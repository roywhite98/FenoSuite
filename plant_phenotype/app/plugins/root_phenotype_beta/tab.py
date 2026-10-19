"""Independent vector phenotype UI. All domain mutations use the facade."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QKeySequence, QShortcut, QImageReader, QWheelEvent
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from plant_phenotype.datasets import mask_io
from plant_phenotype.gui.dataset_panel import DatasetSamplePanel
from plant_phenotype.gui.root_vector_canvas import RootVectorCanvas
from plant_phenotype.gui.utils import equalize_widths
from plant_phenotype.root_phenotype.facade import get_facade
from .controller import RootVectorController


_MODES = (("点选根系", "base"), ("选择根", "select"), ("编辑节点", "nodes"), ("校正交汇", "junction"))
_MODE_HINTS = {
    "base": "点选根系：点击根内任意位置即解析；完成后可继续点击下一株根。",
    "select": "选择根：点击画布上的根，在右侧查看并编辑。",
    "nodes": "编辑节点：先选中一条根，拖动节点修正位置。",
    "junction": "校正交汇：点击橙色交汇点，在右侧选择接续解释并确认。",
}


class _NoWheelDoubleSpinBox(QDoubleSpinBox):
    """滚轮不改变数值，避免滚动侧栏时误改参数。"""
    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


def _vline():
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setFixedWidth(1)
    line.setMinimumHeight(18)
    line.setMaximumHeight(48)
    return line


def _scrolled(widget):
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(widget)
    return scroll


_STATE = {"auto": "自动", "confirmed": "已确认", "pending": "待确认", "stale": "需复核",
          "accepted": "已接受", "complete": "完整", "junction_pending": "交汇待续",
          "pending_junction": "交汇待续", "pending": "待确认", "boundary_truncated": "边界截断",
          "boundary": "边界截断", "loop": "闭环待确认", "unknown": "未定向", "base_to_tip": "基部→根尖"}


def _display(value, digits=2):
    if value is None:
        return "—"
    try:
        number = float(value)
        if number != number:
            return "—"
        return f"{number:.{digits}f}"
    except (ValueError, TypeError):
        return str(value)


def build_root_phenotype_beta_tab(app, parent=None):
    return RootPhenotypeBetaTab(parent=parent, platform_app=app)


class RootPhenotypeBetaTab(QWidget):
    def __init__(self, parent=None, platform_app=None):
        super().__init__(parent)
        if platform_app is None:
            raise ValueError("根系表型beta需要平台任务运行器")
        self._platform_app = platform_app
        self.facade = get_facade()
        self.controller = RootVectorController(platform_app, self)
        self._mask_path = None
        self._original_path = None
        self._current_sample = None
        self._dataset_panel = None
        self._base_marks = []
        self._mask_cache = None
        self._selected_root_id = None
        self._tables = {}
        self._tree_items = {}
        self._updating = False
        self._source_valid = True
        self._build_ui()
        self.controller.documentChanged.connect(self._document_changed)
        self.controller.busyChanged.connect(self._busy_changed)
        self.controller.statusChanged.connect(self.status.setText)
        self.controller.errorOccurred.connect(self._show_error)
        self.controller.historyChanged.connect(self._history_changed)
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self.controller.undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, activated=self.controller.redo)
        self._busy_changed(False)
        self._document_changed(None)

    def _button(self, title, callback, layout, tooltip=None):
        button = QPushButton(title)
        button.clicked.connect(callback)
        if tooltip:
            button.setToolTip(tooltip)
        if layout is not None:
            layout.addWidget(button)
        return button

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        tools_scroll = self._build_sidebar()

        self.canvas = RootVectorCanvas()
        self.canvas.baseClicked.connect(self._add_base)
        self.canvas.rootSelected.connect(self._select_root)
        self.canvas.nodeSelected.connect(self._select_node)
        self.canvas.nodeMoved.connect(self._move_node)
        self.canvas.junctionSelected.connect(self._select_junction)
        self.status = QLabel("选择掩码，然后点击根内任意位置。")
        self.status.setWordWrap(True)
        self.status.setProperty("mdRole", "status")
        canvas_column = QWidget()
        canvas_layout = QVBoxLayout(canvas_column)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self.canvas, 1)
        canvas_layout.addWidget(self.status)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setChildrenCollapsible(False)
        self._class_combo = QComboBox()
        datasets = getattr(self._platform_app, "datasets", None)
        if datasets is not None:
            ds_wrap = QWidget()
            ds_layout = QVBoxLayout(ds_wrap)
            ds_layout.setContentsMargins(0, 0, 0, 0)
            ds_layout.setSpacing(0)
            self._dataset_panel = DatasetSamplePanel(
                datasets, self,
                sample_filter=lambda s: s.has_annotation,
                empty_hint="当前数据集还没有带掩码的样本",
                default_display="mask",
            )
            self._dataset_panel.sampleSelected.connect(self._on_dataset_sample_selected)
            self._dataset_panel.displayModeChanged.connect(self._apply_dataset_view)
            ds_layout.addWidget(self._dataset_panel, 1)
            class_row = QHBoxLayout()
            class_row.setContentsMargins(8, 0, 8, 8)
            class_row.addWidget(QLabel("分析类别:"))
            self._class_combo.setToolTip("多类别或实例数据集只取该类别的像素做根系分析")
            self._class_combo.currentIndexChanged.connect(self._on_class_combo_changed)
            class_row.addWidget(self._class_combo, 1)
            ds_layout.addLayout(class_row)
            ds_wrap.setMinimumHeight(160)
            right_splitter.addWidget(ds_wrap)
        right_splitter.addWidget(self._build_results())
        right_splitter.setSizes([240, 420] if datasets is not None else [660])

        inner_splitter = QSplitter(Qt.Orientation.Horizontal)
        inner_splitter.setChildrenCollapsible(False)
        inner_splitter.setHandleWidth(6)
        inner_splitter.addWidget(canvas_column)
        inner_splitter.addWidget(right_splitter)
        inner_splitter.setStretchFactor(0, 1)
        inner_splitter.setStretchFactor(1, 0)
        inner_splitter.setSizes([760, 320])
        right_splitter.setMinimumWidth(280)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self._build_top_bar())
        content_layout.addWidget(inner_splitter, 1)

        outer_splitter = QSplitter(Qt.Orientation.Horizontal)
        outer_splitter.setChildrenCollapsible(False)
        outer_splitter.setHandleWidth(6)
        outer_splitter.addWidget(tools_scroll)
        outer_splitter.addWidget(content)
        outer_splitter.setStretchFactor(0, 0)
        outer_splitter.setStretchFactor(1, 1)
        outer_splitter.setSizes([280, 1200])
        main_layout.addWidget(outer_splitter, 1)

    def _build_sidebar(self):
        params = QWidget()
        params.setObjectName("rootPhenotypeParams")
        side = QVBoxLayout(params)
        side.setContentsMargins(10, 10, 10, 10)
        side.setSpacing(10)

        inputs = QGroupBox("输入")
        inp = QVBoxLayout(inputs)
        inp.setContentsMargins(8, 10, 8, 8)
        inp.setSpacing(6)
        self.mask_name = QLineEdit()
        self.mask_name.setReadOnly(True)
        self.mask_name.setPlaceholderText("选择分割掩码，前景为非零像素")
        inp.addWidget(self.mask_name)
        file_grid = QGridLayout()
        file_grid.setSpacing(6)
        file_buttons = [
            self._button("选择掩码...", self._choose_mask, None),
            self._button("选择原图...", self._choose_original, None, "原图仅用于显示，尺寸须与掩码一致"),
            self._button("打开文档...", self._open_document, None),
            self._button("保存 JSON...", self._save_document, None),
        ]
        self.save_btn = file_buttons[3]
        for i, button in enumerate(file_buttons):
            file_grid.addWidget(button, i // 2, i % 2)
        inp.addLayout(file_grid)
        side.addWidget(inputs)

        usage = QGroupBox("操作说明")
        use = QVBoxLayout(usage)
        use.setContentsMargins(8, 10, 8, 8)
        hint = QLabel("「点选根系」模式下点击根内任意位置（基部、中段或末梢）即开始解析；"
                      "解析完成后可继续点击下一株根，结果自动合并。\n"
                      "红色：主根；蓝色：一级侧根；紫色：更高级侧根；灰色虚线：待确认。")
        hint.setWordWrap(True)
        use.addWidget(hint)
        side.addWidget(usage)

        repair = QGroupBox("解析参数")
        form = QFormLayout(repair)
        form.setContentsMargins(8, 10, 8, 8)
        form.setSpacing(6)
        self.gap_limit = _NoWheelDoubleSpinBox()
        self.gap_limit.setRange(0, 200)
        self.gap_limit.setValue(20)
        self.gap_limit.setSuffix(" px")
        self.gap_limit.setToolTip('仅连接端点相向的短断口；0 表示关闭。过大可能连接相邻独立根。')
        form.addRow("断根连接上限:", self.gap_limit)
        self.spur_limit = _NoWheelDoubleSpinBox()
        self.spur_limit.setRange(0, 100)
        self.spur_limit.setValue(5)
        self.spur_limit.setSuffix(" px")
        self.spur_limit.setToolTip('删除不超过此长度的末端短枝；0 表示保留全部细侧根。')
        form.addRow("去毛刺长度:", self.spur_limit)
        for control in (self.gap_limit, self.spur_limit):
            control.valueChanged.connect(self._scale_input_changed)
        side.addWidget(repair)

        exports = QGroupBox("标定与导出")
        exp = QVBoxLayout(exports)
        exp.setContentsMargins(8, 10, 8, 8)
        exp.setSpacing(6)
        self.scale = _NoWheelDoubleSpinBox()
        self.scale.setDecimals(8)
        self.scale.setRange(0, 10000)
        self.scale.setSpecialValueText("未标定（px）")
        self.scale.setSuffix(" mm/px")
        self.scale.valueChanged.connect(self._scale_input_changed)
        exp.addWidget(self.scale)
        self.scale_btn = self._button("应用标定", self._set_scale, exp)
        export_grid = QGridLayout()
        export_grid.setSpacing(6)
        self.export_buttons = []
        for i, ext in enumerate(("CSV", "XLSX", "SVG", "PNG")):
            button = self._button(f"导出 {ext}", lambda checked=False, e=ext.lower(): self._export(e), None)
            export_grid.addWidget(button, i // 2, i % 2)
            self.export_buttons.append(button)
        exp.addLayout(export_grid)
        side.addWidget(exports)
        side.addStretch()

        scroll = QScrollArea()
        scroll.setObjectName("rootPhenotypeSideBar")
        scroll.setProperty("mdDensity", "compact")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(params)
        scroll.setMinimumWidth(240)
        scroll.setMaximumWidth(340)
        return scroll

    def _build_top_bar(self):
        bar = QWidget()
        bar.setObjectName("rootPhenotypeTopBar")
        bar.setProperty("mdDensity", "compact")
        top = QHBoxLayout(bar)
        top.setContentsMargins(6, 2, 6, 2)
        top.setSpacing(4)
        top.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(QLabel("模式:"))
        self.mode = QComboBox()
        for text, key in _MODES:
            self.mode.addItem(text, key)
        self.mode.setToolTip("点选根系：点击即解析；选择根：查看与编辑；编辑节点：拖动节点；校正交汇：确认接续方式")
        self.mode.currentIndexChanged.connect(self._mode_changed)
        top.addWidget(self.mode)
        self.remove_base_btn = self._button("撤销上次点选", self._remove_base, top,
            "移除最后一个选点，并用剩余选点重新解析")
        self.analyze_btn = self._button("重新解析", self._analyze, top, "参数改变后用当前全部选点重新解析")
        self.analyze_btn.setProperty("mdRole", "filled")
        self.cancel_btn = self._button("取消", self.controller.cancel, top)
        equalize_widths([self.remove_base_btn, self.analyze_btn])
        top.addWidget(_vline())
        self.undo_btn = self._button("撤销", self.controller.undo, top, "撤销上一次编辑（Ctrl+Z）")
        self.redo_btn = self._button("重做", self.controller.redo, top, "重做（Ctrl+Y）")
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        fit_btn = self._button("适应画面", lambda: self.canvas.fit_image(), top)
        equalize_widths([self.cancel_btn, self.undo_btn, self.redo_btn])
        top.addWidget(_vline())
        self.width_check = QCheckBox("显示直径")
        self.width_check.toggled.connect(self._width_toggled)
        top.addWidget(self.width_check)
        top.addStretch(1)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setMaximumWidth(160)
        self._progress.setMaximumHeight(18)
        self._progress.setTextVisible(False)
        self._progress.hide()
        top.addWidget(self._progress)
        fit_btn.setToolTip("缩放到整张图像")
        bar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        return bar

    def _build_results(self):
        wrap = QWidget()
        right = QVBoxLayout(wrap)
        right.setContentsMargins(8, 8, 8, 8)
        right.setSpacing(6)
        self.overview = QLabel("尚未分析")
        self.overview.setWordWrap(True)
        right.addWidget(self.overview)
        self.result_tabs = QTabWidget()
        self.result_tabs.setUsesScrollButtons(False)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["根", "级次", "长度", "直径", "状态"])
        self.tree.setMinimumHeight(140)
        self.tree.itemSelectionChanged.connect(self._tree_selected)
        self.result_tabs.addTab(self.tree, "根系")
        self.result_tabs.setTabToolTip(0, "根与父子关系")
        self.summary = QTableWidget()
        self.summary.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.result_tabs.addTab(self.summary, "统计")
        self.result_tabs.setTabToolTip(1, "级次统计")
        right.addWidget(self.result_tabs, 1)

        review = QWidget()
        form = QFormLayout(review)
        self.junction_combo = QComboBox()
        self.junction_combo.currentIndexChanged.connect(self._junction_changed)
        self.junction_combo.activated.connect(lambda _: self.mode.setCurrentIndex(self.mode.findData('junction')))
        form.addRow("交汇", self.junction_combo)
        self.alternative_combo = QComboBox()
        self.alternative_combo.currentIndexChanged.connect(self._alternative_changed)
        self.alternative_combo.activated.connect(lambda _: self.mode.setCurrentIndex(self.mode.findData('junction')))
        form.addRow("解释", self.alternative_combo)
        self.decision_help = QLabel("选择交汇，画布将显示端口编号和候选接续。")
        self.decision_help.setWordWrap(True)
        form.addRow(self.decision_help)
        choices = QHBoxLayout()
        self.resolve_btn = self._button("确认该解释", self._resolve, choices)
        self.defer_btn = self._button("暂不确定", self._defer, choices)
        form.addRow(choices)
        self.result_tabs.addTab(_scrolled(review), "交汇")
        self.result_tabs.setTabToolTip(2, "交汇校正")

        editing = QWidget()
        edits = QFormLayout(editing)
        self.root_info = QLabel("请先选择一条根")
        self.root_info.setWordWrap(True)
        edits.addRow(self.root_info)
        self.node_combo = QComboBox()
        self.node_combo.currentIndexChanged.connect(self._node_changed)
        edits.addRow("节点", self.node_combo)
        self.diameter = _NoWheelDoubleSpinBox()
        self.diameter.setRange(0.01, 100000)
        self.diameter.setDecimals(3)
        self.diameter.setSuffix(" px")
        diameter_row = QHBoxLayout()
        diameter_row.addWidget(self.diameter)
        self.diameter_btn = self._button("设置直径", self._set_diameter, diameter_row)
        edits.addRow(diameter_row)
        self.parent_combo = QComboBox()
        self.parent_combo.currentIndexChanged.connect(self._parent_changed)
        edits.addRow("父根", self.parent_combo)
        self.parent_node_combo = QComboBox()
        edits.addRow("附着节点", self.parent_node_combo)
        parent_row = QHBoxLayout()
        self.reparent_btn = self._button("更新附着", self._reparent, parent_row)
        self.detach_btn = self._button("解除附着", lambda: self._root_edit("detach"), parent_row)
        edits.addRow(parent_row)
        root_row = QHBoxLayout()
        self.confirm_btn = self._button("确认根", lambda: self._root_edit("confirm_root"), root_row)
        self.reverse_btn = self._button("反向", lambda: self._root_edit("reverse"), root_row)
        self.delete_btn = self._button("删除根", self._delete_root, root_row,
            "删除所选根，保留其子根为待归属对象；可以撤销")
        edits.addRow(root_row)
        self.result_tabs.addTab(_scrolled(editing), "编辑")
        self.result_tabs.setTabToolTip(3, "所选根与节点")
        return wrap

    def _show_error(self, message):
        self.status.setText(message)
        QMessageBox.warning(self, "根系表型beta", message)

    def _history_changed(self, can_undo, can_redo):
        self.undo_btn.setEnabled(can_undo)
        self.redo_btn.setEnabled(can_redo)

    def _mode_changed(self, _index=None):
        mode = self.mode.currentData()
        self.canvas.set_mode(mode)
        if mode == "junction":
            self.result_tabs.setCurrentIndex(2)
        elif mode == "nodes":
            self.result_tabs.setCurrentIndex(3)
        if not self.controller.busy:
            self.status.setText(_MODE_HINTS.get(mode, ""))

    def _busy_changed(self, busy):
        self._progress.setVisible(busy)
        self.analyze_btn.setEnabled(not busy and bool(self._mask_path) and bool(self._base_marks))
        self.remove_base_btn.setEnabled(bool(self._base_marks))
        self.cancel_btn.setEnabled(busy)
        self.resolve_btn.setEnabled(not busy and self._source_valid and self.junction_combo.count() > 0)
        self.defer_btn.setEnabled(not busy and self._source_valid and self.junction_combo.count() > 0)

    def _choose_mask(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择根系掩码", "", "图像 (*.png *.tif *.tiff *.bmp *.jpg *.jpeg)")
        if path:
            try:
                self._leave_dataset_mode()
                self.load_mask(path)
            except Exception as exc:
                self._show_error(str(exc))

    def _leave_dataset_mode(self):
        self._current_sample = None
        if self._dataset_panel is not None:
            self._dataset_panel.clear_selection()

    def _fill_class_combo(self, ds):
        prev = self._class_combo.currentData()
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        for c in ds.meta.foreground_classes:
            self._class_combo.addItem(f"{c.id} · {c.name}", c.id)
        idx = self._class_combo.findData(prev)
        if idx < 0:
            idx = next((i for i, c in enumerate(ds.meta.foreground_classes) if c.name == "根"), 0)
        self._class_combo.setCurrentIndex(idx)
        self._class_combo.blockSignals(False)

    def _on_class_combo_changed(self, _index):
        if self._current_sample is not None:
            self._on_dataset_sample_selected(*self._current_sample)

    def _on_dataset_sample_selected(self, dataset_id, sample_id):
        try:
            ds = self._platform_app.datasets.store.load(dataset_id)
            sample = ds.sample(sample_id)
            if sample is None:
                return
            self._fill_class_combo(ds)
            mask_path = self._dataset_mask_file(ds, sample, self._class_combo.currentData())
            image_path = ds.abs(sample.image)
        except Exception as exc:
            self._show_error(f"无法读取样本掩码: {exc}")
            return
        self._current_sample = (dataset_id, sample_id)
        try:
            self.load_mask(mask_path)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self._current_sample = (dataset_id, sample_id)
        self.mask_name.setText(f"{ds.name} / {sample.image_name}")
        if image_path.is_file():
            image_size = QImageReader(str(image_path)).size()
            mask_size = QImageReader(str(mask_path)).size()
            if (image_size.width(), image_size.height()) == (mask_size.width(), mask_size.height()):
                self._original_path = str(image_path)
            else:
                self.status.setText(f"已打开「{sample.image_name}」；原图尺寸与掩码不一致，只能显示掩码。")
        self._apply_dataset_view()

    def _apply_dataset_view(self, _mode=None):
        if self._current_sample is None or self._mask_path is None:
            return
        mode = self._dataset_panel.display_mode() if self._dataset_panel is not None else "mask"
        path = self._original_path if mode == "image" and self._original_path else self._mask_path
        if mode == "image" and not self._original_path:
            self.status.setText("该样本没有可用原图，仍显示掩码。")
        self._view_path = str(path)
        if mode == "image" and self._original_path:
            self.canvas.set_background(path)
        else:
            self._show_mask_preview(path)

    def _show_mask_preview(self, path):
        from plant_phenotype.datasets.mask_io import preview_mask_rgb, read_raw_mask
        try:
            self.canvas.set_background_array(preview_mask_rgb(read_raw_mask(path)))
        except Exception:
            self.canvas.set_background(path)

    def _dataset_mask_file(self, ds, sample, class_id):
        """单前景语义掩码直接使用；多类别或实例按所选类别导出二值掩码。"""
        if class_id is None:
            raise ValueError("数据集未定义前景类别")
        fg_ids = [c.id for c in ds.meta.foreground_classes]
        if ds.meta.task_type == "semantic" and sample.mask and fg_ids == [class_id]:
            path = ds.abs(sample.mask)
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        if sample.instances:
            masks = [m for rec, m in ds.instance_masks(sample) if rec.class_id == class_id]
            if not masks:
                raise ValueError(f"该样本没有类别 {class_id} 的实例")
            fg = np.logical_or.reduce(masks)
        else:
            index = ds.load_index_mask(sample)
            if index is None:
                raise FileNotFoundError(sample.mask)
            fg = index == class_id
        out = self._platform_app.datasets.root / ".cache" / "root_masks" / ds.id / f"c{class_id}" / f"{sample.stem}.png"
        mask_io.write_binary_mask(out, fg)
        return out

    def load_mask(self, path):
        if not Path(path).is_file():
            raise ValueError("掩码不存在")
        if not QImageReader(str(path)).canRead():
            raise ValueError("无法读取掩码图像")
        self.controller.reset()
        self._mask_path = str(path)
        self._original_path = None
        self._source_valid = True
        self._base_marks = []
        self.mask_name.setText(str(path))
        self._show_mask_preview(path)
        self._mask_cache = None
        self.canvas.set_base_marks([])
        self.mode.setCurrentIndex(0)
        self._busy_changed(False)
        self.status.setText("请点击要识别的根，根内任意位置均可；解析完成后可继续点击下一株根。")

    def _choose_original(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择对应原图", "", "图像 (*.png *.tif *.tiff *.bmp *.jpg *.jpeg)")
        if path:
            self.controller.cancel()
            reference = self.controller.document.source if self.controller.document is not None else None
            size = QImageReader(path).size()
            mask_size = QImageReader(self._mask_path).size() if self._mask_path else None
            expected = (reference.width, reference.height) if reference else ((mask_size.width(), mask_size.height()) if mask_size is not None else None)
            if expected and (size.width(), size.height()) != expected:
                self._show_error("原图尺寸必须与掩码一致")
                return
            self._original_path = path
            self.canvas.set_background(path)
            self.status.setText("已切换显示原图；重新分析后会保存原图引用。")

    def _add_base(self, x, y):
        if self._mask_path is None:
            return
        mask = self._foreground_mask()
        col, row = round(x), round(y)
        if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]) or not mask[row, col]:
            self.status.setText('请点击根内的前景区域，当前点击在背景上。')
            return
        self._base_marks.append({"id": "base-"+uuid4().hex[:12], "x": x, "y": y})
        self.canvas.set_base_marks(self._base_marks)
        self._analyze()
        if self.controller.busy:
            self.status.setText(f"正在解析 {len(self._base_marks)} 个选点所在的根系…")

    def _foreground_mask(self):
        cache = getattr(self, "_mask_cache", None)
        if cache is None or cache[0] != self._mask_path:
            from plant_phenotype.root_phenotype.vector.features import load_mask
            cache = (self._mask_path, load_mask(self._mask_path))
            self._mask_cache = cache
        return cache[1]

    def _remove_base(self):
        if not self._base_marks:
            self.status.setText("当前没有选点。")
            return
        self._base_marks.pop()
        self.canvas.set_base_marks(self._base_marks)
        if self._base_marks and self._mask_path:
            self._analyze()
            return
        self.controller.reset()
        self._busy_changed(False)
        self.status.setText("已撤销全部选点；请点击根所在位置。")

    def _analyze(self):
        if not self._base_marks:
            self.status.setText('请先点击根内任意位置。')
            return
        if self._mask_path:
            self.controller.analyze(self._mask_path, [], self._original_path,
                selection_points=list(self._base_marks),
                config={"mm_per_pixel": self.scale.value() or None,
                        "connect_gap_px": self.gap_limit.value(), "spur_length_px": self.spur_limit.value()})

    def _document_changed(self, document):
        self._updating = True
        self.tree.clear()
        self._tree_items = {}
        self.junction_combo.clear()
        self.alternative_combo.clear()
        self.node_combo.clear()
        self.parent_combo.clear()
        self.parent_node_combo.clear()
        self.canvas.set_document(document)
        enabled = document is not None
        self.save_btn.setEnabled(enabled)
        self.scale_btn.setEnabled(enabled)
        for button in self.export_buttons:
            button.setEnabled(enabled)
        self.summary.clear()
        self.summary.setRowCount(0)
        self._tables = {}
        if document is None:
            self._selected_root_id = None
            self.overview.setText("尚未分析")
            self.root_info.setText("请先选择一条根")
            self._updating = False
            self._selection_enabled(False)
            return
        self._source_valid = self.facade.vector_source_matches(document)
        self._tables = self.facade.measure_vectors(document)
        self._base_marks = list(document.selection_points) or [{"id": b.id, "x": b.x, "y": b.y} for b in document.base_marks]
        self.canvas.set_base_marks(self._base_marks)
        self.scale.setValue(document.source.mm_per_pixel or 0)
        self.gap_limit.setValue(document.config.connect_gap_px)
        self.spur_limit.setValue(document.config.spur_length_px)
        rows = {row["root_id"]: row for row in self._tables["roots"].to_dict("records")}
        unit = "mm" if document.source.mm_per_pixel else "px"
        self.tree.setHeaderLabels(["根", "级次", "长度", "直径", "状态"])
        self.tree.headerItem().setToolTip(2, f"长度（{unit}）")
        self.tree.headerItem().setToolTip(3, f"平均直径（{unit}）")
        for index, root in enumerate(document.roots):
            row = rows.get(root.id, {})
            order = document.root_order(root.id)
            label = "主根" if order == 0 else (f"{order}级侧根" if order is not None else "待归属")
            item = QTreeWidgetItem([f"R{index+1}", label, _display(row.get("length_"+unit)),
                _display(row.get("mean_diameter_"+unit)), _STATE.get(root.review_state, root.review_state)])
            item.setData(0, Qt.ItemDataRole.UserRole, root.id)
            item.setToolTip(0, root.id)
            item.setForeground(0, QColor(self.facade.vector_root_color(root.id, document)))
            self._tree_items[root.id] = item
        for root in document.roots:
            item = self._tree_items[root.id]
            parent = self._tree_items.get(root.parent_attachment.parent_root_id) if root.parent_attachment else None
            if parent is not None:
                parent.addChild(item)
            else:
                self.tree.addTopLevelItem(item)
        self.tree.expandAll()
        for column in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(column)
        for index, d in enumerate(document.junction_decisions, 1):
            self.junction_combo.addItem(f"交汇 {index} · {_STATE.get(d.state, d.state)}", d.id)
        pending = sum(d.state != "accepted" for d in document.junction_decisions)
        unresolved = sum(document.root_order(r.id) is None for r in document.roots)
        primary = sum(document.root_order(r.id) == 0 for r in document.roots)
        lateral = len(document.roots)-unresolved-primary
        self.overview.setText(f"主根 {primary} 条 · 侧根 {lateral} 条 · 待归属 {unresolved} 段\n{pending} 处交汇待确认 · 单位 {unit}")
        self._populate_summary(self._tables["summary"])
        self._updating = False
        selected = self._selected_root_id if self._selected_root_id in self._tree_items else next(iter(self._tree_items), None)
        self._select_root(selected)
        self._junction_changed()
        self._busy_changed(self.controller.busy)
        messages = list(document.diagnostics[:3])
        if self.mode.currentData() == "base" and self._base_marks:
            messages.insert(0, f"解析完成（{len(self._base_marks)} 个选点），可继续点击下一株根追加分析")
        if messages:
            self.status.setText("；".join(messages))
        self.status.setToolTip("\n".join(document.diagnostics))

    def _populate_summary(self, table):
        doc = self.controller.document
        scale = doc.source.mm_per_pixel if doc else None
        unit = "mm" if scale else "px"
        columns = ["scope", "order", "review_state", "completeness", "root_count", "unassigned_count",
                   "known_root_length_px", "mean_root_length_px", "mean_diameter_px", "diameter_coverage"]
        columns = [c for c in columns if c in table.columns]
        self.summary.setColumnCount(len(columns))
        self.summary.setRowCount(len(table))
        labels = {"scope": "范围", "order": "级次", "review_state": "状态", "completeness": "完整性",
                  "root_count": "已归属根数", "unassigned_count": "待归属段数",
                  "known_root_length_px": f"已归属长度({unit})", "mean_root_length_px": f"平均根长({unit})",
                  "mean_diameter_px": f"几何均径({unit})", "diameter_coverage": "直径覆盖率"}
        self.summary.setHorizontalHeaderLabels([labels[c] for c in columns])
        for r, row in enumerate(table.to_dict("records")):
            for c, key in enumerate(columns):
                value = row[key]
                if key == "scope":
                    text = "全部" if value == "all" else "分组"
                elif key == "order":
                    text = "—" if row.get("scope") == "all" else ("待归属" if value is None or value != value else ("主轴" if value == 0 else f"{int(value)}级侧根"))
                elif key.endswith("_px"):
                    text = _display(value * scale if scale and value is not None else value)
                elif key.endswith("_count"):
                    text = _display(value, 0)
                elif key == "diameter_coverage":
                    text = _display(value * 100, 1)+"%"
                else:
                    text = _STATE.get(str(value), "—" if value is None else str(value))
                item = QTableWidgetItem(text)
                if key == "mean_diameter_px":
                    item.setToolTip("当前分组全部可见矢量几何的有效直径，包含待归属片段；按有效测量长度加权")
                self.summary.setItem(r, c, item)
        self.summary.resizeColumnsToContents()

    def _tree_selected(self):
        if not self._updating:
            items = self.tree.selectedItems()
            if items:
                self._select_root(items[0].data(0, Qt.ItemDataRole.UserRole))

    def _selection_enabled(self, enabled):
        for w in (self.node_combo, self.diameter, self.diameter_btn, self.parent_combo,
                  self.parent_node_combo, self.reparent_btn, self.detach_btn, self.confirm_btn,
                  self.reverse_btn, self.delete_btn):
            w.setEnabled(enabled)

    def _select_root(self, root_id):
        doc = self.controller.document
        self._selected_root_id = root_id
        self._selection_enabled(root_id is not None and doc is not None)
        if root_id is None or doc is None:
            return
        root = doc.root_by_id(root_id)
        self._updating = True
        self.tree.setCurrentItem(self._tree_items.get(root_id))
        self.canvas.select_root(root_id)
        self.node_combo.clear()
        for i, node in enumerate(root.nodes):
            self.node_combo.addItem(f"{i+1}: ({node.x:.1f}, {node.y:.1f})", node.id)
        self.parent_combo.clear()
        for i, parent in enumerate(doc.roots):
            if parent.id != root_id:
                self.parent_combo.addItem(f"R{i+1} · {parent.id[:12]}", parent.id)
        if root.parent_attachment:
            self.parent_combo.setCurrentIndex(self.parent_combo.findData(root.parent_attachment.parent_root_id))
        self.root_info.setText(f"{root_id}\n{_STATE.get(root.direction, root.direction)} · {_STATE.get(root.completeness, root.completeness)}")
        self._updating = False
        self._parent_changed()
        if root.parent_attachment:
            self.parent_node_combo.setCurrentIndex(self.parent_node_combo.findData(root.parent_attachment.parent_node_id))
        self._node_changed()

    def _node_changed(self):
        if self._updating or self.controller.document is None or not self._selected_root_id:
            return
        root = self.controller.document.root_by_id(self._selected_root_id)
        node = next((n for n in root.nodes if n.id == self.node_combo.currentData()), None)
        if node:
            self.diameter.setValue(node.diameter_px or .01)
            self.diameter.setToolTip("直径缺失；设置后记为人工测量" if node.diameter_px is None else node.diameter_source)
            self.canvas.selected_node_id = node.id
            self.canvas._redraw()

    def _select_node(self, root_id, node_id):
        if self._selected_root_id != root_id:
            self._select_root(root_id)
        self.node_combo.setCurrentIndex(self.node_combo.findData(node_id))

    def _move_node(self, root_id, node_id, x, y):
        self.controller.apply_edit({"type": "move_node", "root_id": root_id, "node_id": node_id, "x": x, "y": y})

    def _set_diameter(self):
        if self._selected_root_id and self.node_combo.currentData():
            self.controller.apply_edit({"type": "set_diameter", "root_id": self._selected_root_id,
                "node_id": self.node_combo.currentData(), "diameter_px": self.diameter.value()})

    def _parent_changed(self):
        if self._updating:
            return
        self.parent_node_combo.clear()
        doc = self.controller.document
        if doc is not None and self.parent_combo.currentData():
            parent = doc.root_by_id(self.parent_combo.currentData())
            for i, node in enumerate(parent.nodes):
                self.parent_node_combo.addItem(f"{i+1}: ({node.x:.1f}, {node.y:.1f})", node.id)

    def _reparent(self):
        if self._selected_root_id and self.parent_node_combo.currentData():
            self.controller.apply_edit({"type": "reparent", "root_id": self._selected_root_id,
                "parent_root_id": self.parent_combo.currentData(), "parent_node_id": self.parent_node_combo.currentData()})

    def _root_edit(self, kind):
        if self._selected_root_id:
            self.controller.apply_edit({"type": kind, "root_id": self._selected_root_id})

    def _delete_root(self):
        if self._selected_root_id:
            self.controller.apply_edit({"type": "delete_root", "root_id": self._selected_root_id, "subtree": False})

    def _set_scale(self):
        self.controller.apply_edit({"type": "set_scale", "mm_per_pixel": self.scale.value() or None})

    def _scale_input_changed(self):
        if not self._updating and self.controller.busy:
            self.controller.cancel()
            self.status.setText("分析参数已改变，已取消旧任务；请重新解析或应用标定。")

    def _select_junction(self, junction_id):
        self.result_tabs.setCurrentIndex(2)
        self.junction_combo.setCurrentIndex(self.junction_combo.findData(junction_id))
        self._junction_changed()

    def _junction_changed(self, *_):
        if self._updating:
            return
        self.alternative_combo.blockSignals(True)
        self.alternative_combo.clear()
        doc = self.controller.document
        junction_id = self.junction_combo.currentData()
        decision = next((d for d in doc.junction_decisions if d.id == junction_id), None) if doc else None
        if decision:
            for alt in decision.alternatives:
                ports = {p: str(i+1) for i, p in enumerate(decision.port_ids)}
                pairs = "、".join("↔".join(ports.get(p, p) for p in pair) for pair in alt.pairs)
                kind = {"branch": "分叉", "crossing": "穿越", "continuation": "同根接续", "stop": "停止接续"}.get(alt.kind, alt.kind)
                self.alternative_combo.addItem(f"{kind} {pairs} · 代价 {alt.cost:.3f}", alt.id)
            if decision.chosen:
                self.alternative_combo.setCurrentIndex(self.alternative_combo.findData(decision.chosen))
        self.alternative_combo.blockSignals(False)
        self._alternative_changed()
        self._busy_changed(self.controller.busy)

    def _alternative_changed(self, *_):
        if self._updating:
            return
        doc = self.controller.document
        jid, aid = self.junction_combo.currentData(), self.alternative_combo.currentData()
        decision = next((d for d in doc.junction_decisions if d.id == jid), None) if doc else None
        alt = next((a for a in decision.alternatives if a.id == aid), None) if decision else None
        self.canvas.highlight_junction(jid, alt.pairs if alt else None)
        if alt is not None:
            numbers = {pid: str(i+1) for i, pid in enumerate(decision.port_ids)}
            pairs = "；".join(" ↔ ".join(numbers[p] for p in pair) for pair in alt.pairs)
            explanation = {"crossing": "两组根分别穿越，不建立父子关系。",
                           "branch": "所选两段延续同一条根，其余侧支附着于父根。",
                           "continuation": "所选两段延续为同一条根。",
                           "stop": "在此停止追踪，保留未解析部分。"}.get(alt.kind, "")
            self.decision_help.setText((f"端口 {pairs}。" if pairs else "") + explanation)
        else:
            self.decision_help.setText("暂无交汇需要选择。")

    def _resolve(self):
        jid, aid = self.junction_combo.currentData(), self.alternative_combo.currentData()
        if jid is not None and aid is not None:
            self.controller.resolve(jid, aid)

    def _defer(self):
        jid = self.junction_combo.currentData()
        if jid is not None:
            self.controller.resolve(jid, None)

    def _width_toggled(self, enabled):
        self.canvas.show_diameters = enabled
        self.canvas._redraw()

    def _open_document(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开矢量根系文档", "", "根系文档 (*.json)")
        if path:
            try:
                self.load_document(path)
            except Exception as exc:
                self._show_error(str(exc))

    def load_document(self, path):
        doc = self.facade.load_vector_document(path)
        self._mask_path = doc.source.mask_path if doc.source.mask_path and Path(doc.source.mask_path).is_file() else None
        self._original_path = doc.source.image_path if doc.source.image_path and Path(doc.source.image_path).is_file() else None
        self.mask_name.setText(doc.source.mask_path or "矢量文档（无源掩码）")
        background = self._original_path or self._mask_path
        if background:
            size = QImageReader(background).size()
            if (size.width(), size.height()) != (doc.source.width, doc.source.height):
                background = None
        self.canvas.set_background(background, doc.source.width, doc.source.height)
        self.controller.load_document(doc)
        self.canvas.fit_image()
        if not self._mask_path:
            self.status.setText("源掩码不可用；仍可查看、编辑和导出已保存的矢量结果。")
        elif not self._source_valid:
            self.status.setText("源掩码与文档哈希不一致；接续校正已停用，请重新分析。现有矢量仍可编辑和导出。")
        self._busy_changed(False)

    def _save_document(self):
        if self.controller.document is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存矢量根系文档", "root_vectors.json", "根系文档 (*.json)")
        if path:
            try:
                self.facade.save_vector_document(self.controller.document, path)
                self.status.setText(f"已保存：{path}")
            except Exception as exc:
                self._show_error(str(exc))

    def _export(self, extension):
        doc = self.controller.document
        if doc is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出根系结果", f"root_vectors.{extension}", f"{extension.upper()} (*.{extension})")
        if not path:
            return
        if not Path(path).suffix:
            path += "."+extension
        try:
            if extension in ("csv", "xlsx"):
                self.facade.export_vector_tables(doc, path)
            elif extension == "svg":
                self.facade.export_vector_svg(doc, path)
            else:
                self.canvas.export_png(path)
            self.status.setText(f"已导出：{path}")
        except Exception as exc:
            self._show_error(str(exc))
