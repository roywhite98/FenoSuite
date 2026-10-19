# -*- coding: utf-8 -*-
"""根系表型 Tab：依赖用户主动选根。流程：读 mask → 预处理 → 等待用户点选根 → 分析 → 可校正主根、可撤销点选。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QWheelEvent, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QFormLayout,
    QDoubleSpinBox,
    QSplitter,
    QSpinBox,
    QCheckBox,
    QScrollArea,
    QFrame,
    QProgressBar,
    QSizePolicy,
    QGroupBox,
    QComboBox,
    QTabWidget,
)

from plant_phenotype.datasets import mask_io
from plant_phenotype.gui.dataset_panel import DatasetSamplePanel
from plant_phenotype.gui.workdir_tree import WorkdirTreePanel
from plant_phenotype.gui.root_canvas import RootCanvas
from plant_phenotype.gui.root_phenotype_stats_panel import RootPhenotypeStatsPanel
from plant_phenotype.gui.utils import equalize_widths
from plant_phenotype.root_phenotype.facade import get_facade

_DEFAULTS = get_facade().default_params()


class _NoWheelSpinBox(QSpinBox):
    """滚轮不改变数值，避免误触。"""
    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class _NoWheelDoubleSpinBox(QDoubleSpinBox):
    """滚轮不改变数值，避免误触。"""
    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class _CollapsibleGroupBox(QWidget):
    """带折叠/展开按钮的参数组；高级参数默认折叠，风格贴近 QGroupBox 卡片。"""

    def __init__(self, title: str, parent=None, collapsed: bool = False):
        super().__init__(parent)
        self.setProperty("mdRole", "card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle_btn = QPushButton(f"{'▼' if not collapsed else '▶'}  {title}")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(not collapsed)
        self._toggle_btn.setProperty("mdRole", "text")
        self._toggle_btn.setToolTip("点击展开/折叠")
        self._toggle_btn.clicked.connect(self._on_toggle)
        outer.addWidget(self._toggle_btn)

        self._content = QWidget()
        outer.addWidget(self._content)
        self._content.setVisible(not collapsed)

        self._layout = QVBoxLayout(self._content)
        self._layout.setContentsMargins(8, 4, 8, 8)
        self._layout.setSpacing(4)

        self._title = title

    def _on_toggle(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self._toggle_btn.setText(f"{'▼' if checked else '▶'}  {self._title}")

    def setLayout(self, layout) -> None:  # type: ignore[override]
        """把 layout 中的所有条目迁移到内部 _layout，避免 Qt 的 setLayout 重复赋值限制。"""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                self._layout.addWidget(item.widget())
            elif item.layout():
                self._layout.addLayout(item.layout())
            elif item.spacerItem():
                self._layout.addItem(item.spacerItem())

    def inner_layout(self):
        return self._layout


def _imread_unicode(path: str, flags=None):
    import cv2
    if flags is None:
        flags = cv2.IMREAD_COLOR
    with open(path, "rb") as f:
        import numpy as np
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, flags)


def _find_original_image(mask_path: Path, image_dir: Path | None) -> str | None:
    """同目录下与 mask 同主名的原图（不同扩展名），用于叠加底图。"""
    if image_dir is None:
        image_dir = mask_path.parent
    stem = mask_path.stem
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        cand = image_dir / (stem + ext)
        if cand != mask_path and cand.exists():
            return str(cand)
    return None


class RootPhenotypeTab(QWidget):
    """流程：选图 → 预处理（等待选根）→ 用户点选根 → 分析 → 可校正主根、可撤销点选。无批量。"""

    MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(self, parent=None, platform_app=None):
        super().__init__(parent)
        self._platform_app = platform_app
        self._current_path: Path | None = None
        self._image_dir: Path | None = None
        self._current_sample: tuple[str, str] | None = None  # (dataset_id, sample_id)
        self._dataset_image_path: Path | None = None
        self._current_result = None
        self._preprocessed_mask = None
        self._selected_seeds: List[Tuple[int, int]] = []
        self._last_corrected_stroke: List[Tuple[int, int]] | None = None
        self._result_before_correction = None
        self._main_root_correction_mode = False
        self._task_id: str | None = None
        self._task_logs_shown = 0

        main_layout = QHBoxLayout(self)
        if platform_app is not None:
            platform_app.tasks.task_state_changed.connect(self._on_task_state_changed)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ---- 左侧：可滚动参数面板（QGroupBox 主区 + 高级折叠）----
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        param_container = QWidget()
        param_container.setObjectName("rootPhenotypeParams")
        param_layout = QVBoxLayout(param_container)
        param_layout.setContentsMargins(10, 10, 10, 10)
        param_layout.setSpacing(10)

        path_grp = QGroupBox("输入")
        path_ly = QVBoxLayout(path_grp)
        path_ly.setContentsMargins(8, 10, 8, 8)
        path_ly.setSpacing(6)
        path_ly.addWidget(QLabel("掩码图像:"))
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("选择单张掩码图像")
        self._path_edit.editingFinished.connect(self._on_path_changed)
        path_ly.addWidget(self._path_edit)
        btn_file = QPushButton("选择文件...")
        btn_file.clicked.connect(self._browse_file)
        path_ly.addWidget(btn_file)
        param_layout.addWidget(path_grp)

        disp_grp = QGroupBox("显示")
        disp_ly = QVBoxLayout(disp_grp)
        disp_ly.setContentsMargins(8, 10, 8, 8)
        disp_ly.setSpacing(4)
        self._show_skeleton = QCheckBox("显示骨架")
        self._show_skeleton.setChecked(True)
        disp_ly.addWidget(self._show_skeleton)
        self._show_junctions = QCheckBox("显示交点")
        self._show_junctions.setChecked(True)
        disp_ly.addWidget(self._show_junctions)
        self._show_endpoints = QCheckBox("显示根尖")
        self._show_endpoints.setChecked(True)
        disp_ly.addWidget(self._show_endpoints)
        self._show_main = QCheckBox("显示主根")
        self._show_main.setChecked(True)
        disp_ly.addWidget(self._show_main)
        self._show_secondary = QCheckBox("显示二级根")
        self._show_secondary.setChecked(True)
        disp_ly.addWidget(self._show_secondary)
        param_layout.addWidget(disp_grp)

        self._restore_default_btn = QPushButton("恢复默认")
        self._restore_default_btn.setProperty("mdRole", "tonal")
        self._restore_default_btn.clicked.connect(self._restore_defaults)
        param_layout.addWidget(self._restore_default_btn)

        pre_grp = QGroupBox("预处理")
        pre_ly = QFormLayout(pre_grp)
        pre_ly.setContentsMargins(8, 10, 8, 8)
        pre_ly.setSpacing(6)
        self._morph_kernel = _NoWheelSpinBox()
        self._morph_kernel.setRange(1, 15)
        self._morph_kernel.setValue(_DEFAULTS["morph_kernel_size"])
        pre_ly.addRow("形态学核大小:", self._morph_kernel)
        self._min_component_area = _NoWheelSpinBox()
        self._min_component_area.setRange(1, 10000)
        self._min_component_area.setValue(_DEFAULTS["min_component_area"])
        pre_ly.addRow("最小连通域面积:", self._min_component_area)
        self._morph_open = QCheckBox("开运算")
        self._morph_open.setChecked(_DEFAULTS["morph_open"])
        pre_ly.addRow(self._morph_open)
        param_layout.addWidget(pre_grp)

        stats_grp = QGroupBox("统计与清洗")
        stats_ly = QFormLayout(stats_grp)
        stats_ly.setContentsMargins(8, 10, 8, 8)
        stats_ly.setSpacing(6)
        self._min_root_length = _NoWheelDoubleSpinBox()
        self._min_root_length.setRange(0, 1000)
        self._min_root_length.setValue(_DEFAULTS["min_root_length"])
        stats_ly.addRow("短根长度阈值:", self._min_root_length)
        self._min_root_diameter = _NoWheelDoubleSpinBox()
        self._min_root_diameter.setRange(0, 50)
        self._min_root_diameter.setValue(_DEFAULTS["min_root_diameter"])
        stats_ly.addRow("细根直径阈值:", self._min_root_diameter)
        self._zscore = QCheckBox("Z-score 标准化")
        self._zscore.setChecked(_DEFAULTS["zscore"])
        stats_ly.addRow(self._zscore)
        param_layout.addWidget(stats_grp)

        # ── 高级参数（默认折叠）──
        conn_grp = _CollapsibleGroupBox("断根连接", collapsed=True)
        conn_ly = QFormLayout()
        self._connect_enabled = QCheckBox("启用")
        self._connect_enabled.setChecked(_DEFAULTS["connect_enabled"])
        conn_ly.addRow(self._connect_enabled)
        self._connect_max_search = _NoWheelSpinBox()
        self._connect_max_search.setRange(10, 500)
        self._connect_max_search.setValue(_DEFAULTS["connect_max_search"])
        conn_ly.addRow("最大搜索:", self._connect_max_search)
        self._connect_angle_tol = _NoWheelDoubleSpinBox()
        self._connect_angle_tol.setRange(5, 90)
        self._connect_angle_tol.setValue(_DEFAULTS["connect_angle_tol_deg"])
        conn_ly.addRow("角度容差(度):", self._connect_angle_tol)
        self._connect_min_gap = _NoWheelDoubleSpinBox()
        self._connect_min_gap.setRange(0.5, 50)
        self._connect_min_gap.setValue(_DEFAULTS["connect_min_gap"])
        conn_ly.addRow("最小间隙:", self._connect_min_gap)
        self._connect_max_gap = _NoWheelDoubleSpinBox()
        self._connect_max_gap.setRange(10, 500)
        self._connect_max_gap.setValue(_DEFAULTS["connect_max_gap"])
        conn_ly.addRow("最大间隙:", self._connect_max_gap)
        conn_grp.setLayout(conn_ly)
        param_layout.addWidget(conn_grp)

        topo_grp = _CollapsibleGroupBox("拓扑", collapsed=True)
        topo_ly = QFormLayout()
        self._junction_min = _NoWheelSpinBox()
        self._junction_min.setRange(2, 8)
        self._junction_min.setValue(_DEFAULTS["junction_min_neighbors"])
        topo_ly.addRow("分叉点最少邻域数:", self._junction_min)
        self._spur_max = _NoWheelSpinBox()
        self._spur_max.setRange(0, 64)
        self._spur_max.setValue(int(_DEFAULTS.get("spur_max_length", 8)))
        self._spur_max.setToolTip("删除叶到分叉前长度 ≤ 该值的短毛刺；0 关闭。可减少根毛交点、加速建图")
        topo_ly.addRow("短毛刺剪枝长度:", self._spur_max)
        topo_grp.setLayout(topo_ly)
        param_layout.addWidget(topo_grp)

        weight_grp = _CollapsibleGroupBox("边权权重", collapsed=True)
        weight_ly = QFormLayout()
        self._w_length = _NoWheelDoubleSpinBox()
        self._w_length.setRange(0, 10)
        self._w_length.setValue(_DEFAULTS["weight_length"])
        weight_ly.addRow("长度:", self._w_length)
        self._w_diameter = _NoWheelDoubleSpinBox()
        self._w_diameter.setRange(0, 10)
        self._w_diameter.setValue(_DEFAULTS["weight_diameter"])
        weight_ly.addRow("直径:", self._w_diameter)
        self._w_curvature = _NoWheelDoubleSpinBox()
        self._w_curvature.setRange(-2, 2)
        self._w_curvature.setValue(_DEFAULTS["weight_curvature"])
        weight_ly.addRow("曲率:", self._w_curvature)
        weight_grp.setLayout(weight_ly)
        param_layout.addWidget(weight_grp)

        # ── 分叉竞争主根 ──
        primary_grp = _CollapsibleGroupBox("分叉竞争主根", collapsed=True)
        pri_ly = QFormLayout()

        self._primary_enabled = QCheckBox("启用（多主根 / 递归分级）")
        self._primary_enabled.setChecked(_DEFAULTS["primary_enabled"])
        self._primary_enabled.setToolTip(
            "开启后以分叉竞争取代「全局最长路径」：\n"
            "在每个分叉点按粗细·子树·连续性得分竞争，\n"
            "得分 ≥ τ × 最高分的边均视为主干延续（多赢家），\n"
            "直根系只有一条赢，须根系可同时多条赢。"
        )
        pri_ly.addRow(self._primary_enabled)

        self._primary_tau = _NoWheelDoubleSpinBox()
        self._primary_tau.setRange(0.30, 1.00)
        self._primary_tau.setSingleStep(0.05)
        self._primary_tau.setDecimals(2)
        self._primary_tau.setValue(_DEFAULTS["primary_tau"])
        self._primary_tau.setToolTip("多赢家阈值 τ：越小越容易同时胜出多条主根")
        pri_ly.addRow("多赢家阈值 τ:", self._primary_tau)

        self._primary_alpha = _NoWheelDoubleSpinBox()
        self._primary_alpha.setRange(0, 5)
        self._primary_alpha.setSingleStep(0.1)
        self._primary_alpha.setDecimals(2)
        self._primary_alpha.setValue(_DEFAULTS["primary_alpha"])
        self._primary_alpha.setToolTip("粗细权重 α：越大越看重边的直径")
        pri_ly.addRow("粗细权重 α:", self._primary_alpha)

        self._primary_beta = _NoWheelDoubleSpinBox()
        self._primary_beta.setRange(0, 5)
        self._primary_beta.setSingleStep(0.1)
        self._primary_beta.setDecimals(2)
        self._primary_beta.setValue(_DEFAULTS["primary_beta"])
        self._primary_beta.setToolTip("子树支持权重 β：越大越看重下游结构规模")
        pri_ly.addRow("子树支持 β:", self._primary_beta)

        self._primary_gamma = _NoWheelDoubleSpinBox()
        self._primary_gamma.setRange(0, 5)
        self._primary_gamma.setSingleStep(0.1)
        self._primary_gamma.setDecimals(2)
        self._primary_gamma.setValue(_DEFAULTS["primary_gamma"])
        self._primary_gamma.setToolTip("连续性权重 γ：越大越倾向走直线延续")
        pri_ly.addRow("连续性 γ:", self._primary_gamma)

        self._primary_max_level = _NoWheelSpinBox()
        self._primary_max_level.setRange(1, 8)
        self._primary_max_level.setValue(_DEFAULTS["primary_max_level"])
        self._primary_max_level.setToolTip("递归分级最大层数（1=仅主根，3=主根+二级+三级）")
        pri_ly.addRow("最大分级层数:", self._primary_max_level)

        self._primary_diam_ratio = _NoWheelDoubleSpinBox()
        self._primary_diam_ratio.setRange(0, 1)
        self._primary_diam_ratio.setSingleStep(0.05)
        self._primary_diam_ratio.setDecimals(2)
        self._primary_diam_ratio.setValue(_DEFAULTS["primary_diameter_ratio_min"])
        self._primary_diam_ratio.setToolTip("直径比下限：子边/母边 < 此值则降级为侧枝")
        pri_ly.addRow("直径比下限:", self._primary_diam_ratio)

        self._primary_diam_floor = _NoWheelDoubleSpinBox()
        self._primary_diam_floor.setRange(0, 20)
        self._primary_diam_floor.setSingleStep(0.5)
        self._primary_diam_floor.setDecimals(1)
        self._primary_diam_floor.setValue(_DEFAULTS["primary_diameter_floor"])
        self._primary_diam_floor.setToolTip("直径截断：低于此直径的边不再递归分级")
        pri_ly.addRow("直径截断:", self._primary_diam_floor)

        primary_grp.setLayout(pri_ly)
        param_layout.addWidget(primary_grp)

        param_layout.addStretch()
        scroll.setWidget(param_container)
        left_layout.addWidget(scroll)

        tools_scroll = QScrollArea()
        tools_scroll.setObjectName("rootPhenotypeSideBar")
        tools_scroll.setProperty("mdDensity", "compact")
        tools_scroll.setWidgetResizable(True)
        tools_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tools_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        tools_scroll.setFrameShape(QFrame.Shape.NoFrame)
        tools_scroll.setWidget(left_widget)
        tools_scroll.setMinimumWidth(260)
        tools_scroll.setMaximumWidth(340)

        # ---- 顶栏 + 画布 + 状态（与数据标注页顶栏样式一致）----
        self.canvas = RootCanvas()
        self.canvas.rootPixelClicked.connect(self._on_root_pixel_clicked)
        self.canvas.strokeConfirm.connect(self._on_main_root_stroke_confirmed)
        self.canvas.strokeConfirmInvalid.connect(self._on_main_root_stroke_confirm_invalid)
        self.canvas.strokeSegmentsChanged.connect(self._on_stroke_segments_changed)
        self.canvas.set_display_options(self._get_display_options())
        self.canvas.prevImageRequested.connect(self._go_prev_image)
        self.canvas.nextImageRequested.connect(self._go_next_image)

        self._shortcut_stroke_confirm = QShortcut(QKeySequence("Ctrl+F"), self)
        self._shortcut_stroke_confirm.setContext(Qt.ShortcutContext.WindowShortcut)
        self._shortcut_stroke_confirm.activated.connect(self._on_shortcut_stroke_confirm)
        self._shortcut_stroke_undo = QShortcut(QKeySequence("Ctrl+C"), self)
        self._shortcut_stroke_undo.setContext(Qt.ShortcutContext.WindowShortcut)
        self._shortcut_stroke_undo.activated.connect(self._on_shortcut_stroke_undo)

        self._stats_panel = RootPhenotypeStatsPanel()
        self._stats_panel.setMinimumWidth(180)

        self._workdir_panel = WorkdirTreePanel(
            self,
            title="",
            file_extensions=self.MASK_EXTENSIONS,
            settings_key="root_phenotype_last_work_directory",
        )
        self._workdir_panel.fileSelected.connect(self._on_workdir_file_selected)
        self._side_tabs = QTabWidget()
        self._dataset_panel: DatasetSamplePanel | None = None
        self._class_combo = QComboBox()
        if self._platform_app is not None:
            ds_wrap = QWidget()
            ds_layout = QVBoxLayout(ds_wrap)
            ds_layout.setContentsMargins(0, 0, 0, 0)
            ds_layout.setSpacing(0)
            self._dataset_panel = DatasetSamplePanel(
                self._platform_app.datasets, self,
                sample_filter=lambda s: s.has_annotation,
                empty_hint="当前数据集还没有带掩码的样本",
                default_display="mask",
            )
            self._dataset_panel.sampleSelected.connect(self._on_dataset_sample_selected)
            self._dataset_panel.displayModeChanged.connect(lambda _mode: self._update_preview())
            ds_layout.addWidget(self._dataset_panel, 1)
            class_row = QHBoxLayout()
            class_row.setContentsMargins(8, 0, 8, 8)
            class_row.addWidget(QLabel("分析类别:"))
            self._class_combo.setToolTip("多类别/实例数据集只取该类别的像素做根系分析")
            self._class_combo.currentIndexChanged.connect(self._on_class_combo_changed)
            class_row.addWidget(self._class_combo, 1)
            ds_layout.addLayout(class_row)
            self._side_tabs.addTab(ds_wrap, "数据集")
        self._side_tabs.addTab(self._workdir_panel, "文件夹")

        stats_header = QLabel("表型统计")
        stats_wrap = QWidget()
        stats_wrap_layout = QVBoxLayout(stats_wrap)
        stats_wrap_layout.setContentsMargins(8, 8, 8, 8)
        stats_wrap_layout.setSpacing(6)
        stats_wrap_layout.addWidget(stats_header)
        stats_wrap_layout.addWidget(self._stats_panel, 1)

        top_bar = QWidget()
        top_bar.setObjectName("rootPhenotypeTopBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(6, 2, 6, 2)
        top_layout.setSpacing(4)
        top_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        btn_pick = QPushButton("选择掩码...")
        btn_pick.setToolTip("选择单张掩码图像文件")
        btn_pick.clicked.connect(self._browse_file)
        top_layout.addWidget(btn_pick)

        self._preprocess_btn = QPushButton("预处理")
        self._preprocess_btn.setProperty("mdRole", "filled")
        self._preprocess_btn.setToolTip("加载并预处理当前掩码，完成后请在图上点击根所在位置")
        self._preprocess_btn.clicked.connect(self._on_preprocess)
        top_layout.addWidget(self._preprocess_btn)

        self._undo_btn = QPushButton("撤销上次点选")
        self._undo_btn.clicked.connect(self._on_undo_root)
        top_layout.addWidget(self._undo_btn)
        self._shortcut_undo = QShortcut(QKeySequence.StandardKey.Undo, self)
        self._shortcut_undo.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._shortcut_undo.activated.connect(self._on_undo_root)

        self._correct_main_btn = QPushButton("校正主根")
        self._correct_main_btn.setToolTip(
            "拖拽描绘主根轨迹，松手即应用；按住 Shift 松手可多段，再 Ctrl+F 确认；Ctrl+C 撤销上次校正"
        )
        self._correct_main_btn.clicked.connect(self._on_start_main_root_correction)
        top_layout.addWidget(self._correct_main_btn)
        equalize_widths([btn_pick, self._preprocess_btn, self._undo_btn, self._correct_main_btn])

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setFrameShadow(QFrame.Shadow.Sunken)
        sep1.setFixedWidth(1)
        sep1.setMinimumHeight(18)
        sep1.setMaximumHeight(48)
        top_layout.addWidget(sep1)

        self._export_csv_btn = QPushButton("导出 CSV...")
        self._export_csv_btn.clicked.connect(self._export_csv)
        self._export_csv_btn.setEnabled(False)
        top_layout.addWidget(self._export_csv_btn)
        self._export_xlsx_btn = QPushButton("导出 Excel...")
        self._export_xlsx_btn.clicked.connect(self._export_xlsx)
        self._export_xlsx_btn.setEnabled(False)
        top_layout.addWidget(self._export_xlsx_btn)
        self._export_fig_btn = QPushButton("导出可视化图...")
        self._export_fig_btn.clicked.connect(self._export_visualization)
        self._export_fig_btn.setEnabled(False)
        top_layout.addWidget(self._export_fig_btn)
        equalize_widths([self._export_csv_btn, self._export_xlsx_btn, self._export_fig_btn])

        top_layout.addStretch(1)
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(220)
        self._progress_bar.setMaximumHeight(18)
        self._progress_bar.hide()
        top_layout.addWidget(self._progress_bar)

        top_bar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        self._status_label = QLabel(
            "请在右侧选择工作目录并在树中打开掩码，或左侧填写路径。← → 切换同级掩码；选图后点「预处理」。"
        )
        self._status_label.setWordWrap(True)
        self._status_label.setProperty("mdRole", "status")

        canvas_column = QWidget()
        canvas_col_layout = QVBoxLayout(canvas_column)
        canvas_col_layout.setContentsMargins(0, 0, 0, 0)
        canvas_col_layout.addWidget(self.canvas)
        canvas_col_layout.addWidget(self._status_label)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.addWidget(self._side_tabs)
        right_splitter.addWidget(stats_wrap)
        right_splitter.setSizes([240, 280])

        inner_splitter = QSplitter(Qt.Orientation.Horizontal)
        inner_splitter.setChildrenCollapsible(False)
        inner_splitter.setHandleWidth(6)
        inner_splitter.addWidget(canvas_column)
        inner_splitter.addWidget(right_splitter)
        inner_splitter.setSizes([760, 260])

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(top_bar)
        content_layout.addWidget(inner_splitter, 1)

        outer_splitter = QSplitter(Qt.Orientation.Horizontal)
        outer_splitter.setChildrenCollapsible(False)
        outer_splitter.setHandleWidth(6)
        outer_splitter.addWidget(tools_scroll)
        outer_splitter.addWidget(content_widget)
        outer_splitter.setStretchFactor(0, 0)
        outer_splitter.setStretchFactor(1, 1)
        outer_splitter.setSizes([280, 1200])

        main_layout.addWidget(outer_splitter, 1)

        # 显示选项变化时刷新画布
        for cb in (self._show_skeleton, self._show_junctions, self._show_endpoints, self._show_main, self._show_secondary):
            cb.stateChanged.connect(self._on_display_options_changed)

    def _get_params(self) -> dict:
        return {
            "morph_kernel_size": self._morph_kernel.value(),
            "min_component_area": self._min_component_area.value(),
            "morph_open": self._morph_open.isChecked(),
            "connect_broken_roots": self._connect_enabled.isChecked(),
            "connect_max_search": self._connect_max_search.value(),
            "connect_angle_tol_deg": self._connect_angle_tol.value(),
            "connect_min_gap": self._connect_min_gap.value(),
            "connect_max_gap": self._connect_max_gap.value(),
            "junction_min_neighbors": self._junction_min.value(),
            "spur_max_length": self._spur_max.value(),
            "weight_length": self._w_length.value(),
            "weight_diameter": self._w_diameter.value(),
            "weight_curvature": self._w_curvature.value(),
            "min_root_length": self._min_root_length.value(),
            "min_root_diameter": self._min_root_diameter.value(),
            "zscore": self._zscore.isChecked(),
            "primary_root_enabled": self._primary_enabled.isChecked(),
            "primary_alpha": self._primary_alpha.value(),
            "primary_beta": self._primary_beta.value(),
            "primary_gamma": self._primary_gamma.value(),
            "primary_tau": self._primary_tau.value(),
            "primary_max_level": self._primary_max_level.value(),
            "primary_diameter_ratio_min": self._primary_diam_ratio.value(),
            "primary_diameter_floor": self._primary_diam_floor.value(),
            "primary_normalize_score": True,
        }

    def _get_display_options(self) -> dict:
        return {
            "show_skeleton": self._show_skeleton.isChecked(),
            "show_junctions": self._show_junctions.isChecked(),
            "show_endpoints": self._show_endpoints.isChecked(),
            "show_main": self._show_main.isChecked(),
            "show_secondary": self._show_secondary.isChecked(),
        }

    def _on_display_options_changed(self) -> None:
        self.canvas.set_display_options(self._get_display_options())

    def _browse_file(self) -> None:
        cur = self._path_edit.text().strip()
        start = str(Path(cur).parent) if cur and Path(cur).exists() else str(QSettings().value("last_dir/root_mask", str(Path.home())))
        if not Path(start).exists():
            start = str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "选择掩码图像", start,
            "图像 (*.png *.jpg *.jpeg *.tif *.tiff);;All (*)",
        )
        if path:
            self._leave_dataset_mode()
            self._path_edit.setText(path)
            self._image_dir = Path(path).parent
            QSettings().setValue("last_dir/root_mask", str(Path(path).parent))
            self._on_path_changed()

    def _on_workdir_file_selected(self, path: Path) -> None:
        self._leave_dataset_mode()
        self._path_edit.setText(str(path))
        self._image_dir = path.parent
        self._on_path_changed()

    def _leave_dataset_mode(self) -> None:
        self._current_sample = None
        self._dataset_image_path = None
        if self._dataset_panel is not None:
            self._dataset_panel.clear_selection()

    def _fill_class_combo(self, ds) -> None:
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

    def _on_class_combo_changed(self, _index: int) -> None:
        if self._current_sample is not None:
            did, sid = self._current_sample
            self._current_sample = None
            self._on_dataset_sample_selected(did, sid)

    def _on_dataset_sample_selected(self, dataset_id: str, sample_id: str) -> None:
        try:
            ds = self._platform_app.datasets.store.load(dataset_id)
            sample = ds.sample(sample_id)
            if sample is None:
                return
            self._fill_class_combo(ds)
            mask_path = self._dataset_mask_file(ds, sample, self._class_combo.currentData())
            image_path = ds.abs(sample.image)
        except Exception as e:
            QMessageBox.warning(self, "数据集", f"无法读取样本掩码:\n{e}")
            return
        self._current_sample = (dataset_id, sample_id)
        self._dataset_image_path = image_path if image_path.is_file() else None
        self._path_edit.setText(str(mask_path))
        self._image_dir = ds.abs(sample.image).parent
        self._on_path_changed()

    def _dataset_mask_file(self, ds, sample, class_id) -> Path:
        """返回供根系分析读取的二值掩码文件；单前景类别的语义掩码直接使用，否则按类别导出到缓存。"""
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

    def _get_workdir_mask_list(self) -> List[Path]:
        root = self._workdir_panel.get_root_path() if self._workdir_panel else None
        if root is None or not root.is_dir():
            if self._image_dir is not None and self._image_dir.is_dir():
                root = self._image_dir
            else:
                return []
        exts = {e.lower() for e in self.MASK_EXTENSIONS}
        return sorted(
            [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts],
            key=lambda x: (str(x).lower(), x.name),
        )

    def _get_sibling_mask_list(self) -> List[Path]:
        """仅当前掩码所在同级目录内的掩码列表，用于 ← → 切换，不跨级、不换路径。"""
        if self._current_path is None or not self._current_path.is_file():
            return []
        parent = self._current_path.parent
        if not parent.is_dir():
            return []
        exts = {e.lower() for e in self.MASK_EXTENSIONS}
        return sorted(
            [p for p in parent.iterdir() if p.is_file() and p.suffix.lower() in exts],
            key=lambda x: x.name.lower(),
        )

    def _go_prev_image(self) -> None:
        if self._current_sample is not None and self._dataset_panel is not None:
            self._dataset_panel.select_relative(-1)
            return
        if self._current_path is None:
            return
        images = self._get_sibling_mask_list()
        if not images:
            return
        cur = self._current_path.resolve()
        idx = next((i for i, p in enumerate(images) if p.resolve() == cur), -1)
        if idx <= 0:
            return
        self._path_edit.setText(str(images[idx - 1]))
        self._image_dir = images[idx - 1].parent
        self._on_path_changed()

    def _go_next_image(self) -> None:
        if self._current_sample is not None and self._dataset_panel is not None:
            self._dataset_panel.select_relative(1)
            return
        if self._current_path is None:
            return
        images = self._get_sibling_mask_list()
        if not images:
            return
        cur = self._current_path.resolve()
        idx = next((i for i, p in enumerate(images) if p.resolve() == cur), -1)
        if idx < 0 or idx >= len(images) - 1:
            return
        self._path_edit.setText(str(images[idx + 1]))
        self._image_dir = images[idx + 1].parent
        self._on_path_changed()

    def _on_path_changed(self) -> None:
        """切换图像时清空结果与选根，仅显示预览。"""
        if self._task_id is not None:
            self._platform_app.tasks.wait_task(self._task_id)
        raw = self._path_edit.text().strip()
        if not raw:
            self._current_path = None
            self.canvas.set_result(None)
            self.canvas.set_seed_markers([])
            self.canvas.set_preview(None)
            self._stats_panel.set_result(None)
            self._status_label.setText("请选择掩码图像并点击「预处理」")
            return
        p = Path(raw)
        if not p.is_file():
            self._current_path = None
            self.canvas.set_result(None)
            self.canvas.set_seed_markers([])
            self.canvas.set_preview(None)
            self._stats_panel.set_result(None)
            self._status_label.setText("请选择有效的掩码文件")
            return
        self._current_path = p
        self._current_result = None
        self._preprocessed_mask = None
        self._selected_seeds = []
        self._last_corrected_stroke = None
        self._result_before_correction = None
        self.canvas.set_result(None)
        self.canvas.set_seed_markers([])
        self._stats_panel.set_result(None)
        self._update_preview()
        self._status_label.setText("已选图像，请点击「预处理」")
        self._export_csv_btn.setEnabled(False)
        self._export_xlsx_btn.setEnabled(False)
        self._export_fig_btn.setEnabled(False)

    def _rgb_file(self, path: Path | None, color: bool) -> "np.ndarray | None":
        if path is None or not path.is_file():
            return None
        import cv2
        img = _imread_unicode(str(path), None if color else 0)
        if img is None:
            return None
        if color:
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        from plant_phenotype.datasets.mask_io import preview_mask_rgb
        return preview_mask_rgb(img)

    def _update_preview(self) -> None:
        if not self._current_path:
            self.canvas.set_display_base(None)
            self.canvas.set_preview(None)
            return
        if self._current_sample is not None and self._dataset_panel is not None:
            show_image = self._dataset_panel.display_mode() == "image"
            preview = self._rgb_file(self._dataset_image_path, True) if show_image else None
            if preview is None:
                preview = self._rgb_file(self._current_path, False)
            self.canvas.set_display_base(preview)
            self.canvas.set_preview(preview)
            return
        self.canvas.set_display_base(None)
        import cv2
        preview = None
        if self._image_dir:
            orig = _find_original_image(self._current_path, self._image_dir)
            if orig:
                img = _imread_unicode(orig)
                if img is not None:
                    if img.ndim == 2:
                        preview = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    else:
                        preview = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if preview is None:
            preview = self._rgb_file(self._current_path, False)
        self.canvas.set_preview(preview)

    def _on_preprocess(self) -> None:
        """预处理：run_one(seeds=[], batch_mode=False) → 得到 partial，等待用户点选根。"""
        if not self._current_path:
            self._status_label.setText("请先选择掩码图像")
            QMessageBox.warning(self, "预处理", "请先选择掩码图像。")
            return
        self._selected_seeds = []
        self.canvas.set_seed_markers([])
        self._preprocessed_mask = None
        self._last_corrected_stroke = None
        self._result_before_correction = None
        self._start_root_analysis([])

    def _start_root_analysis(self, seeds: list[tuple[int, int]]) -> None:
        if self._platform_app is None or self._current_path is None:
            return
        if self._task_id is not None:
            self._platform_app.tasks.cancel(self._task_id)
            self._platform_app.tasks.wait_task(self._task_id)
            self._task_id = None
        self._preprocess_btn.setEnabled(False)
        self._progress_bar.show()
        self._progress_bar.setRange(0, 0)
        self._task_logs_shown = 0
        self._task_id = self._platform_app.tasks.run_tool(
            "root.analyze",
            {
                "mask_path": str(self._current_path),
                "original_image_path": _find_original_image(self._current_path, self._image_dir),
                "selected_seeds": seeds,
                "batch_mode": False,
                "preprocessed_mask": self._preprocessed_mask,
                "config_overrides": self._get_params(),
            },
            title="根系分析" if seeds else "根系预处理",
        )

    def _on_task_state_changed(self, state) -> None:
        if state.tool_id == "segmentation.batch_predict" and state.status == "success":
            mask_dir = self._platform_app.context.get_last_mask_dir()
            if mask_dir:
                self._workdir_panel.set_root_path(Path(mask_dir))
        if state.id != self._task_id:
            return
        for line in state.logs[self._task_logs_shown:]:
            self._status_label.setText(line)
        self._task_logs_shown = len(state.logs)
        if state.status not in ("success", "failed", "cancelled"):
            return
        self._task_id = None
        if state.status != "success":
            result = next((item.data for item in (state.result.artifacts if state.result else []) if item.id == "pipeline_result"), None)
            if result is not None:
                self._on_worker_finished(result)
                return
            self._on_worker_error(state.error or state.message)
            return
        result = next(item.data for item in state.result.artifacts if item.id == "pipeline_result")
        self._on_worker_finished(result)

    def _on_worker_finished(self, res) -> None:
        self._progress_bar.hide()
        self._preprocess_btn.setEnabled(True)
        mask = getattr(res, "mask", None)
        mask_empty = mask is None or (hasattr(mask, "size") and mask.size == 0)
        if res.error or mask_empty:
            msg = (res.error if isinstance(res.error, str) else "点选区域无效或为空") or "点选区域无效或为空"
            self._status_label.setText("分析出错: " + msg[:80])
            QMessageBox.warning(self, "分析", "分析过程出错:\n" + msg[:500])
            return
        if not res.error and getattr(res, "G", None) is not None and self._last_corrected_stroke:
            corrected = get_facade().correct_main_root(
                res, self._last_corrected_stroke, config_overrides=self._get_params()
            )
            if corrected is not None:
                res = corrected
        self._current_result = res
        if self._platform_app is not None and self._current_path is not None:
            self._platform_app.context.set_last_mask_path(self._current_path)
            self._platform_app.context.set_pipeline_result(res)
        self.canvas.set_display_options(self._get_display_options())
        self.canvas.set_result(res)
        self.canvas.set_seed_markers(list(self._selected_seeds))
        self._stats_panel.set_result(res)
        err_msg = (res.error if isinstance(getattr(res, "error", None), str) else None) or ""
        if err_msg:
            self._status_label.setText("分析出错: " + err_msg[:80])
            QMessageBox.warning(self, "分析", "分析过程出错:\n" + err_msg[:500])
        elif getattr(res, "skeleton", None) is None:
            self._preprocessed_mask = res.mask
            self._status_label.setText("请点击根所在位置（可多选，可撤销）")
        else:
            n = len(self._selected_seeds)
            # 提取根系数量与主根长度，显示结果摘要（需求 3.3）
            root_count = n
            main_root_length = 0.0
            sdf = getattr(res, "summary_df", None)
            if sdf is not None and not getattr(sdf, "empty", True):
                if "main_root_length" in sdf.columns:
                    val = sdf["main_root_length"].iloc[0]
                    try:
                        main_root_length = float(val)
                    except (TypeError, ValueError):
                        pass
            self._status_label.setText(
                f"分析完成 | 检测到根系: {root_count} 条 | 主根长度: {main_root_length:.2f} px"
            )
            self._export_csv_btn.setEnabled(True)
            self._export_xlsx_btn.setEnabled(True)
            self._export_fig_btn.setEnabled(True)

    def _on_worker_error(self, msg: str) -> None:
        """后台线程发生未处理异常时的 UI 回调（需求 6.1, 6.2, 6.3）。"""
        self._progress_bar.hide()
        self._preprocess_btn.setEnabled(True)
        self._status_label.setText(f"分析出错: {msg[:80]}")
        QMessageBox.critical(self, "分析失败", msg)

    def _on_root_pixel_clicked(self, row: int, col: int) -> None:
        """用户点击图上根位置：加入种子并重新分析（选中的连通域并集）。"""
        if self._current_path is None or self._current_result is None:
            return
        if self._main_root_correction_mode:
            return
        self._selected_seeds.append((row, col))
        self.canvas.set_seed_markers(list(self._selected_seeds))
        self._start_root_analysis(list(self._selected_seeds))

    def _on_start_main_root_correction(self) -> None:
        if self._current_result is None or getattr(self._current_result, "G", None) is None:
            QMessageBox.information(self, "校正主根", "请先完成当前图像分析（点选根并得到结果）后再校正主根。")
            return
        self._main_root_correction_mode = True
        self.canvas.set_stroke_mode(True)
        self.canvas.setFocus(Qt.FocusReason.OtherFocusReason)
        self._status_label.setText("校正主根：沿主根拖拽描绘，松手即应用；Shift+松手可多段后 Ctrl+F 确认")

    def _on_shortcut_stroke_confirm(self) -> None:
        if self._main_root_correction_mode:
            self.canvas.try_confirm_stroke_editing()

    def _on_shortcut_stroke_undo(self) -> None:
        if self._main_root_correction_mode and self.canvas.undo_last_stroke_segment():
            self._status_label.setText("已撤销上一段描绘")
            return
        # 非描绘中：撤销上次主根校正
        if self._result_before_correction is not None:
            self._current_result = self._result_before_correction
            self._result_before_correction = None
            self._last_corrected_stroke = None
            self.canvas.set_result(self._current_result)
            self.canvas.set_seed_markers(list(self._selected_seeds))
            self._stats_panel.set_result(self._current_result)
            self._main_root_correction_mode = False
            self._status_label.setText("已撤销上次主根校正")
            return
        if self._main_root_correction_mode:
            self._status_label.setText("无可撤销的描绘段")

    def _on_stroke_segments_changed(self, n: int) -> None:
        if self._main_root_correction_mode:
            self._status_label.setText(
                "校正主根：已描绘 %d 段，可继续 Shift+拖拽；Ctrl+F 确认，Ctrl+C 撤销上一段" % n
            )

    def _on_main_root_stroke_confirm_invalid(self) -> None:
        QMessageBox.information(
            self,
            "校正主根",
            "点数不足或尚未描绘有效轨迹。\n请沿主根拖拽（采样点不少于 2 个）；多段时按住 Shift 松手累积，再 Ctrl+F 确认。",
        )

    def _on_main_root_stroke_confirmed(self, points: list) -> None:
        self._main_root_correction_mode = False
        if not points or self._current_result is None:
            self._status_label.setText("已选根时可继续点选或撤销")
            return
        self._result_before_correction = self._current_result
        new_result = get_facade().correct_main_root(
            self._current_result,
            points,
            config_overrides=self._get_params(),
        )
        if new_result is not None:
            self._current_result = new_result
            self._last_corrected_stroke = list(points)
            self.canvas.set_result(new_result)
            self.canvas.set_seed_markers(list(self._selected_seeds))
            self._stats_panel.set_result(new_result)
            self._status_label.setText("主根已按描绘轨迹更新（Ctrl+C 可撤销本次校正；后续点选其它根仍保留）")
        else:
            self._result_before_correction = None
            self._status_label.setText("校正失败，请沿主根骨架拖拽描绘轨迹后重试")
            QMessageBox.information(self, "校正主根", "校正失败，请沿主根骨架拖拽描绘轨迹后重试。")

    def _on_undo_root(self) -> None:
        """撤销上次点选：若已无选根则显示预处理图；否则用剩余种子重新分析。"""
        if not self._selected_seeds:
            self._status_label.setText("当前无已选根")
            return
        self._selected_seeds.pop()
        self.canvas.set_seed_markers(list(self._selected_seeds))
        if not self._selected_seeds:
            if self._preprocessed_mask is not None:
                partial = get_facade().create_preprocessed_partial(
                    self._preprocessed_mask,
                    original_image=getattr(self._current_result, "original_image", None),
                )
                self._current_result = partial
                self.canvas.set_result(partial)
                self._stats_panel.set_result(partial)
            self._export_csv_btn.setEnabled(False)
            self._export_xlsx_btn.setEnabled(False)
            self._export_fig_btn.setEnabled(False)
            self._status_label.setText("请点击根所在位置（可多选，可撤销）")
            return
        self._start_root_analysis(list(self._selected_seeds))

    def _export_csv(self) -> None:
        if self._current_result is None or getattr(self._current_result, "skeleton", None) is None:
            QMessageBox.information(self, "导出", "请先点选根并完成分析后再导出。")
            return
        start = str(QSettings().value("last_dir/root_export", str(Path.home())))
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", start, "CSV (*.csv)")
        if not path:
            return
        QSettings().setValue("last_dir/root_export", str(Path(path).parent))
        try:
            sdf = getattr(self._current_result, "summary_df", None)
            if sdf is not None and not sdf.empty:
                sdf.to_csv(path, index=False, encoding="utf-8-sig")
                QMessageBox.information(self, "导出", f"已保存: {path}")
            else:
                QMessageBox.warning(self, "导出", "无汇总数据可导出。")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _export_xlsx(self) -> None:
        if self._current_result is None or getattr(self._current_result, "skeleton", None) is None:
            QMessageBox.information(self, "导出", "请先点选根并完成分析后再导出。")
            return
        start = str(QSettings().value("last_dir/root_export", str(Path.home())))
        path, _ = QFileDialog.getSaveFileName(self, "导出 Excel", start, "Excel (*.xlsx)")
        if not path:
            return
        QSettings().setValue("last_dir/root_export", str(Path(path).parent))
        try:
            import pandas as pd
            sdf = getattr(self._current_result, "summary_df", None)
            ddf = getattr(self._current_result, "detail_df", None)
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                if sdf is not None and not sdf.empty:
                    sdf.to_excel(writer, sheet_name="汇总", index=False)
                if ddf is not None and not ddf.empty:
                    ddf.to_excel(writer, sheet_name="明细", index=False)
            QMessageBox.information(self, "导出", f"已保存: {path}")
        except ImportError:
            QMessageBox.critical(self, "导出失败", "请安装 pandas 与 openpyxl: pip install pandas openpyxl")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _export_visualization(self) -> None:
        if self._current_result is None or getattr(self._current_result, "skeleton", None) is None:
            QMessageBox.information(self, "导出", "请先点选根并完成分析后再导出。")
            return
        start = str(QSettings().value("last_dir/root_export", str(Path.home())))
        path, _ = QFileDialog.getSaveFileName(self, "导出图像", start, "PNG (*.png)")
        if not path:
            return
        QSettings().setValue("last_dir/root_export", str(Path(path).parent))
        try:
            opts = self._get_display_options()
            get_facade().export_figure(self._current_result, path, display_options=opts)
            QMessageBox.information(self, "导出", f"已保存: {path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _restore_defaults(self) -> None:
        self._morph_kernel.setValue(_DEFAULTS["morph_kernel_size"])
        self._min_component_area.setValue(_DEFAULTS["min_component_area"])
        self._morph_open.setChecked(_DEFAULTS["morph_open"])
        self._connect_enabled.setChecked(_DEFAULTS["connect_enabled"])
        self._connect_max_search.setValue(_DEFAULTS["connect_max_search"])
        self._connect_angle_tol.setValue(_DEFAULTS["connect_angle_tol_deg"])
        self._connect_min_gap.setValue(_DEFAULTS["connect_min_gap"])
        self._connect_max_gap.setValue(_DEFAULTS["connect_max_gap"])
        self._junction_min.setValue(_DEFAULTS["junction_min_neighbors"])
        self._spur_max.setValue(int(_DEFAULTS.get("spur_max_length", 8)))
        self._w_length.setValue(_DEFAULTS["weight_length"])
        self._w_diameter.setValue(_DEFAULTS["weight_diameter"])
        self._w_curvature.setValue(_DEFAULTS["weight_curvature"])
        self._min_root_length.setValue(_DEFAULTS["min_root_length"])
        self._min_root_diameter.setValue(_DEFAULTS["min_root_diameter"])
        self._zscore.setChecked(_DEFAULTS["zscore"])
        self._primary_enabled.setChecked(_DEFAULTS["primary_enabled"])
        self._primary_alpha.setValue(_DEFAULTS["primary_alpha"])
        self._primary_beta.setValue(_DEFAULTS["primary_beta"])
        self._primary_gamma.setValue(_DEFAULTS["primary_gamma"])
        self._primary_tau.setValue(_DEFAULTS["primary_tau"])
        self._primary_max_level.setValue(_DEFAULTS["primary_max_level"])
        self._primary_diam_ratio.setValue(_DEFAULTS["primary_diameter_ratio_min"])
        self._primary_diam_floor.setValue(_DEFAULTS["primary_diameter_floor"])
        QMessageBox.information(self, "恢复默认", "已恢复 config/root_phenotype 默认值。")

    def wait_workers(self) -> None:
        """关闭前等待后台分析线程结束，避免 QThread destroyed while still running。"""
        if self._task_id is not None:
            self._platform_app.tasks.wait_task(self._task_id)


def build_root_phenotype_tab(app=None, parent=None) -> QWidget:
    return RootPhenotypeTab(parent, platform_app=app)
