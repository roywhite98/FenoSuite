# -*- coding: utf-8 -*-
"""数据标注 Tab：与 dataset-tool 一致的画布标注、保存、切块与还原。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from PyQt6.QtCore import Qt, QSettings, QThread, pyqtSignal, QRectF
from PyQt6.QtGui import (
    QImage,
    QIcon,
    QPixmap,
    QPainter,
    QColor,
    QKeySequence,
    QShortcut,
    QPen,
    QBrush,
)
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QLabel,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QFrame,
    QSplitter,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QDoubleSpinBox,
    QButtonGroup,
    QSlider,
    QGraphicsEllipseItem,
    QApplication,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTabWidget,
)

from plant_phenotype.annotation.facade import BrushShape, get_facade
from plant_phenotype.datasets.store import DatasetError, DatasetProtectedError
from plant_phenotype.gui.canvas import CanvasView
from plant_phenotype.gui.dataset_panel import DatasetSamplePanel
from plant_phenotype.gui.utils import equalize_widths
from plant_phenotype.gui.workdir_tree import WorkdirTreePanel

_facade = get_facade()
IMAGE_EXTENSIONS = _facade.image_extensions()
MASK_SUFFIX = _facade.mask_suffix()
_TILE_DEFAULTS = _facade.default_tile_params()
DEFAULT_TILE_SIZE = _TILE_DEFAULTS["tile_size"]
DEFAULT_MIN_FOREGROUND_RATIO = _TILE_DEFAULTS["min_foreground_ratio"]


def _brush_icon(shape: BrushShape, size: int = 24) -> QIcon:
    """生成圆形或方形笔刷的工具栏图标。"""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QColor(60, 60, 60))
    painter.setBrush(QColor(0, 160, 0))
    margin = 2
    r = (size - 2 * margin) // 2
    cx = cy = size // 2
    if shape == BrushShape.CIRCLE:
        painter.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)
    else:
        painter.drawRect(cx - r, cy - r, 2 * r, 2 * r)
    painter.end()
    return QIcon(pix)


def _load_qimage(path: Path) -> tuple[QImage, np.ndarray]:
    """加载图像（支持 TIFF/JPG/GIF 等），返回 QImage 与 numpy 数组（用于油漆桶/魔法棒）。"""
    from PIL import Image
    pil = Image.open(path)
    # 统一转为 8 位 RGB，兼容 16/32 位 TIFF、调色板图、灰度、RGBA 等
    if pil.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
        # 整型或浮点：线性缩放到 0–255
        arr = np.array(pil)
        if arr.size == 0:
            arr = np.zeros((pil.height, pil.width), dtype=np.uint8)
        else:
            mx = float(np.nanmax(arr))
            mn = float(np.nanmin(arr))
            if mx > mn:
                arr = (arr.astype(np.float64) - mn) / (mx - mn) * 255
            else:
                arr = np.zeros_like(arr, dtype=np.float64)
            arr = np.nan_to_num(arr, nan=0.0, posinf=255, neginf=0).astype(np.uint8)
        pil = Image.fromarray(arr).convert("RGB")
    elif pil.mode not in ("RGB", "L"):
        # P(调色板)、RGBA、PA、LA 等
        if pil.mode == "RGBA":
            bg = Image.new("RGB", pil.size, (255, 255, 255))
            bg.paste(pil, mask=pil.split()[-1])
            pil = bg
        else:
            pil = pil.convert("RGB")
    else:
        if pil.mode == "L":
            pil = pil.convert("RGB")
    arr = np.array(pil)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.ascontiguousarray(arr[:, :, :3].astype(np.uint8))
    buf = arr.tobytes()
    qimg = QImage(buf, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return qimg, arr


def _mask_bool_to_overlay_qimage(mask: np.ndarray, rgba: tuple[int, int, int, int] = (255, 200, 0, 120)) -> QImage:
    """二值 mask → RGBA QImage，用于画布半透明预览。"""
    h, w = mask.shape[:2]
    rgba_arr = np.zeros((h, w, 4), dtype=np.uint8)
    m = mask.astype(bool)
    rgba_arr[m, 0] = rgba[0]
    rgba_arr[m, 1] = rgba[1]
    rgba_arr[m, 2] = rgba[2]
    rgba_arr[m, 3] = rgba[3]
    if not rgba_arr.flags["C_CONTIGUOUS"]:
        rgba_arr = np.ascontiguousarray(rgba_arr)
    return QImage(rgba_arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()


class Sam2Worker(QThread):
    finished_ok = pyqtSignal(object, object, str, int)
    finished_err = pyqtSignal(str, int)

    def __init__(
        self,
        model: Any,
        image_rgb: np.ndarray,
        marks: List[dict],
        cached_emb: Optional[dict],
        cache_key: str,
        generation: int,
    ) -> None:
        super().__init__()
        self.setObjectName("Sam2Worker")
        self._model = model
        self._image_rgb = np.ascontiguousarray(image_rgb)
        self._marks = [dict(m) for m in marks]
        self._cached_emb = cached_emb
        self._cache_key = cache_key
        self._generation = generation

    def run(self) -> None:
        try:
            mask_f, emb_out = get_facade().predict_sam2_mask(
                self._model,
                self._image_rgb,
                self._marks,
                embedding=self._cached_emb,
            )
            mask_b = mask_f > 0.0
            self.finished_ok.emit(mask_b, emb_out, self._cache_key, self._generation)
        except Exception as e:
            self.finished_err.emit(str(e), self._generation)


class AnnotationTab(QWidget):
    """数据标注页：左侧工具 + 上方主操作栏/工具选项 + 画布 + 右侧工作目录树（与 dataset-tool/rootmask 一致）。"""

    def __init__(self, parent=None, platform_app=None):
        super().__init__(parent)
        self._platform_app = platform_app
        self._task_id: str | None = None
        self._task_kind: str | None = None
        self._current_image_path: Path | None = None
        self._current_sample: tuple[str, str] | None = None  # (dataset_id, sample_id)
        self._sample_base_qimg: QImage | None = None
        self._output_dir: Path | None = None
        self._tiles_base_dir: Path | None = None
        self._undo_stack = _facade.new_undo_stack()
        self._annotation_modified = False  # 仅在实际编辑过标注时为 True，加载/保存后为 False
        self._auto_save_on_switch = False  # 勾选“下次默认保存”后为 True，切换时自动保存不提示；重启后重置
        self._last_save_dir: Path | None = None  # 上次保存所在目录，勾选默认保存后自动保存时使用，不再弹窗选路径
        self._sam2_model: Any = None
        self._sam2_model_paths: tuple[Path, Path] | None = None
        self._sam2_marks: List[dict] = []
        self._sam2_marker_items: List[QGraphicsEllipseItem] = []
        self._sam2_preview_mask: Optional[np.ndarray] = None
        self._sam2_worker: Optional[Sam2Worker] = None
        self._sam2_gen: int = 0
        self._sam2_cache = _facade.new_sam2_embedding_cache(max_items=2)
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ---------- 左侧：仅标注工具（Labelme 式窄栏）----------
        tools_side = QWidget()
        tools_side.setObjectName("toolsSideBar")
        tools_side_layout = QVBoxLayout(tools_side)
        tools_side_layout.setContentsMargins(8, 8, 8, 8)
        tools_side_layout.setSpacing(8)

        self._brush_btn = QPushButton("笔刷")
        self._brush_btn.setCheckable(True)
        self._brush_btn.setChecked(True)
        self._brush_btn.setToolTip("笔刷绘制前景")
        tools_side_layout.addWidget(self._brush_btn)

        self._erase_btn = QPushButton("橡皮")
        self._erase_btn.setCheckable(True)
        self._erase_btn.setToolTip("橡皮擦除")
        tools_side_layout.addWidget(self._erase_btn)

        self._fill_btn = QPushButton("油漆桶")
        self._fill_btn.setCheckable(True)
        self._fill_btn.setToolTip("填充连通区域")
        tools_side_layout.addWidget(self._fill_btn)

        self._otsu_fill_btn = QPushButton("大津油漆桶")
        self._otsu_fill_btn.setCheckable(True)
        self._otsu_fill_btn.setToolTip("基于阈值的填充，阈值可用滑块调节；可点「取大津」用当前图自动计算阈值")
        tools_side_layout.addWidget(self._otsu_fill_btn)

        self._region_grow_btn = QPushButton("区域增长")
        self._region_grow_btn.setCheckable(True)
        self._region_grow_btn.setToolTip("区域增长法填充：从种子点扩展，灰度与种子点差值在容差内的连通区域")
        tools_side_layout.addWidget(self._region_grow_btn)

        self._sam2_btn = QPushButton("SAM2\n点提示")
        self._sam2_btn.setCheckable(True)
        self._sam2_btn.setToolTip(
            "权重按 resources/configs/sam2_weights.yaml 自动下载或使用缓存。左键落点（半透明圆为预览），"
            "右键删附近误点；调「预览透明度」看 mask；满意后点「确定分割并写入标注」。"
            "其它工具下右键仍切换标注层显示。"
        )
        tools_side_layout.addWidget(self._sam2_btn)

        tools_side_layout.addStretch(1)

        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_group.addButton(self._brush_btn)
        self._tool_group.addButton(self._erase_btn)
        self._tool_group.addButton(self._fill_btn)
        self._tool_group.addButton(self._otsu_fill_btn)
        self._tool_group.addButton(self._region_grow_btn)
        self._tool_group.addButton(self._sam2_btn)
        self._tool_group.buttonClicked.connect(self._on_tool_changed)

        # ---------- 工具选项栏（随左侧工具切换）----------
        self._tool_options_stack = QStackedWidget()
        self._tool_options_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        # 页 0：笔刷 / 橡皮（共用形状与大小）
        page_brush = QWidget()
        brush_row = QHBoxLayout(page_brush)
        brush_row.setContentsMargins(4, 0, 4, 0)
        brush_row.setSpacing(6)
        brush_row.addWidget(QLabel("笔刷形状:"))
        self._brush_shape_group = QButtonGroup(self)
        self._brush_circle_btn = QPushButton()
        self._brush_circle_btn.setToolTip("圆形笔刷")
        self._brush_circle_btn.setIcon(_brush_icon(BrushShape.CIRCLE, size=20))
        self._brush_circle_btn.setCheckable(True)
        self._brush_circle_btn.setChecked(True)
        self._brush_circle_btn.setProperty("mdShape", "square")
        self._brush_circle_btn.clicked.connect(lambda: self._on_brush_shape_changed(BrushShape.CIRCLE))
        self._brush_shape_group.addButton(self._brush_circle_btn)
        brush_row.addWidget(self._brush_circle_btn)
        self._brush_square_btn = QPushButton()
        self._brush_square_btn.setToolTip("方形笔刷")
        self._brush_square_btn.setIcon(_brush_icon(BrushShape.SQUARE, size=20))
        self._brush_square_btn.setCheckable(True)
        self._brush_square_btn.setProperty("mdShape", "square")
        self._brush_square_btn.clicked.connect(lambda: self._on_brush_shape_changed(BrushShape.SQUARE))
        self._brush_shape_group.addButton(self._brush_square_btn)
        brush_row.addWidget(self._brush_square_btn)
        brush_row.addWidget(QLabel("笔刷大小:"))
        self._brush_spin = QSpinBox()
        self._brush_spin.setRange(1, 200)
        self._brush_spin.setValue(20)
        self._brush_spin.valueChanged.connect(self._on_brush_size_changed)
        self._brush_spin.setMaximumHeight(28)
        brush_row.addWidget(self._brush_spin)
        brush_row.addStretch(1)
        self._tool_options_stack.addWidget(page_brush)

        # 页 1：油漆桶
        page_fill = QWidget()
        fill_row = QHBoxLayout(page_fill)
        fill_row.setContentsMargins(4, 0, 4, 0)
        fill_row.addWidget(
            QLabel("点击图像中要标注为前景的连通区域进行填充。")
        )
        fill_row.addStretch(1)
        self._tool_options_stack.addWidget(page_fill)

        # 页 2：大津油漆桶
        page_otsu = QWidget()
        otsu_row = QHBoxLayout(page_otsu)
        otsu_row.setContentsMargins(4, 0, 4, 0)
        otsu_row.setSpacing(6)
        otsu_row.addWidget(QLabel("阈值:"))
        self._otsu_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._otsu_threshold_slider.setRange(0, 255)
        self._otsu_threshold_slider.setValue(128)
        self._otsu_threshold_slider.setMinimumWidth(120)
        self._otsu_threshold_slider.setMaximumWidth(220)
        self._otsu_threshold_slider.valueChanged.connect(self._on_otsu_threshold_changed)
        self._otsu_threshold_slider.setMaximumHeight(24)
        otsu_row.addWidget(self._otsu_threshold_slider)
        self._otsu_threshold_label = QLabel("128")
        self._otsu_threshold_label.setMinimumWidth(28)
        otsu_row.addWidget(self._otsu_threshold_label)
        self._otsu_auto_btn = QPushButton("取大津")
        self._otsu_auto_btn.setToolTip("根据当前图像自动计算大津阈值并设为滑块值")
        self._otsu_auto_btn.clicked.connect(self._on_otsu_auto_clicked)
        otsu_row.addWidget(self._otsu_auto_btn)
        otsu_row.addStretch(1)
        self._tool_options_stack.addWidget(page_otsu)

        # 页 3：区域增长
        page_region = QWidget()
        region_row = QHBoxLayout(page_region)
        region_row.setContentsMargins(4, 0, 4, 0)
        region_row.setSpacing(6)
        region_row.addWidget(QLabel("容差:"))
        self._region_grow_slider = QSlider(Qt.Orientation.Horizontal)
        self._region_grow_slider.setRange(0, 255)
        self._region_grow_slider.setValue(32)
        self._region_grow_slider.setMinimumWidth(100)
        self._region_grow_slider.setMaximumWidth(200)
        self._region_grow_slider.valueChanged.connect(self._on_region_grow_tolerance_changed)
        self._region_grow_slider.setMaximumHeight(24)
        region_row.addWidget(self._region_grow_slider)
        self._region_grow_label = QLabel("32")
        self._region_grow_label.setMinimumWidth(28)
        region_row.addWidget(self._region_grow_label)
        region_row.addStretch(1)
        self._tool_options_stack.addWidget(page_region)

        # 页 4：SAM2（无 Encoder/Decoder 路径 UI，权重仅由配置/缓存与 QSettings 解析）
        page_sam2 = QWidget()
        sam2_outer = QVBoxLayout(page_sam2)
        sam2_outer.setContentsMargins(0, 0, 0, 0)
        sam2_scroll = QScrollArea()
        sam2_scroll.setWidgetResizable(True)
        sam2_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sam2_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sam2_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sam2_scroll.setMaximumHeight(72)
        sam2_inner = QWidget()
        # 分两行：上行为提示点参数，下行为清除/写入，窄屏下主操作仍可见
        sam2_v = QVBoxLayout(sam2_inner)
        sam2_v.setContentsMargins(4, 0, 4, 0)
        sam2_v.setSpacing(2)
        sam2_h = QHBoxLayout()
        sam2_h.setSpacing(6)
        sam2_action_row = QHBoxLayout()
        sam2_action_row.setSpacing(6)
        sam2_v.addLayout(sam2_h)
        sam2_v.addLayout(sam2_action_row)

        self._sam2_pos_btn = QPushButton("正点")
        self._sam2_pos_btn.setCheckable(True)
        self._sam2_pos_btn.setChecked(True)
        self._sam2_pos_btn.setToolTip("左键在图像上添加正样本点；SAM2 模式下右键可删除附近的点")
        self._sam2_neg_btn = QPushButton("负点")
        self._sam2_neg_btn.setCheckable(True)
        self._sam2_neg_btn.setToolTip("左键在图像上添加负样本点；SAM2 模式下右键可删除附近的点")
        self._sam2_polarity_group = QButtonGroup(self)
        self._sam2_polarity_group.setExclusive(True)
        self._sam2_polarity_group.addButton(self._sam2_pos_btn)
        self._sam2_polarity_group.addButton(self._sam2_neg_btn)
        self._sam2_pos_btn.clicked.connect(self._on_sam2_polarity_changed)
        self._sam2_neg_btn.clicked.connect(self._on_sam2_polarity_changed)
        sam2_h.addWidget(self._sam2_pos_btn)
        sam2_h.addWidget(self._sam2_neg_btn)
        equalize_widths([self._sam2_pos_btn, self._sam2_neg_btn])

        sam2_h.addWidget(QLabel("点半径:"))
        self._sam2_point_radius_spin = QSpinBox()
        self._sam2_point_radius_spin.setRange(2, 40)
        pr_saved = QSettings().value("annotation_sam2_point_radius", 8)
        try:
            self._sam2_point_radius_spin.setValue(int(pr_saved))
        except (TypeError, ValueError):
            self._sam2_point_radius_spin.setValue(8)
        self._sam2_point_radius_spin.setToolTip("提示点在图上的显示大小；SAM2 模式下可 Ctrl+滚轮微调")
        self._sam2_point_radius_spin.valueChanged.connect(self._on_sam2_point_radius_changed)
        self._sam2_point_radius_spin.setMaximumHeight(28)
        sam2_h.addWidget(self._sam2_point_radius_spin)

        ov_saved = QSettings().value("annotation_sam2_overlay_opacity", 45)
        try:
            ov_i = int(ov_saved)
        except (TypeError, ValueError):
            ov_i = 45
        ov_i = max(5, min(100, ov_i))
        self._sam2_overlay_opacity = ov_i / 100.0
        sam2_h.addWidget(QLabel("预览透明度:"))
        self._sam2_overlay_slider = QSlider(Qt.Orientation.Horizontal)
        self._sam2_overlay_slider.setRange(5, 100)
        self._sam2_overlay_slider.setValue(ov_i)
        self._sam2_overlay_slider.setMinimumWidth(60)
        self._sam2_overlay_slider.setMaximumWidth(120)
        self._sam2_overlay_slider.setToolTip("SAM2 分割预览 mask 叠加透明度（不影响写入标注后的结果）")
        self._sam2_overlay_slider.valueChanged.connect(self._on_sam2_overlay_opacity_changed)
        self._sam2_overlay_slider.setMaximumHeight(24)
        sam2_h.addWidget(self._sam2_overlay_slider)
        self._sam2_overlay_opacity_label = QLabel(str(ov_i))
        self._sam2_overlay_opacity_label.setMinimumWidth(28)
        sam2_h.addWidget(self._sam2_overlay_opacity_label)

        sam2_h.addStretch(1)

        self._sam2_clear_btn = QPushButton("清除 SAM2 提示")
        self._sam2_clear_btn.setToolTip("清除正负点与预览 overlay，保留当前图的 embedding 缓存")
        self._sam2_clear_btn.clicked.connect(self._sam2_clear_prompts)
        sam2_action_row.addWidget(self._sam2_clear_btn)
        self._sam2_apply_btn = QPushButton("确定分割并写入标注")
        self._sam2_apply_btn.setToolTip(
            "确认将当前 SAM2 预览 mask 合并到标注层；写入后可 Ctrl+Z 撤销，再保存 Mask 落盘。"
        )
        self._sam2_apply_btn.setProperty("mdRole", "filled")
        self._sam2_apply_btn.clicked.connect(self._sam2_apply_preview)
        sam2_action_row.addWidget(self._sam2_apply_btn)
        sam2_action_row.addStretch(1)

        sam2_scroll.setWidget(sam2_inner)
        sam2_outer.addWidget(sam2_scroll)
        self._tool_options_stack.addWidget(page_sam2)

        # ---------- 顶栏：上行文件与切块，下行工具选项与透明度 ----------
        top_bar = QWidget()
        top_bar.setObjectName("annotationTopBar")
        self._top_bar = top_bar
        top_outer = QVBoxLayout(top_bar)
        top_outer.setContentsMargins(6, 2, 6, 2)
        top_outer.setSpacing(2)

        file_row = QHBoxLayout()
        file_row.setSpacing(4)
        file_row.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        open_btn = QPushButton("打开图像")
        open_btn.clicked.connect(self._open_image)
        file_row.addWidget(open_btn)

        save_btn = QPushButton("保存 Mask")
        save_btn.clicked.connect(self._save_mask)
        file_row.addWidget(save_btn)

        out_btn = QPushButton("选择标注目录")
        out_btn.clicked.connect(self._select_output_dir)
        file_row.addWidget(out_btn)

        tile_btn = QPushButton("对目录执行切块")
        tile_btn.setToolTip("切割已标注的原图或直接切割原图：选择源目录（可有或没有 *_mask.png）、切块输出目录与参数")
        tile_btn.clicked.connect(self._run_preprocessing)
        file_row.addWidget(tile_btn)
        reassemble_btn = QPushButton("还原切块为整图")
        reassemble_btn.setToolTip("从含 meta.json 的切块目录拼接还原整图与 mask")
        reassemble_btn.clicked.connect(self._run_reassemble)
        file_row.addWidget(reassemble_btn)
        equalize_widths([open_btn, save_btn, out_btn, tile_btn, reassemble_btn])
        file_row.addStretch(1)
        self._tile_btn = tile_btn
        self._reassemble_btn = reassemble_btn
        top_outer.addLayout(file_row)

        tool_row = QHBoxLayout()
        tool_row.setSpacing(4)
        tool_row.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        tool_row.addWidget(self._tool_options_stack, 1)

        sep_view = QFrame()
        sep_view.setFrameShape(QFrame.Shape.VLine)
        sep_view.setFrameShadow(QFrame.Shadow.Sunken)
        sep_view.setFixedWidth(1)
        sep_view.setMinimumHeight(18)
        sep_view.setMaximumHeight(48)
        tool_row.addWidget(sep_view)

        tool_row.addWidget(QLabel("标注透明度:"))
        self._annotation_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._annotation_opacity_slider.setRange(10, 100)
        self._annotation_opacity_slider.setValue(100)
        self._annotation_opacity_slider.setMaximumWidth(100)
        self._annotation_opacity_slider.setMaximumHeight(24)
        self._annotation_opacity_slider.valueChanged.connect(self._on_annotation_opacity_changed)
        tool_row.addWidget(self._annotation_opacity_slider)
        self._show_mask_check = QCheckBox("显示标注层")
        self._show_mask_check.setChecked(True)
        self._show_mask_check.toggled.connect(self._on_show_mask_toggled)
        tool_row.addWidget(self._show_mask_check)
        top_outer.addLayout(tool_row)

        # 垂直方向固定为内容高度，随当前工具选项行变化，不参与画布的拉伸分配
        top_bar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        # ---------- 中间：顶栏 + 画布 ----------
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self._canvas = CanvasView(self)
        self._canvas.brushSizeChanged.connect(self._on_canvas_brush_size_changed)
        self._canvas.annotationVisibilityChanged.connect(self._on_annotation_visibility_changed)
        self._canvas.prevImageRequested.connect(self._go_prev_image)
        self._canvas.nextImageRequested.connect(self._go_next_image)
        self._canvas.sam2PromptClicked.connect(self._on_sam2_prompt_clicked)
        self._canvas.sam2RemoveNearestRequested.connect(self._sam2_remove_nearest_point)
        self._canvas.sam2PointRadiusChanged.connect(self._on_canvas_sam2_point_radius_changed)
        self._canvas.set_sam2_point_radius(self._sam2_point_radius_spin.value())

        def on_before_edit():
            layer = self._canvas.get_annotation_layer()
            if layer:
                self._undo_stack.push(layer.copy())
            self._annotation_modified = True

        self._canvas.set_on_before_edit(on_before_edit)
        self._canvas.set_on_stroke_done(lambda: None)

        QShortcut(QKeySequence.StandardKey.Undo, self, self._undo)
        QShortcut(QKeySequence.StandardKey.Save, self, self._save_mask)
        QShortcut(QKeySequence(Qt.Key.Key_BracketLeft), self,
                  lambda: self._brush_spin.setValue(max(1, self._brush_spin.value() - 5)))
        QShortcut(QKeySequence(Qt.Key.Key_BracketRight), self,
                  lambda: self._brush_spin.setValue(min(200, self._brush_spin.value() + 5)))

        content_layout.addWidget(top_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self._canvas)
        self._status_label = QLabel("请在右侧选择数据集样本（或切到「文件夹」打开图像）。← → 键切换图片；Ctrl+滚轮调整笔刷。")
        self._status_label.setProperty("mdRole", "status")
        left_layout.addWidget(self._status_label)
        splitter.addWidget(left)

        self._workdir_panel = WorkdirTreePanel(
            self,
            title="",
            file_extensions=IMAGE_EXTENSIONS,
            settings_key="annotation_last_work_directory",
        )
        self._workdir_panel.fileSelected.connect(self._load_image_at_path)
        self._side_tabs = QTabWidget()
        self._side_tabs.setMaximumWidth(300)
        self._dataset_panel: DatasetSamplePanel | None = None
        if platform_app is not None:
            self._dataset_panel = DatasetSamplePanel(platform_app.datasets, self)
            self._dataset_panel.sampleSelected.connect(self._on_dataset_sample_selected)
            self._dataset_panel.displayModeChanged.connect(self._apply_dataset_display)
            self._side_tabs.addTab(self._dataset_panel, "数据集")
        self._side_tabs.addTab(self._workdir_panel, "文件夹")
        splitter.addWidget(self._side_tabs)
        splitter.setSizes([760, 260])
        content_layout.addWidget(splitter)

        tools_scroll = QScrollArea()
        tools_scroll.setWidgetResizable(True)
        tools_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tools_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        tools_scroll.setFrameShape(QFrame.Shape.NoFrame)
        tools_scroll.setWidget(tools_side)
        tools_scroll.setMinimumWidth(84)
        tools_scroll.setMaximumWidth(112)

        outer_splitter = QSplitter(Qt.Orientation.Horizontal)
        outer_splitter.setChildrenCollapsible(False)
        outer_splitter.setHandleWidth(6)
        outer_splitter.addWidget(tools_scroll)
        outer_splitter.addWidget(content_widget)
        outer_splitter.setStretchFactor(0, 0)
        outer_splitter.setStretchFactor(1, 1)
        outer_splitter.setSizes([88, 1200])

        main_layout.addWidget(outer_splitter, 1)

        self._sync_tool_options_page()
        if platform_app is not None:
            platform_app.tasks.task_state_changed.connect(self._on_task_state_changed)

    def _sam2_reset_on_new_image(self) -> None:
        self._sam2_gen += 1
        self._sam2_marks.clear()
        self._sam2_remove_all_markers()
        self._sam2_preview_mask = None
        self._canvas.set_overlay(None)
        self._sam2_cache.clear()

    def _sam2_remove_all_markers(self) -> None:
        sc = self._canvas.scene()
        for it in self._sam2_marker_items:
            sc.removeItem(it)
        self._sam2_marker_items.clear()

    def _sam2_invalidate_model(self) -> None:
        self._sam2_model = None
        self._sam2_model_paths = None

    def _on_sam2_polarity_changed(self) -> None:
        self._canvas.set_sam2_positive_point(self._sam2_pos_btn.isChecked())

    def _on_sam2_point_radius_changed(self, value: int) -> None:
        self._canvas.set_sam2_point_radius(value)
        QSettings().setValue("annotation_sam2_point_radius", value)
        self._sam2_sync_marker_sizes()

    def _on_canvas_sam2_point_radius_changed(self, value: int) -> None:
        self._sam2_point_radius_spin.blockSignals(True)
        if self._sam2_point_radius_spin.value() != value:
            self._sam2_point_radius_spin.setValue(value)
        self._sam2_point_radius_spin.blockSignals(False)
        QSettings().setValue("annotation_sam2_point_radius", value)
        self._sam2_sync_marker_sizes()

    def _on_sam2_overlay_opacity_changed(self, value: int) -> None:
        v = max(5, min(100, int(value)))
        self._sam2_overlay_opacity = v / 100.0
        self._sam2_overlay_opacity_label.setText(str(v))
        QSettings().setValue("annotation_sam2_overlay_opacity", v)
        self._canvas.set_overlay_opacity(self._sam2_overlay_opacity)

    def _sam2_sync_marker_sizes(self) -> None:
        r = self._sam2_point_radius_spin.value()
        if len(self._sam2_marker_items) != len(self._sam2_marks):
            return
        for item, m in zip(self._sam2_marker_items, self._sam2_marks):
            if m.get("type") != "point":
                continue
            x, y = m["data"]
            item.setRect(QRectF(x - r, y - r, 2 * r, 2 * r))

    def _ensure_sam2_model(self) -> bool:
        st = QSettings()
        enc_s = str(st.value("annotation_sam2_encoder_path", "") or "").strip()
        dec_s = str(st.value("annotation_sam2_decoder_path", "") or "").strip()
        enc = Path(enc_s) if enc_s else None
        dec = Path(dec_s) if dec_s else None
        enc_ok = enc is not None and enc.is_file()
        dec_ok = dec is not None and dec.is_file()

        if enc_ok and dec_ok:
            resolved_enc, resolved_dec = enc.resolve(), dec.resolve()
        else:
            try:
                def _pulse() -> None:
                    QApplication.processEvents()

                cache_enc, cache_dec = get_facade().ensure_sam2_weights(
                    status=self._status_label.setText,
                    pulse=_pulse,
                )
                resolved_enc = enc.resolve() if enc_ok else cache_enc.resolve()
                resolved_dec = dec.resolve() if dec_ok else cache_dec.resolve()
                if not enc_ok:
                    st.setValue("annotation_sam2_encoder_path", str(cache_enc))
                if not dec_ok:
                    st.setValue("annotation_sam2_decoder_path", str(cache_dec))
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "SAM2",
                    "无法使用本地 ONNX，且根据配置自动下载失败：\n"
                    f"{e}\n\n"
                    "请检查网络，或编辑 resources/configs/sam2_weights.yaml / 缓存目录中的权重。",
                )
                return False

        pair = (resolved_enc, resolved_dec)
        if self._sam2_model is not None and self._sam2_model_paths == pair:
            return True
        try:
            self._sam2_model = get_facade().create_sam2_session(resolved_enc, resolved_dec)
            self._sam2_model_paths = pair
            return True
        except Exception as e:
            QMessageBox.warning(self, "SAM2", f"加载 ONNX 失败:\n{e}")
            self._sam2_model = None
            self._sam2_model_paths = None
            return False

    def _sam2_remove_nearest_point(self, x: int, y: int) -> None:
        if not self._sam2_btn.isChecked() or not self._sam2_marks:
            return
        layer = self._canvas.get_annotation_layer()
        if layer is None:
            return
        r = self._canvas.get_sam2_point_radius()
        max_dist = max(12.0, float(r) * 2.8)
        w, h = layer.width, layer.height
        xi = max(0, min(w - 1, x))
        yi = max(0, min(h - 1, y))
        best_i = -1
        best_d2 = max_dist * max_dist
        for i, m in enumerate(self._sam2_marks):
            if m.get("type") != "point":
                continue
            px, py = m["data"]
            d2 = (px - xi) ** 2 + (py - yi) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best_i = i
        if best_i < 0:
            self._status_label.setText("SAM2：附近无提示点，请右键更靠近绿/红圆点")
            return
        self._sam2_marks.pop(best_i)
        it = self._sam2_marker_items.pop(best_i)
        self._canvas.scene().removeItem(it)
        self._status_label.setText("SAM2：已删除一个提示点")
        self._sam2_schedule_infer_or_clear_preview()

    def _sam2_schedule_infer_or_clear_preview(self) -> None:
        if not get_facade().marks_need_positive(self._sam2_marks):
            self._sam2_preview_mask = None
            self._canvas.set_overlay(None)
            self._status_label.setText("SAM2：请至少添加一个正点后再预测")
            return
        self._sam2_schedule_infer()

    def _on_sam2_prompt_clicked(self, x: int, y: int, positive: bool) -> None:
        if not self._sam2_btn.isChecked():
            return
        layer = self._canvas.get_annotation_layer()
        if layer is None:
            return
        w, h = layer.width, layer.height
        xi = max(0, min(w - 1, x))
        yi = max(0, min(h - 1, y))
        self._sam2_marks.append({"type": "point", "data": [xi, yi], "label": 1 if positive else 0})
        self._sam2_add_marker_item(xi, yi, positive)
        self._sam2_schedule_infer()

    def _sam2_add_marker_item(self, x: int, y: int, positive: bool) -> None:
        r = self._sam2_point_radius_spin.value()
        item = QGraphicsEllipseItem(QRectF(x - r, y - r, 2 * r, 2 * r))
        item.setZValue(80)
        col = QColor(0, 200, 0) if positive else QColor(220, 40, 40)
        item.setPen(QPen(col, 2))
        item.setBrush(QBrush(col))
        item.setVisible(self._sam2_btn.isChecked())
        self._canvas.scene().addItem(item)
        self._sam2_marker_items.append(item)

    def _sam2_show_overlay(self, mask_b: np.ndarray) -> None:
        qimg = _mask_bool_to_overlay_qimage(mask_b)
        self._canvas.set_overlay(qimg, opacity=self._sam2_overlay_opacity)

    def _sam2_schedule_infer(self) -> None:
        if self._sam2_worker is not None and self._sam2_worker.isRunning():
            return
        if not get_facade().marks_need_positive(self._sam2_marks):
            self._status_label.setText("SAM2：请至少添加一个正点后再预测")
            return
        if not self._ensure_sam2_model():
            return
        img = self._canvas.get_base_image_np()
        if img is None or img.size == 0:
            return
        key = str(self._current_image_path.resolve()) if self._current_image_path else "_nosave_"
        emb = self._sam2_cache.get(key)
        self._sam2_worker = Sam2Worker(
            self._sam2_model, img, self._sam2_marks, emb, key, self._sam2_gen
        )
        self._sam2_worker.finished_ok.connect(self._on_sam2_infer_ok)
        self._sam2_worker.finished_err.connect(self._on_sam2_infer_err)
        self._sam2_worker.finished.connect(self._on_sam2_worker_finished)
        self._status_label.setText("SAM2：推理中…")
        self._sam2_worker.start()

    def _on_sam2_infer_ok(self, mask_b, emb_out, cache_key: str, gen: int) -> None:
        if gen != self._sam2_gen:
            return
        self._sam2_cache.put(cache_key, emb_out)
        self._sam2_preview_mask = mask_b
        self._sam2_show_overlay(mask_b)
        self._status_label.setText("SAM2：预览已更新，确认后请点击「确定分割并写入标注」")

    def _on_sam2_infer_err(self, msg: str, gen: int) -> None:
        if gen != self._sam2_gen:
            return
        QMessageBox.warning(self, "SAM2 推理失败", msg)

    def _on_sam2_worker_finished(self) -> None:
        w = self.sender()
        if w is self._sam2_worker:
            self._sam2_worker = None

    def wait_workers(self) -> None:
        """关闭窗口时等待 SAM2 推理线程结束，避免 QThread 析构告警。"""
        w = self._sam2_worker
        if w is not None:
            if w.isRunning():
                w.wait(600_000)
            self._sam2_worker = None
        if self._task_id is not None:
            self._platform_app.tasks.wait_task(self._task_id)

    def _sam2_clear_prompts(self) -> None:
        self._sam2_gen += 1
        self._sam2_marks.clear()
        self._sam2_remove_all_markers()
        self._sam2_preview_mask = None
        self._canvas.set_overlay(None)
        self._status_label.setText("已清除 SAM2 提示与预览")

    def _sam2_apply_preview(self) -> None:
        pm = self._sam2_preview_mask
        if pm is None or not np.any(pm):
            QMessageBox.information(
                self,
                "确定分割",
                "当前没有可用的预览 mask。请先放置至少一个正点，待预览出现后再点击「确定分割并写入标注」。",
            )
            return
        layer = self._canvas.get_annotation_layer()
        if layer is None:
            return
        self._undo_stack.push(layer.copy())
        self._annotation_modified = True
        layer.apply_mask(pm.astype(np.uint8), is_erase=False)
        self._canvas.refresh_annotation_display()
        self._sam2_marks.clear()
        self._sam2_remove_all_markers()
        self._sam2_preview_mask = None
        self._canvas.set_overlay(None)
        self._status_label.setText("已确定分割：SAM2 预览已写入标注层（可 Ctrl+Z 撤销）")

    def _settings_key_open_dir(self) -> str:
        return "annotation_last_open_directory"

    def _settings_key_output_dir(self) -> str:
        return "annotation_last_output_directory"

    def _select_output_dir(self) -> None:
        start = str(Path.home())
        saved = QSettings().value(self._settings_key_output_dir(), None)
        if saved is not None and Path(str(saved)).is_dir():
            start = str(saved)
        elif self._output_dir and self._output_dir.is_dir():
            start = str(self._output_dir)
        d = QFileDialog.getExistingDirectory(self, "选择标注输出目录（保存 mask 的目录）", start)
        if d:
            self._output_dir = Path(d)
            QSettings().setValue(self._settings_key_output_dir(), str(self._output_dir.resolve()))
            self._status_label.setText(f"标注输出目录: {self._output_dir}")

    def _open_image(self) -> None:
        start = str(Path.home())
        saved = QSettings().value(self._settings_key_open_dir(), None)
        if saved is not None and Path(str(saved)).is_dir():
            start = str(saved)
        elif self._current_image_path and self._current_image_path.parent.is_dir():
            start = str(self._current_image_path.parent)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图像",
            start,
            "图像 (" + " ".join(f"*{e}" for e in IMAGE_EXTENSIONS) + ")",
        )
        if path:
            self._load_image_at_path(Path(path))

    def _on_dataset_sample_selected(self, dataset_id: str, sample_id: str) -> None:
        if self._current_sample == (dataset_id, sample_id):
            return
        try:
            ds = self._platform_app.datasets.store.load(dataset_id)
        except Exception as e:
            QMessageBox.warning(self, "数据集", str(e))
            return
        sample = ds.sample(sample_id)
        if sample is None:
            return
        if not self._load_image_at_path(ds.abs(sample.image), sample_ref=(dataset_id, sample_id)):
            if self._dataset_panel is not None:
                if self._current_sample and self._current_sample[0] == dataset_id:
                    self._dataset_panel.select_sample(self._current_sample[1], emit=False)
                else:
                    self._dataset_panel.clear_selection()

    def _load_image_at_path(self, path: Path, sample_ref: tuple[str, str] | None = None) -> bool:
        if path == self._current_image_path and sample_ref == self._current_sample:
            return True
        if not path.is_file():
            if sample_ref is not None:
                QMessageBox.warning(self, "打开失败", f"原图不存在，请在「数据集」页运行数据校验:\n{path}")
            return False
        if self._current_image_path is not None and not self._ask_save_before_switch():
            return False
        self._sam2_reset_on_new_image()
        try:
            qimg, arr = _load_qimage(path)
        except Exception as e:
            QMessageBox.warning(self, "打开失败", str(e))
            return False
        self._current_image_path = path
        self._current_sample = sample_ref
        self._sample_base_qimg = qimg if sample_ref is not None else None
        if sample_ref is None and self._dataset_panel is not None:
            self._dataset_panel.clear_selection()
        if path.parent.is_dir() and sample_ref is None:
            QSettings().setValue(self._settings_key_open_dir(), str(path.parent.resolve()))
        self._undo_stack.clear()
        self._canvas.set_image(qimg, arr)
        status = path.name
        if sample_ref is not None:
            status = self._load_dataset_mask(sample_ref, qimg.width(), qimg.height())
        else:
            self._load_existing_mask_if_any(path, qimg.width(), qimg.height())
        self._annotation_modified = False
        self._canvas.set_brush_size(self._brush_spin.value())
        self._status_label.setText(status)
        if sample_ref is not None:
            self._apply_dataset_display()
        else:
            self._canvas.set_annotation_visible(True)
        return True

    def _apply_dataset_display(self, _mode: str | None = None) -> None:
        """在原图与掩码之间切换底图，不改动已加载的标注。"""
        if self._current_sample is None or self._dataset_panel is None or self._sample_base_qimg is None:
            return
        mode = self._dataset_panel.display_mode()
        if mode == "image":
            self._canvas.replace_base_image(self._sample_base_qimg)
            self._canvas.set_annotation_visible(True)
            return
        try:
            ds = self._platform_app.datasets.store.load(self._current_sample[0])
            sample = ds.sample(self._current_sample[1])
            index = ds.load_index_mask(sample) if sample is not None else None
        except Exception as e:
            self._status_label.setText(f"掩码读取失败: {e}")
            return
        if index is None:
            self._status_label.setText("该样本没有掩码，仍显示原图")
            self._dataset_panel.set_display_mode("image")
            return
        if index.shape[:2] != (self._sample_base_qimg.height(), self._sample_base_qimg.width()):
            self._status_label.setText("掩码尺寸与原图不一致，仍显示原图")
            self._dataset_panel.set_display_mode("image")
            return
        from plant_phenotype.datasets.mask_io import preview_mask_rgb
        rgb = np.ascontiguousarray(preview_mask_rgb(index))
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format.Format_RGB888).copy()
        if self._canvas.replace_base_image(qimg):
            self._canvas.set_annotation_visible(False)
            self._status_label.setText("正在显示掩码；切回「原图」可继续标注")

    def _load_dataset_mask(self, sample_ref: tuple[str, str], width: int, height: int) -> str:
        """加载数据集样本的掩码（多类别/实例合并为前景仅供显示），返回状态栏文字。"""
        ds = self._platform_app.datasets.store.load(sample_ref[0])
        sample = ds.sample(sample_ref[1])
        text = f"[{ds.name}] {sample.image_name}"
        try:
            index = ds.load_index_mask(sample)
        except Exception as e:
            return f"{text}　⚠ 掩码读取失败: {e}"
        if index is not None:
            fg = index > 0
            if fg.shape != (height, width):
                return f"{text}　⚠ 掩码尺寸与原图不一致，未加载；请在「数据集」页校验"
            layer = self._canvas.get_annotation_layer()
            if layer is not None:
                layer.apply_mask(fg.astype(np.uint8), is_erase=False)
                self._canvas.refresh_annotation_display()
        reason = ds.protection_reason()
        if reason:
            text += "　⚠ 只读显示（各类别合并为前景）：" + reason
        return text

    def _mask_path_for_image(self, image_path: Path) -> Path | None:
        """返回与图像对应的 mask 路径。

        依次尝试：
        1) 与图像同目录 ``{stem}_mask.png``（dataset-tool 约定）
        2) 父目录的兄弟文件夹 ``masks/{stem}.png``（如 ``data/images/a.png`` → ``data/masks/a.png``）
        3) 图像目录下 ``masks/{stem}.png``
        4) 标注输出目录中的 ``{stem}_mask.png`` 或 ``{stem}.png``
        """
        stem = image_path.stem
        if stem.endswith("_mask"):
            stem = stem[:-5]
        parent = image_path.parent
        candidates: list[Path] = []
        # 1) 同目录 stem_mask.png
        candidates.append(parent / f"{stem}{MASK_SUFFIX}.png")
        # 2) 与 parent 同级 masks/（常见：data/images/x → data/masks/x.png）
        if parent.parent.is_dir():
            sibling_masks = parent.parent / "masks"
            candidates.append(sibling_masks / f"{stem}.png")
            candidates.append(sibling_masks / f"{stem}{MASK_SUFFIX}.png")
        # 3) 当前目录下 masks 子文件夹
        sub_masks = parent / "masks"
        candidates.append(sub_masks / f"{stem}.png")
        candidates.append(sub_masks / f"{stem}{MASK_SUFFIX}.png")
        # 4) 标注输出目录
        if self._output_dir is not None and self._output_dir.is_dir():
            candidates.append(self._output_dir / f"{stem}{MASK_SUFFIX}.png")
            candidates.append(self._output_dir / f"{stem}.png")
        for p in candidates:
            if p.is_file():
                return p
        return None

    def _load_existing_mask_if_any(self, image_path: Path, width: int, height: int) -> None:
        """若存在对应的 mask 文件，则加载到标注层并叠加显示（受透明度调节）；用户可在此基础上继续标注或擦除。"""
        mask_path = self._mask_path_for_image(image_path)
        if mask_path is None:
            return
        try:
            from PIL import Image
            mask_img = Image.open(mask_path)
            mask_arr = np.array(mask_img)
        except Exception:
            return
        if mask_arr.ndim == 2:
            fg = mask_arr > 0
        else:
            fg = np.any(mask_arr > 0, axis=-1)
        if fg.shape[0] != height or fg.shape[1] != width:
            fg_pil = Image.fromarray((fg.astype(np.uint8) * 255))
            fg_pil = fg_pil.resize((width, height), Image.Resampling.NEAREST)
            fg = np.array(fg_pil) > 0
        layer = self._canvas.get_annotation_layer()
        if layer is not None:
            layer.apply_mask(fg.astype(np.uint8), is_erase=False)
            self._canvas.refresh_annotation_display()

    def _undo(self) -> None:
        layer = self._canvas.get_annotation_layer()
        if layer is None or not self._undo_stack.can_undo():
            return
        prev = self._undo_stack.undo(layer)
        if prev is not None:
            layer.replace_with(prev)
            self._canvas.set_annotation_layer(layer)
            self._canvas.scene().update()
            self._status_label.setText("已撤销")

    def _has_unsaved_annotation(self) -> bool:
        """当前图像是否有未保存的修改（仅当用户实际编辑过标注时为 True，仅加载已有 mask 未改不算）。"""
        if self._current_image_path is None or not self._annotation_modified:
            return False
        return self._canvas.get_annotation_layer() is not None

    def _ask_save_before_switch(self) -> bool:
        """切换图片前若存在未保存标注则提示。返回 True 表示可以继续切换，False 表示取消切换。"""
        if not self._has_unsaved_annotation():
            return True
        if self._auto_save_on_switch:
            self._save_mask()
            return True
        dlg = QDialog(self)
        dlg.setWindowTitle("未保存的标注")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("当前图像有未保存的标注，是否保存？"))
        auto_save_cb = QCheckBox("下次默认保存（切换图片时自动保存，不再询问；重启软件后重置）")
        auto_save_cb.setChecked(False)
        layout.addWidget(auto_save_cb)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No | QDialogButtonBox.StandardButton.Cancel,
        )
        result_holder: list = []

        def on_yes():
            self._auto_save_on_switch = auto_save_cb.isChecked()
            result_holder.append(self._save_mask())
            dlg.accept()

        def on_no():
            self._auto_save_on_switch = auto_save_cb.isChecked()
            result_holder.append(True)
            dlg.accept()

        def on_cancel():
            result_holder.append(False)
            dlg.reject()

        yes_btn = btns.button(QDialogButtonBox.StandardButton.Yes)
        no_btn = btns.button(QDialogButtonBox.StandardButton.No)
        cancel_btn = btns.button(QDialogButtonBox.StandardButton.Cancel)
        yes_btn.clicked.connect(on_yes)
        no_btn.clicked.connect(on_no)
        cancel_btn.clicked.connect(on_cancel)
        layout.addWidget(btns)
        dlg.exec()
        if not result_holder:
            return False
        return result_holder[0]

    def _save_mask(self) -> bool:
        """保存当前标注到 mask 文件。返回 True 表示已保存，False 表示未保存（取消或失败）。"""
        layer = self._canvas.get_annotation_layer()
        if layer is None:
            QMessageBox.information(self, "保存 Mask", "请先打开一张图像并标注。")
            return False
        if self._current_sample is not None:
            return self._save_to_dataset(layer.get_foreground_mask())
        # 优先使用标注输出目录，否则使用上次保存目录（勾选默认保存后自动保存时不弹窗）
        effective_out_dir = self._output_dir if self._output_dir is not None else self._last_save_dir
        if effective_out_dir is not None and self._current_image_path is not None:
            effective_out_dir = Path(effective_out_dir)
            if effective_out_dir.is_dir():
                effective_out_dir.mkdir(parents=True, exist_ok=True)
                stem = self._current_image_path.stem
                if stem.endswith("_mask"):
                    stem = stem[:-5]
                out_path = effective_out_dir / f"{stem}{MASK_SUFFIX}.png"
                try:
                    mask = layer.get_foreground_mask()
                    out = (mask.astype(np.uint8)) * 255
                    from PIL import Image
                    Image.fromarray(out).save(out_path)
                    self._annotation_modified = False
                    self._last_save_dir = effective_out_dir
                    self._status_label.setText(f"已保存: {out_path.name}")
                    return True
                except Exception as e:
                    QMessageBox.warning(self, "保存失败", str(e))
                    return False
        default_path = None
        if self._current_image_path is not None:
            stem = self._current_image_path.stem
            if stem.endswith("_mask"):
                stem = stem[:-5]
            default_path = self._current_image_path.parent / f"{stem}{MASK_SUFFIX}.png"
        start = str(default_path) if default_path else str(QSettings().value("last_dir/annotation_save_mask", str(Path.home())))
        if default_path and not default_path.parent.exists():
            start = str(QSettings().value("last_dir/annotation_save_mask", str(Path.home())))
        path, _ = QFileDialog.getSaveFileName(
            self, "保存 Mask", start, "PNG (*.png)",
        )
        if not path:
            return False
        path = Path(path)
        try:
            mask = layer.get_foreground_mask()
            out = (mask.astype(np.uint8)) * 255
            from PIL import Image
            Image.fromarray(out).save(path)
            self._annotation_modified = False
            self._last_save_dir = path.parent
            QSettings().setValue("last_dir/annotation_save_mask", str(path.parent))
            self._status_label.setText(f"已保存: {path.name}")
            return True
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))
            return False

    def _save_to_dataset(self, foreground: np.ndarray) -> bool:
        dataset_id, sample_id = self._current_sample
        service = self._platform_app.datasets
        try:
            ds = service.store.load(dataset_id)
            sample = ds.save_binary_annotation(sample_id, foreground)
        except DatasetProtectedError as e:
            QMessageBox.warning(
                self, "已保护，未保存",
                f"{e}\n\n当前标注工具只能写二值掩码。完整的多类别/实例标注需另行扩展；"
                "如需二值化，请在「数据集」页先复制数据集，再合并类别。",
            )
            return False
        except DatasetError as e:
            QMessageBox.warning(self, "保存失败", str(e))
            return False
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))
            return False
        self._annotation_modified = False
        self._status_label.setText(f"已保存到数据集「{ds.name}」: {sample.mask}")
        service.notify_changed(dataset_id)
        return True

    def _on_brush_shape_changed(self, shape: BrushShape) -> None:
        self._canvas.set_brush_shape(shape)
        if shape == BrushShape.CIRCLE:
            self._brush_circle_btn.setChecked(True)
        else:
            self._brush_square_btn.setChecked(True)

    def _on_annotation_opacity_changed(self, value: int) -> None:
        self._canvas.set_annotation_opacity(value / 100.0)

    def _on_show_mask_toggled(self, checked: bool) -> None:
        self._canvas.set_annotation_visible(checked)

    def _on_annotation_visibility_changed(self, visible: bool) -> None:
        self._show_mask_check.blockSignals(True)
        self._show_mask_check.setChecked(visible)
        self._show_mask_check.blockSignals(False)

    def _get_workdir_image_list(self) -> list:
        root = self._workdir_panel.get_root_path() if self._workdir_panel else None
        if root is None or not root.is_dir():
            return []
        exts = {e.lower() for e in IMAGE_EXTENSIONS}
        return sorted(
            [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts],
            key=lambda x: (str(x).lower(), x.name),
        )

    def _get_sibling_image_list(self) -> list:
        """仅当前图像所在同级目录内的图像列表，用于 ← → 切换，不跨级、不换路径。"""
        if self._current_image_path is None or not self._current_image_path.is_file():
            return []
        parent = self._current_image_path.parent
        if not parent.is_dir():
            return []
        exts = {e.lower() for e in IMAGE_EXTENSIONS}
        return sorted(
            [p for p in parent.iterdir() if p.is_file() and p.suffix.lower() in exts],
            key=lambda x: x.name.lower(),
        )

    def _go_prev_image(self) -> None:
        if self._current_sample is not None and self._dataset_panel is not None:
            self._dataset_panel.select_relative(-1)
            return
        if self._current_image_path is None:
            return
        images = self._get_sibling_image_list()
        if not images:
            return
        cur = self._current_image_path.resolve()
        idx = next((i for i, p in enumerate(images) if p.resolve() == cur), -1)
        if idx <= 0:
            return
        self._load_image_at_path(images[idx - 1])

    def _go_next_image(self) -> None:
        if self._current_sample is not None and self._dataset_panel is not None:
            self._dataset_panel.select_relative(1)
            return
        if self._current_image_path is None:
            return
        images = self._get_sibling_image_list()
        if not images:
            return
        cur = self._current_image_path.resolve()
        idx = next((i for i, p in enumerate(images) if p.resolve() == cur), -1)
        if idx < 0 or idx >= len(images) - 1:
            return
        self._load_image_at_path(images[idx + 1])

    def _on_brush_size_changed(self, value: int) -> None:
        self._canvas.set_brush_size(value)

    def _on_canvas_brush_size_changed(self, value: int) -> None:
        self._brush_spin.blockSignals(True)
        self._brush_spin.setValue(value)
        self._brush_spin.blockSignals(False)

    def _sync_tool_options_page(self, btn=None) -> None:
        """左侧工具切换时，更新画布上方「工具选项」面板。"""
        if btn is None:
            btn = self._tool_group.checkedButton()
        if btn is self._brush_btn or btn is self._erase_btn:
            self._tool_options_stack.setCurrentIndex(0)
        elif btn is self._fill_btn:
            self._tool_options_stack.setCurrentIndex(1)
        elif btn is self._otsu_fill_btn:
            self._tool_options_stack.setCurrentIndex(2)
        elif btn is self._region_grow_btn:
            self._tool_options_stack.setCurrentIndex(3)
        elif btn is self._sam2_btn:
            self._tool_options_stack.setCurrentIndex(4)
        # 工具选项独占第二行。非 SAM2 压成单行；SAM2 为两行控件，另留横向滚动条高度
        if btn is self._sam2_btn:
            self._tool_options_stack.setMinimumHeight(58)
            self._tool_options_stack.setMaximumHeight(74)
        else:
            self._tool_options_stack.setMinimumHeight(0)
            self._tool_options_stack.setMaximumHeight(28)
        self._top_bar.updateGeometry()

    def _on_tool_changed(self, btn) -> None:
        self._sync_tool_options_page(btn)
        is_sam2 = btn is self._sam2_btn
        self._canvas.set_sam2_mode(is_sam2)
        for it in self._sam2_marker_items:
            it.setVisible(is_sam2)
        if is_sam2:
            self._canvas.set_erase(False)
            self._canvas.set_fill(False)
            self._canvas.set_otsu_fill(False)
            self._canvas.set_region_grow_fill(False)
            self._canvas.set_sam2_positive_point(self._sam2_pos_btn.isChecked())
            self._canvas.set_sam2_point_radius(self._sam2_point_radius_spin.value())
            if self._sam2_preview_mask is not None:
                self._sam2_show_overlay(self._sam2_preview_mask)
        else:
            self._canvas.set_erase(btn is self._erase_btn)
            self._canvas.set_fill(btn is self._fill_btn)
            self._canvas.set_otsu_fill(btn is self._otsu_fill_btn)
            self._canvas.set_region_grow_fill(btn is self._region_grow_btn)
            self._canvas.set_overlay(None)

    def _on_region_grow_tolerance_changed(self, value: int) -> None:
        self._canvas.set_region_grow_tolerance(value)
        self._region_grow_label.setText(str(value))

    def _on_otsu_threshold_changed(self, value: int) -> None:
        self._canvas.set_otsu_threshold(value)
        self._otsu_threshold_label.setText(str(value))

    def _on_otsu_auto_clicked(self) -> None:
        """根据当前底图计算大津阈值并更新滑块与画布。"""
        base = self._canvas.get_base_image_np()
        if base is None or base.size == 0:
            QMessageBox.information(
                self,
                "取大津",
                "请先打开一张图像。",
            )
            return
        th = get_facade().compute_otsu_threshold(base)
        self._otsu_threshold_slider.blockSignals(True)
        self._otsu_threshold_slider.setValue(th)
        self._otsu_threshold_slider.blockSignals(False)
        self._otsu_threshold_label.setText(str(th))
        self._canvas.set_otsu_threshold(th)

    def _run_preprocessing(self) -> None:
        """对目录执行切块：已标注原图（含 *_mask.png）或直接切割原图。默认源目录为当前工作目录。"""
        if self._task_id is not None:
            return
        start = str(Path.home())
        work_root = self._workdir_panel.get_root_path() if self._workdir_panel else None
        if work_root is not None and work_root.is_dir():
            start = str(work_root)
        elif self._output_dir and self._output_dir.is_dir():
            start = str(self._output_dir)
        elif self._tiles_base_dir and self._tiles_base_dir.is_dir():
            start = str(self._tiles_base_dir)
        saved_src = QSettings().value("last_dir/annotation_tiles_source", None)
        if saved_src and Path(str(saved_src)).is_dir():
            start = str(saved_src)
        source_dir = QFileDialog.getExistingDirectory(
            self,
            "选择要切块的目录（原图所在，可有或没有 *_mask.png）",
            start,
        )
        if not source_dir:
            return
        source_dir = Path(source_dir)
        QSettings().setValue("last_dir/annotation_tiles_source", str(source_dir))
        tiles_start = str(self._tiles_base_dir) if self._tiles_base_dir and self._tiles_base_dir.is_dir() else str(source_dir)
        saved_tiles = QSettings().value("last_dir/annotation_tiles_dir", None)
        if saved_tiles and Path(str(saved_tiles)).is_dir():
            tiles_start = str(saved_tiles)
        tiles_dir = QFileDialog.getExistingDirectory(
            self, "选择切块输出根目录", tiles_start,
        )
        if not tiles_dir:
            return
        self._tiles_base_dir = Path(tiles_dir)
        QSettings().setValue("last_dir/annotation_tiles_dir", tiles_dir)
        dlg = QDialog(self)
        dlg.setWindowTitle("切块参数")
        form = QFormLayout(dlg)
        tile_spin = QSpinBox()
        tile_spin.setRange(64, 2048)
        tile_spin.setValue(DEFAULT_TILE_SIZE)
        tile_spin.setSingleStep(128)
        form.addRow("切块大小 (像素):", tile_spin)
        ratio_spin = QDoubleSpinBox()
        ratio_spin.setRange(0.0, 1.0)
        ratio_spin.setValue(DEFAULT_MIN_FOREGROUND_RATIO)
        ratio_spin.setSingleStep(0.01)
        ratio_spin.setDecimals(4)
        form.addRow("前景比例阈值 (低于则排除):", ratio_spin)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        tile_size = tile_spin.value()
        min_ratio = ratio_spin.value()
        self._start_tool_task(
            "annotation.tile_directory",
            {
                "annotated_dir": str(source_dir),
                "tiles_base_dir": str(self._tiles_base_dir),
                "tile_size": tile_size,
                "min_foreground_ratio": min_ratio,
            },
            "tile",
        )

    def _run_reassemble(self) -> None:
        """从切块目录按 meta.json 拼接还原整图与 mask。"""
        if self._task_id is not None:
            return
        start = str(self._tiles_base_dir) if self._tiles_base_dir and self._tiles_base_dir.is_dir() else str(Path.home())
        saved = QSettings().value("last_dir/annotation_reassemble_dir", None)
        if saved and Path(str(saved)).is_dir():
            start = str(saved)
        tile_dir = QFileDialog.getExistingDirectory(self, "选择切块目录（含 meta.json）", start)
        if not tile_dir:
            return
        tile_dir = Path(tile_dir)
        QSettings().setValue("last_dir/annotation_reassemble_dir", str(tile_dir))
        if not (tile_dir / "meta.json").exists():
            QMessageBox.warning(self, "错误", "该目录下无 meta.json")
            return
        save_start = str(QSettings().value("last_dir/annotation_reassemble_save", str(Path.home())))
        out_path, _ = QFileDialog.getSaveFileName(self, "保存还原后的图像", save_start, "PNG (*.png)")
        if not out_path:
            return
        out_path = Path(out_path)
        QSettings().setValue("last_dir/annotation_reassemble_save", str(out_path.parent))
        self._start_tool_task(
            "annotation.reassemble_tiles",
            {
                "tile_dir": str(tile_dir),
                "output_image_path": str(out_path),
                "output_mask_path": str(out_path.parent / (out_path.stem + MASK_SUFFIX + ".png")),
            },
            "reassemble",
        )

    def _start_tool_task(self, tool_id: str, inputs: dict, kind: str) -> None:
        self._task_kind = kind
        self._tile_btn.setEnabled(False)
        self._reassemble_btn.setEnabled(False)
        self._task_id = self._platform_app.tasks.run_tool(tool_id, inputs)

    def _on_task_state_changed(self, state) -> None:
        if state.id != self._task_id or state.status not in ("success", "failed", "cancelled"):
            return
        kind = self._task_kind
        self._task_id = None
        self._task_kind = None
        self._tile_btn.setEnabled(True)
        self._reassemble_btn.setEnabled(True)
        if state.status != "success":
            QMessageBox.critical(self, "任务失败", state.error or state.message)
        elif kind == "tile":
            count = state.result.metrics["image_count"]
            if count:
                QMessageBox.information(self, "完成", f"已对 {count} 张图执行切块，结果写入: {self._tiles_base_dir}")
            else:
                QMessageBox.warning(self, "切块", "源目录中未找到可切块的图像。")
        else:
            artifact_paths = [str(item.path) for item in state.result.artifacts]
            QMessageBox.information(self, "完成", "已保存: " + " 与 ".join(artifact_paths))


def build_annotation_tab(parent=None, app=None) -> QWidget:
    """构建数据标注 Tab 控件（供插件注册调用）。"""
    return AnnotationTab(parent, platform_app=app)
