# -*- coding: utf-8 -*-
"""全局主题：Material Design 3 设计令牌（中性灰表面 + slate blue 强调色）与 QSS。

只负责控件外观；画布内的绘制颜色（标注点、骨架、笔刷等）由各 canvas 自行管理。

约定的动态属性（在控件上 setProperty 后由本表生效）：
    mdRole="filled"   主操作按钮（每个页面通常仅一个）
    mdRole="tonal"    次要强调按钮
    mdRole="text"     无边框按钮，左对齐加粗，用于折叠区标题
    mdRole="status"   底部状态提示文字
    mdRole="caption"  弱化的辅助说明文字
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from string import Template

APP_VERSION = "1.0.0"

_FONT_FAMILY = '"Microsoft YaHei", "PingFang SC", sans-serif'
_FONT_SIZE = "13px"

# MD3 形状令牌
_RADIUS_FIELD = "6px"
_RADIUS_BUTTON = "10px"
_RADIUS_CONTAINER = "12px"
_RADIUS_SMALL = "4px"

_LIGHT_TOKENS = {
    "primary": "#456990",
    "on_primary": "#FFFFFF",
    "primary_container": "#D6E3F5",
    "on_primary_container": "#16324F",
    "surface": "#FAFAFC",
    "surface_container_lowest": "#FFFFFF",
    "surface_container_low": "#F4F5F7",
    "surface_container": "#EEEFF2",
    "surface_container_high": "#E8E9ED",
    "surface_container_highest": "#E2E3E8",
    "on_surface": "#1B1C1E",
    "on_surface_variant": "#494A4E",
    "outline": "#7A7C81",
    "outline_variant": "#C8C9CE",
    "error": "#B3261E",
    # 图像预览始终使用深色底，便于观察掩码叠加
    "canvas_backdrop": "#202124",
    "on_canvas_backdrop": "#C9CACD",
}

_DARK_TOKENS = {
    "primary": "#A8C4E0",
    "on_primary": "#133049",
    "primary_container": "#2C4A68",
    "on_primary_container": "#D6E3F5",
    "surface": "#131416",
    "surface_container_lowest": "#0E0F10",
    "surface_container_low": "#1B1C1E",
    "surface_container": "#1F2022",
    "surface_container_high": "#292A2C",
    "surface_container_highest": "#343538",
    "on_surface": "#E3E2E6",
    "on_surface_variant": "#C3C5CA",
    "outline": "#8D8F94",
    "outline_variant": "#43454A",
    "error": "#F2B8B5",
    "canvas_backdrop": "#17181A",
    "on_canvas_backdrop": "#B4B6BA",
}


def _to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgba(hex_color: str, alpha_percent: int) -> str:
    """MD3 state layer：在父背景之上叠加半透明强调色。"""
    r, g, b = _to_rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha_percent}%)"


def _mix(base: str, overlay: str, overlay_percent: int) -> str:
    """把 overlay 按百分比混入 base，得到不透明色（用于实心按钮的 hover/pressed）。"""
    br, bg, bb = _to_rgb(base)
    orr, og, ob = _to_rgb(overlay)
    w = overlay_percent / 100.0
    return "#{:02X}{:02X}{:02X}".format(
        round(br + (orr - br) * w),
        round(bg + (og - bg) * w),
        round(bb + (ob - bb) * w),
    )


_QSS_TEMPLATE = Template("""
/* ---------- 基础 ---------- */
QWidget {
    font-family: $font_family;
    font-size: $font_size;
}
QMainWindow, QDialog {
    background: $surface;
}
QScrollArea {
    border: none;
    background: transparent;
}
QToolTip {
    background: $surface_container_highest;
    color: $on_surface;
    border: 1px solid $outline_variant;
    border-radius: $radius_field;
    padding: 4px 8px;
}

/* ---------- 顶部菜单 / 状态栏 ---------- */
QMenuBar {
    background: $surface;
    color: $on_surface;
    border: none;
    padding: 2px 4px;
}
QMenuBar::item {
    background: transparent;
    padding: 5px 12px;
    border-radius: $radius_field;
}
QMenuBar::item:selected { background: $state_hover; }
QMenuBar::item:pressed { background: $state_pressed; }
QMenu {
    background: $surface_container_low;
    color: $on_surface;
    border: 1px solid $outline_variant;
    border-radius: $radius_button;
    padding: 4px;
}
QMenu::item {
    padding: 6px 18px;
    border-radius: $radius_field;
}
QMenu::item:selected {
    background: $primary_container;
    color: $on_primary_container;
}
QMenu::separator {
    height: 1px;
    background: $outline_variant;
    margin: 4px 8px;
}
QStatusBar {
    background: $surface_container_low;
    color: $on_surface_variant;
    border-top: 1px solid $outline_variant;
}
QStatusBar::item { border: none; }

/* ---------- Tab ---------- */
QTabWidget::pane {
    border: none;
    background: $surface;
}
QTabBar {
    background: $surface;
    qproperty-drawBase: 0;
}
QTabBar::tab {
    background: transparent;
    color: $on_surface_variant;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 6px 18px;
    min-height: 30px;
    margin-right: 2px;
}
QTabBar::tab:hover {
    background: $state_hover;
    color: $on_surface;
}
QTabBar::tab:selected {
    color: $primary;
    border-bottom: 2px solid $primary;
    font-weight: 600;
}

/* ---------- 容器 ---------- */
QGroupBox {
    background: $surface_container_low;
    border: 1px solid $outline_variant;
    border-radius: $radius_container;
    margin-top: 8px;
    padding: 8px 6px 6px 6px;
    font-weight: 600;
    color: $on_surface_variant;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
    color: $on_surface_variant;
}
QSplitter::handle { background: transparent; }
QSplitter::handle:horizontal { width: 6px; }
QSplitter::handle:vertical { height: 6px; }
QSplitter::handle:hover { background: $state_selected; }

/* ---------- 按钮 ---------- */
QPushButton {
    background: transparent;
    color: $on_surface;
    border: 1px solid $outline_variant;
    border-radius: $radius_button;
    padding: 3px 14px;
    min-height: 28px;
}
QPushButton:hover {
    background: $state_hover;
    border-color: $primary;
}
QPushButton:pressed { background: $state_pressed; }
QPushButton:checked {
    background: $primary_container;
    color: $on_primary_container;
    border-color: $primary_container;
    font-weight: 600;
}
QPushButton:checked:hover { background: $tonal_hover; }
QPushButton:disabled {
    background: transparent;
    color: $outline;
    border-color: $outline_variant;
}
/* 主/次强调按钮跨页面保持同一尺寸基线 */
QPushButton[mdRole="filled"], QPushButton[mdRole="tonal"] { min-width: 88px; }
QPushButton[mdRole="filled"] {
    background: $primary;
    color: $on_primary;
    border: 1px solid $primary;
    font-weight: 600;
}
QPushButton[mdRole="filled"]:hover {
    background: $filled_hover;
    border-color: $filled_hover;
}
QPushButton[mdRole="filled"]:pressed {
    background: $filled_pressed;
    border-color: $filled_pressed;
}
QPushButton[mdRole="filled"]:disabled {
    background: $surface_container_high;
    color: $outline;
    border-color: $surface_container_high;
}
QPushButton[mdRole="tonal"] {
    background: $primary_container;
    color: $on_primary_container;
    border: 1px solid $primary_container;
    font-weight: 600;
}
QPushButton[mdRole="tonal"]:hover {
    background: $tonal_hover;
    border-color: $tonal_hover;
}
QPushButton[mdRole="tonal"]:pressed { background: $tonal_pressed; }
QPushButton[mdRole="tonal"]:disabled {
    background: $surface_container;
    color: $outline;
    border-color: $surface_container;
}
/* 折叠区标题按钮：无边框、左对齐 */
QPushButton[mdRole="text"] {
    background: transparent;
    color: $on_surface;
    border: none;
    border-radius: $radius_field;
    padding: 4px 6px;
    text-align: left;
    font-weight: 600;
    min-height: 24px;
}
QPushButton[mdRole="text"]:hover { background: $state_hover; }
QPushButton[mdRole="text"]:pressed { background: $state_pressed; }
QPushButton[mdRole="text"]:checked {
    background: transparent;
    color: $on_surface;
}
QPushButton[mdRole="text"]:checked:hover { background: $state_hover; }

/* ---------- 输入控件 ---------- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: $surface_container_lowest;
    color: $on_surface;
    border: 1px solid $outline_variant;
    border-radius: $radius_field;
    padding: 3px 6px;
    min-height: 26px;
    selection-background-color: $primary;
    selection-color: $on_primary;
}
QLineEdit:hover, QPlainTextEdit:hover, QTextEdit:hover,
QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {
    border-color: $outline;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QComboBox:on {
    border-color: $primary;
}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    background: $surface_container;
    color: $outline;
}
QComboBox QAbstractItemView {
    background: $surface_container_low;
    color: $on_surface;
    border: 1px solid $outline_variant;
    selection-background-color: $primary_container;
    selection-color: $on_primary_container;
    outline: none;
}
QCheckBox, QRadioButton { spacing: 6px; }

/* ---------- 滑块 / 进度 ---------- */
QSlider::groove:horizontal {
    height: 4px;
    background: $surface_container_highest;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: $primary;
    border-radius: 2px;
}
QSlider::add-page:horizontal {
    background: $surface_container_highest;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 12px;
    height: 12px;
    margin: -5px 0;
    background: $primary;
    border: none;
    border-radius: 6px;
}
QSlider::handle:horizontal:hover { background: $filled_hover; }
QSlider::handle:horizontal:disabled { background: $outline; }
QSlider::sub-page:horizontal:disabled { background: $outline_variant; }
QProgressBar {
    background: $surface_container_highest;
    border: none;
    border-radius: $radius_field;
    text-align: center;
    color: $on_surface_variant;
    min-height: 14px;
}
QProgressBar::chunk {
    background: $primary;
    border-radius: $radius_field;
}

/* ---------- 滚动条 ---------- */
QScrollBar:vertical {
    background: transparent;
    width: 12px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: $scroll_handle;
    min-height: 28px;
    border-radius: $radius_small;
    margin: 2px;
}
QScrollBar::handle:vertical:hover { background: $scroll_handle_hover; }
QScrollBar:horizontal {
    background: transparent;
    height: 12px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: $scroll_handle;
    min-width: 28px;
    border-radius: $radius_small;
    margin: 2px;
}
QScrollBar::handle:horizontal:hover { background: $scroll_handle_hover; }
QScrollBar::add-line, QScrollBar::sub-line {
    width: 0;
    height: 0;
    background: none;
    border: none;
}
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

/* ---------- 列表 / 树 / 表格 ---------- */
QTreeView, QListView, QTableView, QTableWidget {
    background: $surface_container_lowest;
    alternate-background-color: $surface_container_low;
    color: $on_surface;
    border: 1px solid $outline_variant;
    border-radius: $radius_button;
    outline: none;
    gridline-color: $outline_variant;
}
QTreeView::item, QListView::item, QTableView::item {
    min-height: 22px;
    padding: 1px 2px;
}
QTreeView::item:hover, QListView::item:hover, QTableView::item:hover {
    background: $state_hover;
}
QTreeView::item:selected, QListView::item:selected, QTableView::item:selected {
    background: $primary_container;
    color: $on_primary_container;
}
QHeaderView::section {
    background: $surface_container_high;
    color: $on_surface_variant;
    border: none;
    border-bottom: 1px solid $outline_variant;
    padding: 4px 8px;
    font-weight: 600;
}
QTableCornerButton::section {
    background: $surface_container_high;
    border: none;
}

/* ---------- 语义化文字 ---------- */
QLabel[mdRole="status"] {
    color: $on_surface_variant;
    padding: 4px 8px;
}
QLabel[mdRole="caption"] {
    color: $on_surface_variant;
    font-size: 12px;
}

/* ---------- 紧凑表单页（mdDensity="compact"）：按钮与输入框同高 ---------- */
[mdDensity="compact"] QPushButton {
    padding: 2px 10px;
    min-height: 20px;
}
[mdDensity="compact"] QLineEdit,
[mdDensity="compact"] QSpinBox,
[mdDensity="compact"] QDoubleSpinBox {
    padding: 2px 6px;
    min-height: 20px;
}
/* 上下 padding 小于 3px 时 Qt 算出的弹出列表高度偏小，会出现滚动箭头 */
[mdDensity="compact"] QComboBox {
    padding: 3px 6px;
    min-height: 18px;
}

/* ---------- 紧凑顶栏（标注 / 根系表型） ---------- */
#annotationTopBar, #rootPhenotypeTopBar {
    background: $surface_container_low;
    border-bottom: 1px solid $outline_variant;
    padding: 0px;
    margin: 0px;
}
#annotationTopBar QLabel, #rootPhenotypeTopBar QLabel {
    margin: 0px;
    padding: 0px;
    color: $on_surface_variant;
}
#annotationTopBar QPushButton, #rootPhenotypeTopBar QPushButton {
    padding: 0px 8px;
    min-height: 0px;
    max-height: 26px;
    border-radius: 8px;
}
#annotationTopBar QCheckBox, #rootPhenotypeTopBar QCheckBox {
    spacing: 4px;
    min-height: 0px;
    max-height: 26px;
    padding: 0px;
}
#annotationTopBar QSpinBox, #rootPhenotypeTopBar QSpinBox {
    padding: 0px 4px;
    min-height: 18px;
    max-height: 28px;
}
#annotationTopBar QSlider, #rootPhenotypeTopBar QSlider {
    min-height: 16px;
    max-height: 24px;
}
#rootPhenotypeTopBar QProgressBar {
    min-height: 14px;
    max-height: 18px;
}

/* ---------- 工具侧栏 ---------- */
/* 统一高度，使两行文案的按钮与单行按钮等高 */
#toolsSideBar QPushButton {
    padding: 4px 6px;
    text-align: center;
    min-height: 40px;
}

/* ---------- 根系表型参数侧栏（宽于标注图标栏） ---------- */
#rootPhenotypeSideBar {
    background: $surface;
    border-right: 1px solid $outline_variant;
}
#rootPhenotypeSideBar QScrollArea,
#rootPhenotypeSideBar > QWidget > QScrollArea {
    background: transparent;
    border: none;
}
#rootPhenotypeParams {
    background: $surface;
}
#rootPhenotypeSideBar QGroupBox {
    background: $surface_container_low;
    border: 1px solid $outline_variant;
    border-radius: $radius_container;
    margin-top: 10px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}
#rootPhenotypeSideBar QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: $on_surface_variant;
}
#rootPhenotypeSideBar QPushButton[mdRole="text"] {
    text-align: left;
    padding: 2px 10px;
    min-height: 20px;
    border: 1px solid $outline_variant;
    border-radius: $radius_container;
    background: $surface_container_low;
    font-weight: 600;
    color: $on_surface_variant;
}
#rootPhenotypeSideBar QPushButton[mdRole="text"]:checked {
    background: $surface_container_low;
    color: $on_surface;
    border-color: $outline_variant;
}
#rootPhenotypeSideBar QFormLayout,
#rootPhenotypeSideBar QLabel {
    color: $on_surface;
}
#rootPhenotypeSideBar QLineEdit,
#rootPhenotypeSideBar QSpinBox,
#rootPhenotypeSideBar QDoubleSpinBox {
    border-radius: 8px;
}

/* ---------- 纯图标按钮：固定为正方形 ---------- */
QPushButton[mdShape="square"],
#annotationTopBar QPushButton[mdShape="square"] {
    min-width: 26px;
    max-width: 26px;
    min-height: 26px;
    max-height: 26px;
    padding: 0px;
}

/* ---------- 图像预览深色底 ---------- */
QGraphicsView#previewCanvas,
QGraphicsView#previewCanvas > QWidget#qt_scrollarea_viewport {
    background: $canvas_backdrop;
}
QGraphicsView#previewCanvas, QLabel#previewPlaceholder {
    border: 1px solid $outline_variant;
}
QLabel#previewPlaceholder {
    background: $canvas_backdrop;
    color: $on_canvas_backdrop;
}
""")


# 勾选图形只能由 QSS 引用文件，无法内联绘制；按主色生成一次后缓存复用
_INDICATOR_SVGS = {
    "check": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 18 18">'
        '<path d="M3.5 9.5 L7 13 L14.5 5.2" fill="none" stroke="{color}"'
        ' stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    ),
    "dot": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 18 18">'
        '<circle cx="9" cy="9" r="4.2" fill="{color}"/></svg>'
    ),
}

_INDICATOR_TEMPLATE = Template("""
QCheckBox::indicator, QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border: 2px solid $outline;
    background: transparent;
}
QCheckBox::indicator { border-radius: $radius_small; }
QRadioButton::indicator { border-radius: 9px; }
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: $primary; }
QCheckBox::indicator:checked, QCheckBox::indicator:indeterminate,
QRadioButton::indicator:checked {
    background: $primary;
    border-color: $primary;
}
QCheckBox::indicator:checked { image: url("$icon_check"); }
QRadioButton::indicator:checked { image: url("$icon_dot"); }
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {
    border-color: $outline_variant;
}
QCheckBox::indicator:checked:disabled, QRadioButton::indicator:checked:disabled {
    background: $outline_variant;
    border-color: $outline_variant;
}
""")


def _indicator_icons(tokens: dict) -> dict | None:
    """生成勾选/圆点图形文件，失败时返回 None（退回基础样式自绘）。"""
    color = tokens["on_primary"]
    try:
        cache_dir = Path(tempfile.gettempdir()) / "plant_phenotype_theme"
        cache_dir.mkdir(parents=True, exist_ok=True)
        icons = {}
        for name, svg in _INDICATOR_SVGS.items():
            path = cache_dir / f"{name}_{color.lstrip('#')}.svg"
            if not path.is_file():
                path.write_text(svg.format(color=color), encoding="utf-8")
            icons[f"icon_{name}"] = path.as_posix()
        return icons
    except OSError:
        return None


def _build_context(tokens: dict) -> dict:
    ctx = dict(tokens)
    primary = tokens["primary"]
    ctx.update(
        {
            "font_family": _FONT_FAMILY,
            "font_size": _FONT_SIZE,
            "radius_field": _RADIUS_FIELD,
            "radius_button": _RADIUS_BUTTON,
            "radius_container": _RADIUS_CONTAINER,
            "radius_small": _RADIUS_SMALL,
            "state_hover": _rgba(primary, 10),
            "state_pressed": _rgba(primary, 20),
            "state_selected": _rgba(primary, 26),
            "filled_hover": _mix(primary, tokens["on_primary"], 12),
            "filled_pressed": _mix(primary, tokens["on_primary"], 22),
            "tonal_hover": _mix(tokens["primary_container"], tokens["on_primary_container"], 10),
            "tonal_pressed": _mix(tokens["primary_container"], tokens["on_primary_container"], 18),
            "scroll_handle": _rgba(tokens["on_surface_variant"], 32),
            "scroll_handle_hover": _rgba(tokens["on_surface_variant"], 55),
        }
    )
    return ctx


def build_stylesheet(tokens: dict) -> str:
    """按给定令牌生成完整 QSS。"""
    ctx = _build_context(tokens)
    qss = _QSS_TEMPLATE.substitute(ctx)
    icons = _indicator_icons(tokens)
    if icons is not None:
        qss += _INDICATOR_TEMPLATE.substitute({**ctx, **icons})
    return qss


def _build_palette(tokens: dict):
    """QSS 未覆盖的部分（复选框、单选框、原生对话框等）由 palette 统一配色。"""
    from PyQt6.QtGui import QColor, QPalette

    pal = QPalette()
    role = QPalette.ColorRole
    group = QPalette.ColorGroup
    c = {k: QColor(v) for k, v in tokens.items()}

    pal.setColor(role.Window, c["surface"])
    pal.setColor(role.WindowText, c["on_surface"])
    pal.setColor(role.Base, c["surface_container_lowest"])
    pal.setColor(role.AlternateBase, c["surface_container_low"])
    pal.setColor(role.Text, c["on_surface"])
    pal.setColor(role.PlaceholderText, c["on_surface_variant"])
    pal.setColor(role.Button, c["surface_container"])
    pal.setColor(role.ButtonText, c["on_surface"])
    pal.setColor(role.Highlight, c["primary"])
    pal.setColor(role.HighlightedText, c["on_primary"])
    pal.setColor(role.ToolTipBase, c["surface_container_highest"])
    pal.setColor(role.ToolTipText, c["on_surface"])
    pal.setColor(role.Link, c["primary"])
    pal.setColor(role.LinkVisited, c["primary"])
    pal.setColor(role.Light, c["surface_container_lowest"])
    pal.setColor(role.Midlight, c["surface_container"])
    pal.setColor(role.Mid, c["outline_variant"])
    pal.setColor(role.Dark, c["outline"])
    pal.setColor(role.Shadow, c["outline_variant"])
    pal.setColor(role.BrightText, c["error"])

    for disabled_role in (role.WindowText, role.Text, role.ButtonText):
        pal.setColor(group.Disabled, disabled_role, c["outline"])
    pal.setColor(group.Disabled, role.Base, c["surface_container"])
    pal.setColor(group.Disabled, role.Button, c["surface_container"])
    pal.setColor(group.Disabled, role.Highlight, c["surface_container_highest"])
    return pal


def _is_dark_mode(app) -> bool:
    """检测系统是否处于深色模式（须在应用自定义 palette 之前调用）。"""
    try:
        palette = app.palette()
        from PyQt6.QtGui import QPalette
        bg = palette.color(QPalette.ColorRole.Window)
        # 背景亮度低于 128 视为深色模式
        return bg.lightness() < 128
    except Exception:
        return False


def apply_app_theme(app) -> None:
    """在 QApplication 启动后调用：按系统深浅色套用 MD3 令牌。"""
    tokens = _DARK_TOKENS if _is_dark_mode(app) else _LIGHT_TOKENS
    try:
        from PyQt6.QtWidgets import QStyleFactory

        # Fusion 是各平台表现一致的基础样式，保证 QSS 与 palette 生效范围可预期
        fusion = QStyleFactory.create("Fusion")
        if fusion is not None:
            app.setStyle(fusion)
    except Exception:
        pass
    try:
        app.setPalette(_build_palette(tokens))
    except Exception:
        pass
    app.setStyleSheet(build_stylesheet(tokens))


def window_title() -> str:
    return f"植物表型分析软件 v{APP_VERSION}"
