# -*- coding: utf-8 -*-
"""可缩放、平移的画布，显示底图 + 可选预测叠加 + 标注层。"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QEventLoop, QEvent
from PyQt6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QColor,
    QPen,
    QBrush,
    QPolygonF,
    QWheelEvent,
    QMouseEvent,
    QCursor,
    QKeyEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsItem,
    QApplication,
)

from plant_phenotype.annotation.facade import AnnotationLayer, BrushShape, get_facade


class BrushPreviewItem(QGraphicsItem):
    """笔刷预览：场景中的圆/方，中心=鼠标位置，半径=笔刷大小。超出 viewport 部分由渲染自动裁剪。"""

    def __init__(self, brush_size: int, shape: BrushShape, is_erase: bool):
        super().__init__()
        self._brush_size = brush_size
        self._shape = shape
        self._is_erase = is_erase
        self.setZValue(100)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setAcceptHoverEvents(False)

    def set_brush(self, size: int, shape: BrushShape, is_erase: bool) -> None:
        if self._brush_size != size or self._shape != shape or self._is_erase != is_erase:
            self.prepareGeometryChange()
            self._brush_size = size
            self._shape = shape
            self._is_erase = is_erase

    def boundingRect(self) -> QRectF:
        r = self._brush_size / 2.0 + 2
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter: QPainter, option, widget) -> None:
        r = self._brush_size / 2.0
        if self._is_erase:
            pen = QPen(QColor(220, 50, 50), 2)
        else:
            pen = QPen(QColor(0, 160, 0), 2)
        pen.setCosmetic(True)  # 边宽固定为 2 像素，不随缩放变粗
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(-r, -r, 2 * r, 2 * r)
        if self._shape == BrushShape.CIRCLE:
            painter.drawEllipse(rect)
        else:
            painter.drawRect(rect)


class Sam2PointPreviewItem(QGraphicsItem):
    """SAM2 点提示：鼠标跟随的圆预览（大小与正/负点颜色与落点一致）。"""

    def __init__(self, radius: int, positive: bool):
        super().__init__()
        self._radius = max(2, min(40, int(radius)))
        self._positive = bool(positive)
        self.setZValue(95)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setAcceptHoverEvents(False)

    def set_radius(self, radius: int) -> None:
        r = max(2, min(40, int(radius)))
        if r != self._radius:
            self.prepareGeometryChange()
            self._radius = r

    def set_positive(self, positive: bool) -> None:
        p = bool(positive)
        if p != self._positive:
            self._positive = p
            self.update()

    def boundingRect(self) -> QRectF:
        m = 4.0
        rr = self._radius + m
        return QRectF(-rr, -rr, 2 * rr, 2 * rr)

    def paint(self, painter: QPainter, option, widget) -> None:
        col = QColor(0, 200, 0) if self._positive else QColor(220, 40, 40)
        pen = QPen(col, 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 72)))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = float(self._radius)
        painter.drawEllipse(QRectF(-r, -r, 2 * r, 2 * r))


class AnnotationGraphicsItem(QGraphicsItem):
    def __init__(self, layer: AnnotationLayer):
        super().__init__()
        self.layer = layer
        self.setZValue(10)
        # 避免场景缓存旧位图；标注数据每帧从 layer 读取
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.layer.width, self.layer.height)

    def paint(self, painter: QPainter, option, widget) -> None:
        arr = self.layer.get_data_readonly()
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        h, w = arr.shape[:2]
        if w <= 0 or h <= 0:
            return
        # 与 _load_qimage 一致：先 copy，避免 QImage 挂 numpy 缓冲区在部分环境下不绘制/花屏
        qimg = QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(0, 0, qimg)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)


def _make_fill_cursor() -> QCursor:
    """油漆桶工具光标：倾斜箭头三角形，热点在左上顶点，青色。"""
    w, h = 22, 22
    pix = QPixmap(w, h)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    # 箭头三角形：左上为尖，热点在 (0,0)
    arrow = QPolygonF([
        QPointF(0, 0),      # 左上顶点 = 热点
        QPointF(0, 20),     # 左下
        QPointF(12, 16),     # 右侧
    ])
    cyan = QColor(0, 255, 255)
    painter.setPen(QPen(cyan, 1.5))
    painter.setBrush(QBrush(cyan, Qt.BrushStyle.SolidPattern))
    painter.drawPolygon(arrow)
    painter.end()
    return QCursor(pix, 0, 0)


class CanvasView(QGraphicsView):
    brushSizeChanged = pyqtSignal(int)
    magicWandClicked = pyqtSignal(int, int)
    sam2PromptClicked = pyqtSignal(int, int, bool)  # x, y, is_positive
    sam2RemoveNearestRequested = pyqtSignal(int, int)  # image x, y（删除附近提示点）
    sam2PointRadiusChanged = pyqtSignal(int)  # Ctrl+滚轮 调整点半径时同步工具栏
    annotationVisibilityChanged = pyqtSignal(bool)
    prevImageRequested = pyqtSignal()
    nextImageRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._base_item: Optional[QGraphicsPixmapItem] = None
        self._base_image_np: Optional[np.ndarray] = None
        self._overlay_item: Optional[QGraphicsPixmapItem] = None
        self._annotation_item: Optional[AnnotationGraphicsItem] = None
        self._annotation_layer: Optional[AnnotationLayer] = None
        self._annotation_opacity: float = 1.0
        self._last_pos: Optional[QPointF] = None
        self._brush_size = 20
        self._brush_shape = BrushShape.CIRCLE
        self._is_erase = False
        self._is_fill = False
        self._is_otsu_fill = False
        self._is_region_grow_fill = False
        self._is_magic_wand = False
        self._is_sam2_mode = False
        self._sam2_positive_point = True
        self._sam2_point_radius = 8
        self._sam2_point_preview_item: Optional[Sam2PointPreviewItem] = None
        self._fill_tolerance = 32
        self._otsu_threshold = 128
        self._region_grow_tolerance = 32
        self._on_stroke_done: Optional[Callable[[], None]] = None
        self._on_before_edit: Optional[Callable[[], None]] = None
        self._scene_rect = QRectF(0, 0, 1, 1)
        self._brush_preview_item: Optional[BrushPreviewItem] = None
        self._mouse_pos: Optional[QPointF] = None
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.viewport().installEventFilter(self)
        self._update_cursor()

    def set_on_stroke_done(self, cb: Callable[[], None]) -> None:
        self._on_stroke_done = cb

    def set_on_before_edit(self, cb: Optional[Callable[[], None]]) -> None:
        self._on_before_edit = cb

    def set_image(self, qimg: QImage, base_image_np: Optional[np.ndarray] = None) -> None:
        pix = QPixmap.fromImage(qimg)
        if self._base_item is None:
            self._base_item = QGraphicsPixmapItem(pix)
            self._base_item.setZValue(0)
            self.scene().addItem(self._base_item)
        else:
            self._base_item.setPixmap(pix)
        w, h = qimg.width(), qimg.height()
        self._base_image_np = base_image_np
        self._scene_rect = QRectF(0, 0, w, h)
        self.scene().setSceneRect(self._scene_rect)
        if self._overlay_item:
            self._overlay_item.setVisible(False)
        if self._annotation_layer is not None:
            self._annotation_layer.clear()
            self._annotation_layer = get_facade().new_annotation_layer(w, h)
        else:
            self._annotation_layer = get_facade().new_annotation_layer(w, h)
        if self._annotation_item:
            self.scene().removeItem(self._annotation_item)
        self._annotation_item = AnnotationGraphicsItem(self._annotation_layer)
        self._annotation_item.setOpacity(self._annotation_opacity)
        self.scene().addItem(self._annotation_item)
        if self._brush_preview_item is None:
            self._brush_preview_item = BrushPreviewItem(
                self._brush_size, self._brush_shape, self._is_erase
            )
            self.scene().addItem(self._brush_preview_item)
        self._brush_preview_item.set_brush(self._brush_size, self._brush_shape, self._is_erase)
        self._brush_preview_item.setVisible(False)
        if self._sam2_point_preview_item is None:
            self._sam2_point_preview_item = Sam2PointPreviewItem(
                self._sam2_point_radius, self._sam2_positive_point
            )
            self.scene().addItem(self._sam2_point_preview_item)
        else:
            self._sam2_point_preview_item.set_radius(self._sam2_point_radius)
            self._sam2_point_preview_item.set_positive(self._sam2_positive_point)
        self._sam2_point_preview_item.setVisible(False)
        self.fitInView(self._scene_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def replace_base_image(self, qimg: QImage) -> bool:
        """只更换底图，保留标注层。尺寸不一致时不更换。"""
        if self._base_item is None:
            return False
        if qimg.width() != int(self._scene_rect.width()) or qimg.height() != int(self._scene_rect.height()):
            return False
        self._base_item.setPixmap(QPixmap.fromImage(qimg))
        return True

    def set_overlay(self, qimg: Optional[QImage], opacity: float = 0.5) -> None:
        if qimg is None:
            if self._overlay_item:
                self._overlay_item.setVisible(False)
            return
        pix = QPixmap.fromImage(qimg)
        if self._overlay_item is None:
            self._overlay_item = QGraphicsPixmapItem(pix)
            self._overlay_item.setZValue(5)
            self.scene().addItem(self._overlay_item)
        else:
            self._overlay_item.setPixmap(pix)
        self._overlay_item.setOpacity(opacity)
        self._overlay_item.setVisible(True)

    def set_overlay_opacity(self, opacity: float) -> None:
        """仅调整当前预测/SAM2 叠加层透明度（0～1），无 overlay 时忽略。"""
        if self._overlay_item is not None and self._overlay_item.isVisible():
            self._overlay_item.setOpacity(max(0.0, min(1.0, float(opacity))))

    def get_annotation_layer(self) -> Optional[AnnotationLayer]:
        return self._annotation_layer

    def get_base_image_np(self) -> Optional[np.ndarray]:
        return self._base_image_np

    def set_annotation_layer(self, layer: AnnotationLayer) -> None:
        self._annotation_layer = layer
        if self._annotation_item:
            self.scene().removeItem(self._annotation_item)
        self._annotation_item = AnnotationGraphicsItem(layer)
        self._annotation_item.setOpacity(self._annotation_opacity)
        self.scene().addItem(self._annotation_item)
        if self._brush_preview_item is None:
            self._brush_preview_item = BrushPreviewItem(
                self._brush_size, self._brush_shape, self._is_erase
            )
            self.scene().addItem(self._brush_preview_item)
            self._brush_preview_item.setVisible(False)
        if self._sam2_point_preview_item is None:
            self._sam2_point_preview_item = Sam2PointPreviewItem(
                self._sam2_point_radius, self._sam2_positive_point
            )
            self.scene().addItem(self._sam2_point_preview_item)
            self._sam2_point_preview_item.setVisible(False)

    def refresh_annotation_display(self) -> None:
        """标注数据在层内被修改（如加载 mask）后调用，强制重绘标注图元。"""
        if self._annotation_item is not None:
            self._annotation_item.update(self._scene_rect)

    def set_annotation_opacity(self, opacity: float) -> None:
        """标注层透明度 0.0～1.0。"""
        self._annotation_opacity = max(0.0, min(1.0, opacity))
        if self._annotation_item is not None:
            self._annotation_item.setOpacity(self._annotation_opacity)

    def set_annotation_visible(self, visible: bool) -> None:
        """显示或隐藏标注层。"""
        if self._annotation_item is not None:
            self._annotation_item.setVisible(visible)
            self.annotationVisibilityChanged.emit(visible)

    def is_annotation_visible(self) -> bool:
        """标注层是否可见。"""
        return self._annotation_item.isVisible() if self._annotation_item is not None else True

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, min(200, size))
        self._update_cursor()
        self._update_brush_preview()

    def set_brush_shape(self, shape: BrushShape) -> None:
        self._brush_shape = shape
        self._update_cursor()
        self._update_brush_preview()

    def set_erase(self, erase: bool) -> None:
        self._is_erase = erase
        self._update_cursor()
        self._update_brush_preview()

    def set_fill(self, fill: bool) -> None:
        self._is_fill = fill
        self._update_cursor()
        self._update_brush_preview()

    def set_otsu_fill(self, on: bool) -> None:
        self._is_otsu_fill = on
        self._update_cursor()
        self._update_brush_preview()

    def set_otsu_threshold(self, value: int) -> None:
        self._otsu_threshold = max(0, min(255, value))

    def set_region_grow_fill(self, on: bool) -> None:
        self._is_region_grow_fill = on
        self._update_cursor()
        self._update_brush_preview()

    def set_region_grow_tolerance(self, value: int) -> None:
        self._region_grow_tolerance = max(0, min(255, value))

    def set_magic_wand(self, value: bool) -> None:
        self._is_magic_wand = value
        self._update_cursor()

    def set_sam2_mode(self, active: bool) -> None:
        self._is_sam2_mode = active
        if active:
            self._is_magic_wand = False
        self._update_cursor()
        self._update_brush_preview()

    def set_sam2_positive_point(self, positive: bool) -> None:
        self._sam2_positive_point = bool(positive)
        if self._sam2_point_preview_item is not None:
            self._sam2_point_preview_item.set_positive(self._sam2_positive_point)
            self._sam2_point_preview_item.update()
        self._update_brush_preview()

    def set_sam2_point_radius(self, radius: int) -> None:
        r = max(2, min(40, int(radius)))
        if r == self._sam2_point_radius:
            return
        self._sam2_point_radius = r
        if self._sam2_point_preview_item is not None:
            self._sam2_point_preview_item.set_radius(self._sam2_point_radius)
        self._update_brush_preview()

    def get_sam2_point_radius(self) -> int:
        return self._sam2_point_radius

    def set_fill_tolerance(self, tolerance: int) -> None:
        self._fill_tolerance = max(0, min(255, tolerance))

    def _update_cursor(self) -> None:
        """更新光标；笔刷/橡皮的圆由 BrushPreviewItem 绘制，此处仅设十字中心光标。"""
        if self._is_fill or self._is_otsu_fill or self._is_region_grow_fill:
            self.setCursor(_make_fill_cursor())
            return
        if self._is_sam2_mode:
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if self._is_magic_wand:
            self.unsetCursor()
            return
        if not self._is_fill and not self._is_otsu_fill and not self._is_region_grow_fill and not self._is_magic_wand:
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        self.unsetCursor()

    def _update_brush_preview(self) -> None:
        """更新笔刷预览 overlay：中心=鼠标位置，半径=笔刷大小；超出 viewport 由渲染裁剪。"""
        if self._brush_preview_item is None:
            return
        show_brush = (
            not self._is_fill
            and not self._is_otsu_fill
            and not self._is_region_grow_fill
            and not self._is_magic_wand
            and not self._is_sam2_mode
            and self._annotation_layer is not None
            and self._mouse_pos is not None
        )
        self._brush_preview_item.setVisible(show_brush)
        if show_brush:
            self._brush_preview_item.set_brush(
                self._brush_size, self._brush_shape, self._is_erase
            )
            inv, ok = self.viewportTransform().inverted()
            pt = inv.map(QPointF(self._mouse_pos.x(), self._mouse_pos.y()))
            self._brush_preview_item.setPos(pt)

        show_sam2 = (
            self._is_sam2_mode
            and self._annotation_layer is not None
            and self._mouse_pos is not None
        )
        if self._sam2_point_preview_item is not None:
            self._sam2_point_preview_item.setVisible(show_sam2)
            if show_sam2:
                self._sam2_point_preview_item.set_radius(self._sam2_point_radius)
                self._sam2_point_preview_item.set_positive(self._sam2_positive_point)
                inv, ok = self.viewportTransform().inverted()
                pt = inv.map(QPointF(self._mouse_pos.x(), self._mouse_pos.y()))
                self._sam2_point_preview_item.setPos(pt)

    def _scene_to_image(self, pos: QPointF) -> Tuple[int, int]:
        # 使用浮点精度映射，避免单像素点击偏移
        inv, _ = self.viewportTransform().inverted()
        pt = inv.map(QPointF(pos.x(), pos.y()))
        # 像素 (i,j) 中心在场景 (i+0.5, j+0.5)，取最近像素
        return int(round(pt.x() - 0.5)), int(round(pt.y() - 0.5))

    def _scene_to_brush_center(self, pos: QPointF) -> Tuple[int, int]:
        """将视图坐标转为图像坐标，作为笔刷/橡皮中心（与光标圆心一致）。"""
        return self._scene_to_image(pos)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            step = max(1, abs(delta) // 30)
            if self._is_sam2_mode:
                nr = (
                    min(40, self._sam2_point_radius + step)
                    if delta > 0
                    else max(2, self._sam2_point_radius - step)
                )
                if nr != self._sam2_point_radius:
                    self._sam2_point_radius = nr
                    if self._sam2_point_preview_item is not None:
                        self._sam2_point_preview_item.set_radius(self._sam2_point_radius)
                    self._update_brush_preview()
                    self.sam2PointRadiusChanged.emit(self._sam2_point_radius)
                event.accept()
                return
            new_size = min(200, self._brush_size + step) if delta > 0 else max(1, self._brush_size - step)
            if new_size != self._brush_size:
                self._brush_size = new_size
                self._update_cursor()
                self._update_brush_preview()
                self.brushSizeChanged.emit(self._brush_size)
            event.accept()
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self.scale(1.15, 1.15)
        else:
            self.scale(1 / 1.15, 1 / 1.15)
        self._update_cursor()
        self._update_brush_preview()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self._mouse_pos = None
        self._update_brush_preview()

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self._mouse_pos = QPointF(self.mapFromGlobal(QCursor.pos()))
        self._update_brush_preview()

    def eventFilter(self, obj, event: QEvent) -> bool:
        if obj is self.viewport() and event.type() == QEvent.Type.MouseMove:
            if isinstance(event, QMouseEvent):
                self._mouse_pos = event.position()
                self._update_brush_preview()
        return False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._last_pos = event.position()
            return
        if event.button() == Qt.MouseButton.RightButton:
            if self._is_sam2_mode and self._annotation_layer is not None:
                x, y = self._scene_to_image(event.position())
                self.sam2RemoveNearestRequested.emit(x, y)
                event.accept()
                return
            if self._annotation_item is not None:
                self._annotation_item.setVisible(not self._annotation_item.isVisible())
                self.annotationVisibilityChanged.emit(self._annotation_item.isVisible())
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._annotation_layer is not None:
            if self._is_sam2_mode:
                x, y = self._scene_to_image(event.position())
                self.sam2PromptClicked.emit(x, y, self._sam2_positive_point)
                return
            if self._is_magic_wand:
                x, y = self._scene_to_image(event.position())
                self.magicWandClicked.emit(x, y)
                return
            if self._is_fill:
                x, y = self._scene_to_image(event.position())
                if self._on_before_edit:
                    self._on_before_edit()
                self._annotation_layer.fill_at(
                    x, y, self._is_erase,
                    base_image=self._base_image_np,
                    tolerance=self._fill_tolerance,
                )
                self._annotation_item.update(self._scene_rect)
                if self._on_stroke_done:
                    self._on_stroke_done()
            elif self._is_otsu_fill:
                x, y = self._scene_to_image(event.position())
                if self._on_before_edit:
                    self._on_before_edit()
                self._annotation_layer.fill_at_otsu(
                    x, y, self._is_erase,
                    base_image=self._base_image_np,
                    threshold=self._otsu_threshold,
                )
                self._annotation_item.update(self._scene_rect)
                if self._on_stroke_done:
                    self._on_stroke_done()
            elif self._is_region_grow_fill:
                x, y = self._scene_to_image(event.position())
                if self._on_before_edit:
                    self._on_before_edit()
                self._annotation_layer.fill_at_region_growing(
                    x, y, self._is_erase,
                    base_image=self._base_image_np,
                    tolerance=self._region_grow_tolerance,
                )
                self._annotation_item.update(self._scene_rect)
                if self._on_stroke_done:
                    self._on_stroke_done()
            else:
                x, y = self._scene_to_brush_center(event.position())
                if self._on_before_edit:
                    self._on_before_edit()
                self._annotation_layer.single_stamp(
                    x, y, self._brush_size, self._brush_shape, self._is_erase
                )
                if self._on_stroke_done:
                    self._on_stroke_done()
                r = self._brush_size // 2 + 1
                self._annotation_item.update(QRectF(max(0, x - r), max(0, y - r), 2 * r + 2, 2 * r + 2))
            self._last_pos = event.position()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.MiddleButton:
            if self._last_pos is not None:
                delta = event.position() - self._last_pos
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
                self._last_pos = event.position()
            return
        if event.buttons() & Qt.MouseButton.LeftButton and not self._is_fill and not self._is_otsu_fill and not self._is_region_grow_fill and not self._is_sam2_mode and self._annotation_layer is not None and self._last_pos is not None:
            x0, y0 = self._scene_to_brush_center(self._last_pos)
            x1, y1 = self._scene_to_brush_center(event.position())
            self._annotation_layer.stroke(
                (x0, y0), (x1, y1), self._brush_size, self._brush_shape, self._is_erase
            )
            self._last_pos = event.position()
            r = self._brush_size // 2 + 1
            xmin = max(0, min(x0, x1) - r)
            ymin = max(0, min(y0, y1) - r)
            xmax = min(self._annotation_layer.width, max(x0, x1) + r + 1)
            ymax = min(self._annotation_layer.height, max(y0, y1) + r + 1)
            self._annotation_item.update(QRectF(xmin, ymin, xmax - xmin, ymax - ymin))
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        self._mouse_pos = event.position()
        self._update_brush_preview()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._last_pos = None
        if event.button() == Qt.MouseButton.LeftButton:
            if self._last_pos is not None and self._on_stroke_done is not None:
                self._on_stroke_done()
            self._last_pos = None
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_cursor()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._update_cursor()

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        # 分辨率/DPI/屏幕变化时刷新光标
        refresh_types = {
            QEvent.Type.ApplicationFontChange,
            QEvent.Type.ApplicationPaletteChange,
        }
        if hasattr(QEvent.Type, "DevicePixelRatioChange"):
            refresh_types.add(QEvent.Type.DevicePixelRatioChange)
        if event.type() in refresh_types:
            self._update_cursor()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Left:
            self.prevImageRequested.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Right:
            self.nextImageRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)
