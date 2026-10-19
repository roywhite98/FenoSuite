# -*- coding: utf-8 -*-
"""分割预览画布与 RGB 图像转换。"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QImage, QPixmap, QWheelEvent, QPainter, QMouseEvent
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem


def numpy_rgb_to_qimage(rgb: np.ndarray) -> QImage:
    h, w = rgb.shape[:2]
    if not rgb.flags["C_CONTIGUOUS"]:
        rgb = np.ascontiguousarray(rgb)
    return QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


class PreviewCanvasView(QGraphicsView):
    """支持滚轮缩放与中键平移的预览画布。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._pix_item: QGraphicsPixmapItem | None = None
        self._last_pos: Optional[QPointF] = None

    def set_pixmap(self, pix: QPixmap) -> None:
        if pix.isNull():
            return
        if self._pix_item is None:
            self._pix_item = QGraphicsPixmapItem(pix)
            self._pix_item.setZValue(0)
            self.scene().addItem(self._pix_item)
        else:
            self._pix_item.setPixmap(pix)
        self.scene().setSceneRect(QRectF(pix.rect()))

    def clear(self) -> None:
        self.scene().clear()
        self._pix_item = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        scale = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(scale, scale)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._last_pos = event.position()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.MiddleButton:
            if self._last_pos is not None:
                delta = event.position() - self._last_pos
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
                self._last_pos = event.position()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._last_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)
