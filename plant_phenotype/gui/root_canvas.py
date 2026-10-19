# -*- coding: utf-8 -*-
"""根系画布：预览/叠加图、点选根像素、描绘主根轨迹；滚轮缩放、中键拖拽。"""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QPointF, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QWheelEvent, QPen, QColor, QKeyEvent, QFont, QCursor
from PyQt6.QtWidgets import QApplication

from plant_phenotype.root_phenotype.facade import get_facade


class RootCanvas(QLabel):
    """显示 mask/原图预览或分析叠加图；左键点击选根，拖拽描绘主根轨迹；← → 切换图像。"""
    rootPixelClicked = pyqtSignal(int, int)  # (row, col) 图像坐标
    imagePixelClicked = pyqtSignal(int, int)  # (row, col) 任意点击
    prevImageRequested = pyqtSignal()
    nextImageRequested = pyqtSignal()
    # 主根校正：默认松手即提交；按住 Shift 可多段，再 Ctrl+F 确认；无效时发 strokeConfirmInvalid
    strokeConfirm = pyqtSignal(object)  # List[Tuple[int,int]] 图像坐标
    strokeConfirmInvalid = pyqtSignal()
    strokeSegmentsChanged = pyqtSignal(int)  # 已完成段数（Shift 多段时）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._result = None
        self._preview_image = None
        self._display_base = None
        self._current_pixmap = None
        self._image_size = (0, 0)  # (width, height)
        self._zoom_scale = 1.0
        self._pan_delta = QPointF(0, 0)
        self._middle_drag_start = None
        self._stroke_mode = False
        self._stroke_points: list = []
        self._stroke_last_add = None
        self._stroke_segments: list = []  # 多段描绘，每段为 list[(row,col)]
        self._display_options = {
            "show_skeleton": True,
            "show_junctions": True,
            "show_endpoints": True,
            "show_main": True,
            "show_secondary": True,
        }
        # 用户点选「主根/根位置」的像素标记 (row, col)，叠在图像之上
        self._seed_markers: list = []
        # 描绘主根时：光标贴边自动平移；按住不动时定时器持续平移
        self._stroke_edge_timer = QTimer(self)
        self._stroke_edge_timer.setInterval(55)
        self._stroke_edge_timer.timeout.connect(self._on_stroke_edge_pan_timer)
        self._stroke_edge_margin = 40
        # 控件像素/次；宜小，否则贴边拖拽 + 定时器叠加会显得过快
        self._stroke_edge_step = 5.0

    def set_seed_markers(self, seeds: list | None) -> None:
        """显示每次点选根的位置（图像坐标 row, col），便于用户确认点击。"""
        self._seed_markers = list(seeds) if seeds else []
        self.update()

    def set_display_options(self, opts: dict) -> None:
        if opts:
            self._display_options.update(opts)
        self._refresh_image()

    def set_display_base(self, image: np.ndarray | None) -> None:
        """分析叠加图的底图。为 None 时使用分析结果自带的图像。"""
        self._display_base = np.asarray(image).copy() if image is not None and np.asarray(image).size > 0 else None
        self._refresh_image()

    def set_preview(self, image: np.ndarray | None) -> None:
        """未分析时显示选中的原图或 mask。image: RGB (H,W,3) 或灰度 (H,W)。"""
        self._preview_image = np.asarray(image).copy() if image is not None and image.size > 0 else None
        self._refresh_image()

    def set_result(self, result) -> None:
        self._result = result
        self._stroke_points = []
        self._stroke_segments = []
        self._refresh_image()

    def set_stroke_mode(self, on: bool) -> None:
        """开启时左键拖拽描绘主根；默认松手即提交；按住 Shift 松手可累积多段，Ctrl+F 确认。"""
        self._stroke_mode = on
        self._stroke_points = []
        self._stroke_last_add = None
        self._stroke_segments = []
        if not on:
            self._stroke_edge_timer.stop()
        self.update()

    def _flatten_strokes_for_confirm(self) -> list:
        """将已完成段与当前未释放段（若有）合并为一条点列，供主根校正。"""
        segs = [list(s) for s in self._stroke_segments]
        if len(self._stroke_points) >= 2:
            segs.append(list(self._stroke_points))
        out: list = []
        for seg in segs:
            for p in seg:
                if not out or out[-1] != p:
                    out.append(p)
        return out

    def _emit_stroke_confirm(self, points: list) -> None:
        """提交轨迹并退出描绘模式。"""
        self.set_stroke_mode(False)
        self.strokeConfirm.emit(list(points))

    def try_confirm_stroke_editing(self) -> None:
        """Ctrl+F：多段描绘时结束会话并提交（若点数不足则发 strokeConfirmInvalid）。"""
        if not self._stroke_mode:
            return
        flat = self._flatten_strokes_for_confirm()
        if len(flat) < 2:
            self.strokeConfirmInvalid.emit()
            return
        self._emit_stroke_confirm(flat)

    def undo_last_stroke_segment(self) -> bool:
        """描绘中：清除当前笔迹或撤销上一段；非描绘模式返回 False（由 Tab 撤销上次校正）。"""
        if not self._stroke_mode:
            return False
        if self._stroke_points:
            self._stroke_points = []
            self._stroke_last_add = None
            self.update()
            return True
        if not self._stroke_segments:
            return False
        self._stroke_segments.pop()
        self.update()
        self.strokeSegmentsChanged.emit(len(self._stroke_segments))
        return True

    def _refresh_image(self) -> None:
        if self._result is not None and not getattr(self._result, "error", None) and getattr(self._result, "G", None) is not None:
            opts = dict(self._display_options)
            base = self._display_base if self._display_base is not None else getattr(self._result, "original_image", None)
            img = get_facade().render_overlay(
                self._result,
                base_image=base,
                display_options=opts,
                line_width=2,
                circle_radius=8,
                font_scale=0.5,
            )
            if img is not None and img.size > 0:
                self._image_size = (img.shape[1], img.shape[0])
                h, w = img.shape[0], img.shape[1]
                if img.ndim == 3:
                    img = np.ascontiguousarray(img)
                    qimg = QImage(img.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
                else:
                    qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
                self._current_pixmap = QPixmap.fromImage(qimg)
                self._zoom_to_fit()
                self._update_pixmap()
                return
        if self._result is not None and not getattr(self._result, "error", None) and getattr(self._result, "mask", None) is not None and getattr(self._result, "mask", None).size > 0:
            mask = self._result.mask
            if mask.ndim == 2:
                import cv2
                img = cv2.cvtColor((mask > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
            else:
                img = np.asarray(mask)
            if img.size > 0:
                h, w = img.shape[0], img.shape[1]
                img = np.ascontiguousarray(img.astype(np.uint8))
                qimg = QImage(img.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
                self._image_size = (w, h)
                self._current_pixmap = QPixmap.fromImage(qimg)
                self._zoom_to_fit()
                self._update_pixmap()
                return
        if self._preview_image is not None and self._preview_image.size > 0:
            img = np.ascontiguousarray(self._preview_image)
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 255)).astype(np.uint8)
            if img.ndim == 2:
                h, w = img.shape
                qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
            else:
                h, w, ch = img.shape
                qimg = QImage(img.data, w, h, w * ch, QImage.Format.Format_RGB888).copy()
            self._image_size = (w, h)
            self._current_pixmap = QPixmap.fromImage(qimg)
            self._zoom_to_fit()
            self._update_pixmap()
            return
        self._current_pixmap = None
        self._image_size = (0, 0)
        self.clear()

    def _zoom_to_fit(self) -> None:
        if self._image_size[0] <= 0 or self._image_size[1] <= 0:
            return
        rw, rh = self.width(), self.height()
        if rw <= 0 or rh <= 0:
            return
        w, h = self._image_size[0], self._image_size[1]
        scale = min(rw / w, rh / h)
        self._zoom_scale = max(0.2, min(10.0, scale))
        self._pan_delta = QPointF(0, 0)

    def _display_rect(self) -> tuple:
        """当前缩放与平移下，图像在 widget 中的绘制区域 (x, y, w, h)。"""
        if self._current_pixmap is None or self._image_size[0] == 0 or self._image_size[1] == 0:
            return 0, 0, 0, 0
        w, h = self._image_size[0], self._image_size[1]
        sw, sh = w * self._zoom_scale, h * self._zoom_scale
        cx, cy = self.width() / 2, self.height() / 2
        x = cx - sw / 2 + self._pan_delta.x()
        y = cy - sh / 2 + self._pan_delta.y()
        return x, y, sw, sh

    def _update_pixmap(self) -> None:
        self.update()

    def _widget_to_image(self, pos: QPoint) -> tuple:
        """控件坐标 → 图像坐标 (col, row)。"""
        if self._image_size[0] == 0 or self._image_size[1] == 0:
            return (0, 0)
        x, y, sw, sh = self._display_rect()
        if sw <= 0 or sh <= 0:
            return (0, 0)
        ix = (pos.x() - x) * self._image_size[0] / sw
        iy = (pos.y() - y) * self._image_size[1] / sh
        return (ix, iy)

    def _image_to_widget(self, col: float, row: float) -> QPointF:
        if self._image_size[0] == 0 or self._image_size[1] == 0:
            return QPointF(0, 0)
        x, y, sw, sh = self._display_rect()
        if sw <= 0 or sh <= 0:
            return QPointF(0, 0)
        wx = x + col * sw / self._image_size[0]
        wy = y + row * sh / self._image_size[1]
        return QPointF(wx, wy)

    def _widget_radius_for_seed(self) -> float:
        """约 8 个图像像素对应的控件半径，随缩放变化。"""
        x, y, sw, sh = self._display_rect()
        if self._image_size[0] <= 0:
            return 8.0
        return max(5.0, min(28.0, 8.0 * sw / self._image_size[0]))

    def _cursor_in_stroke_edge_zone(self, pos: QPoint) -> bool:
        m = self._stroke_edge_margin
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return False
        return pos.x() < m or pos.x() > w - m or pos.y() < m or pos.y() > h - m

    def _clamp_pan_delta(self) -> None:
        """仅在图像大于视口（放大）时限制平移，避免整幅图被拖没。"""
        rw, rh = self.width(), self.height()
        if rw <= 0 or rh <= 0 or self._image_size[0] <= 0:
            return
        iw, ih = self._image_size[0], self._image_size[1]
        sw, sh = iw * self._zoom_scale, ih * self._zoom_scale
        cx, cy = rw / 2, rh / 2
        pad = 16
        pdx = self._pan_delta.x()
        pdy = self._pan_delta.y()
        x0 = cx - sw / 2 + pdx
        y0 = cy - sh / 2 + pdy
        x1, y1 = x0 + sw, y0 + sh
        if sw > rw:
            if x0 > pad:
                pdx -= x0 - pad
            if x1 < rw - pad:
                pdx += (rw - pad) - x1
        if sh > rh:
            if y0 > pad:
                pdy -= y0 - pad
            if y1 < rh - pad:
                pdy += (rh - pad) - y1
        self._pan_delta.setX(pdx)
        self._pan_delta.setY(pdy)

    def _pan_for_stroke_near_edge(self, pos: QPoint) -> bool:
        """光标靠近可视区边缘时平移画布，便于继续描绘。返回是否发生了平移。"""
        m = self._stroke_edge_margin
        step = self._stroke_edge_step
        rw, rh = self.width(), self.height()
        if rw <= 0 or rh <= 0:
            return False
        dx = dy = 0.0
        if pos.x() < m:
            dx = step
        elif pos.x() > rw - m:
            dx = -step
        if pos.y() < m:
            dy = step
        elif pos.y() > rh - m:
            dy = -step
        if dx == 0 and dy == 0:
            return False
        self._pan_delta.setX(self._pan_delta.x() + dx)
        self._pan_delta.setY(self._pan_delta.y() + dy)
        self._clamp_pan_delta()
        self.update()
        return True

    def _append_stroke_point_from_widget(self, pos: QPoint) -> None:
        """将当前光标映射到图像坐标并追加轨迹点（坐标钳制在图像内）。"""
        if self._image_size[0] <= 0 or self._image_size[1] <= 0:
            return
        ix, iy = self._widget_to_image(pos)
        row = int(np.clip(iy, 0, self._image_size[1] - 1))
        col = int(np.clip(ix, 0, self._image_size[0] - 1))
        if self._stroke_last_add is None or (row - self._stroke_last_add[0]) ** 2 + (col - self._stroke_last_add[1]) ** 2 >= 9:
            self._stroke_points.append((row, col))
            self._stroke_last_add = (row, col)
            self.update()

    def _on_stroke_edge_pan_timer(self) -> None:
        """按住左键且光标停在边缘时持续平移并采样轨迹。"""
        if not self._stroke_mode or not self._stroke_points:
            self._stroke_edge_timer.stop()
            return
        if not (QApplication.mouseButtons() & Qt.MouseButton.LeftButton):
            self._stroke_edge_timer.stop()
            return
        pos = self.mapFromGlobal(QCursor.pos())
        if not self.rect().contains(pos):
            return
        if self._cursor_in_stroke_edge_zone(pos):
            self._pan_for_stroke_near_edge(pos)
            self._append_stroke_point_from_widget(pos)

    def paintEvent(self, event) -> None:
        if self._current_pixmap is None or self._current_pixmap.isNull():
            super().paintEvent(event)
            return
        x, y, sw, sh = self._display_rect()
        if sw <= 0 or sh <= 0:
            super().paintEvent(event)
            return
        scaled = self._current_pixmap.scaled(
            int(sw), int(sh),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().color(self.backgroundRole()))
        painter.drawPixmap(int(x), int(y), scaled)
        if self._stroke_mode:
            pen = QPen(QColor(255, 200, 0), 3)
            painter.setPen(pen)
            for seg in self._stroke_segments:
                for i in range(1, len(seg)):
                    r0, c0 = seg[i - 1]
                    r1, c1 = seg[i]
                    p0 = self._image_to_widget(c0, r0)
                    p1 = self._image_to_widget(c1, r1)
                    painter.drawLine(int(p0.x()), int(p0.y()), int(p1.x()), int(p1.y()))
            if self._stroke_points:
                for i in range(1, len(self._stroke_points)):
                    r0, c0 = self._stroke_points[i - 1]
                    r1, c1 = self._stroke_points[i]
                    p0 = self._image_to_widget(c0, r0)
                    p1 = self._image_to_widget(c1, r1)
                    painter.drawLine(int(p0.x()), int(p0.y()), int(p1.x()), int(p1.y()))
        if self._seed_markers:
            rad = self._widget_radius_for_seed()
            font = QFont()
            font.setPointSize(max(8, int(9 * self._zoom_scale)))
            font.setBold(True)
            painter.setFont(font)
            for idx, rc in enumerate(self._seed_markers):
                if len(rc) < 2:
                    continue
                row, col = int(rc[0]), int(rc[1])
                pt = self._image_to_widget(col, row)
                cx, cy = int(pt.x()), int(pt.y())
                # 白边 + 品红圈 + 十字，任意背景下可见
                painter.setPen(QPen(QColor(255, 255, 255), 4))
                painter.setBrush(QColor(0, 0, 0, 0))
                painter.drawEllipse(int(cx - rad - 2), int(cy - rad - 2), int((rad + 2) * 2), int((rad + 2) * 2))
                painter.setPen(QPen(QColor(255, 45, 200), 2))
                painter.drawEllipse(int(cx - rad), int(cy - rad), int(rad * 2), int(rad * 2))
                painter.drawLine(int(cx - rad - 4), cy, int(cx + rad + 4), cy)
                painter.drawLine(cx, int(cy - rad - 4), cx, int(cy + rad + 4))
                label = str(idx + 1)
                tx, ty = int(cx + rad + 4), int(cy + 4)
                painter.setPen(QPen(QColor(0, 0, 0), 3))
                painter.drawText(tx + 1, ty + 1, label)
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.drawText(tx, ty, label)
        painter.end()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._current_pixmap is None:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        old_scale = self._zoom_scale
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._zoom_scale = max(0.2, min(10.0, self._zoom_scale * factor))
        if old_scale != self._zoom_scale:
            pos = event.position()
            cx, cy = self.width() / 2, self.height() / 2
            self._pan_delta.setX(
                self._pan_delta.x() + (pos.x() - cx - self._pan_delta.x()) * (1 - self._zoom_scale / old_scale)
            )
            self._pan_delta.setY(
                self._pan_delta.y() + (pos.y() - cy - self._pan_delta.y()) * (1 - self._zoom_scale / old_scale)
            )
            self.update()
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_drag_start = event.position().toPoint()
        elif event.button() == Qt.MouseButton.LeftButton and self._stroke_mode:
            self._stroke_points = []
            self._stroke_edge_timer.stop()
            pos = event.position().toPoint()
            self._append_stroke_point_from_widget(pos)
            if self._cursor_in_stroke_edge_zone(pos):
                self._pan_for_stroke_near_edge(pos)
                self._append_stroke_point_from_widget(pos)
                self._stroke_edge_timer.start()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if self._middle_drag_start is not None:
            self._pan_delta.setX(self._pan_delta.x() + pos.x() - self._middle_drag_start.x())
            self._pan_delta.setY(self._pan_delta.y() + pos.y() - self._middle_drag_start.y())
            self._middle_drag_start = pos
            self.update()
            return
        if self._stroke_mode and self._stroke_points and (event.buttons() & Qt.MouseButton.LeftButton):
            if self._cursor_in_stroke_edge_zone(pos):
                self._pan_for_stroke_near_edge(pos)
                self._append_stroke_point_from_widget(pos)
                if not self._stroke_edge_timer.isActive():
                    self._stroke_edge_timer.start()
            else:
                self._stroke_edge_timer.stop()
                self._append_stroke_point_from_widget(pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_drag_start = None
        elif event.button() == Qt.MouseButton.LeftButton:
            if self._stroke_mode:
                self._stroke_edge_timer.stop()
                shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                if len(self._stroke_points) >= 2:
                    if shift:
                        # 可选多段：累积后 Ctrl+F 确认
                        self._stroke_segments.append(list(self._stroke_points))
                        self._stroke_points = []
                        self._stroke_last_add = None
                        self.update()
                        self.strokeSegmentsChanged.emit(len(self._stroke_segments))
                    else:
                        # 默认：松手立即提交（与 260216 strokeDrawn 一致）
                        flat = self._flatten_strokes_for_confirm()
                        self._emit_stroke_confirm(flat)
                else:
                    self._stroke_points = []
                    self._stroke_last_add = None
                    self.update()
                event.accept()
                return
            if self._image_size[0] > 0 and self._image_size[1] > 0:
                ix, iy = self._widget_to_image(event.position().toPoint())
                row, col = int(iy), int(ix)
                if 0 <= row < self._image_size[1] and 0 <= col < self._image_size[0]:
                    self.imagePixelClicked.emit(row, col)
                    if self._result is not None and not getattr(self._result, "error", None):
                        self.rootPixelClicked.emit(row, col)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._stroke_mode and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_F:
                self.try_confirm_stroke_editing()
                event.accept()
                return
            if event.key() == Qt.Key.Key_C:
                self.undo_last_stroke_segment()
                event.accept()
                return
        if event.key() == Qt.Key.Key_Left:
            self.prevImageRequested.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Right:
            self.nextImageRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)
