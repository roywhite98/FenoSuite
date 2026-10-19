"""Image-coordinate vector canvas with screen-space picking and node dragging."""
from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsScene, QGraphicsView

from plant_phenotype.root_phenotype.facade import get_facade


class RootVectorCanvas(QGraphicsView):
    baseClicked = pyqtSignal(float, float)
    rootSelected = pyqtSignal(str)
    nodeSelected = pyqtSignal(str, str)
    nodeMoved = pyqtSignal(str, str, float, float)
    junctionSelected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor("#17202d"))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMinimumSize(280, 240)
        self.document = None
        self.selected_root_id = None
        self.selected_node_id = None
        self.mode = "base"
        self.show_diameters = False
        self._base_item = None
        self._overlays = []
        self._marks = []
        self._drag = None
        self._drag_marker = None
        self._drag_path = None
        self._pan = None
        self._image_rect = QRectF(0, 0, 640, 480)
        self._junction_id = None
        self._candidate_pairs = []
        self._last_pick = None
        self.setSceneRect(self._image_rect)

    def set_background(self, path=None, width=0, height=0):
        if self._base_item is not None:
            self.scene().removeItem(self._base_item)
            self._base_item = None
        pixmap = QPixmap(str(path)) if path else QPixmap()
        if not pixmap.isNull():
            self._base_item = self.scene().addPixmap(pixmap)
            self._base_item.setZValue(-10)
            width, height = pixmap.width(), pixmap.height()
        self._image_rect = QRectF(0, 0, max(1, width or 640), max(1, height or 480))
        self.setSceneRect(self._image_rect)
        self.fit_image()
        return not pixmap.isNull()

    def set_background_array(self, image):
        """用数组作底图。image 为 HxW 或 HxWx3。"""
        import numpy as np
        from PyQt6.QtGui import QImage

        arr = np.ascontiguousarray(image)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            arr = np.ascontiguousarray(arr)
        h, w = arr.shape[:2]
        qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        if self._base_item is not None:
            self.scene().removeItem(self._base_item)
            self._base_item = None
        pixmap = QPixmap.fromImage(qimg)
        if not pixmap.isNull():
            self._base_item = self.scene().addPixmap(pixmap)
            self._base_item.setZValue(-10)
        self._image_rect = QRectF(0, 0, max(1, w), max(1, h))
        self.setSceneRect(self._image_rect)
        self.fit_image()
        return not pixmap.isNull()

    def fit_image(self):
        self.fitInView(self._image_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def set_mode(self, mode):
        self.mode = mode
        self.setCursor(Qt.CursorShape.CrossCursor if mode in ("base", "nodes") else Qt.CursorShape.ArrowCursor)
        self._redraw()

    def set_document(self, document):
        self.document = document
        ids = {r.id for r in document.roots} if document else set()
        if self.selected_root_id not in ids:
            self.selected_root_id = None
            self.selected_node_id = None
        if document is not None and self._base_item is None:
            self._image_rect = QRectF(0, 0, max(1, document.source.width), max(1, document.source.height))
            self.setSceneRect(self._image_rect)
        self._redraw()

    def set_base_marks(self, marks):
        self._marks = marks
        self._redraw()

    def select_root(self, root_id):
        self.selected_root_id = root_id
        self.selected_node_id = None
        self._redraw()

    def highlight_junction(self, junction_id, pairs=None):
        self._junction_id = junction_id
        self._candidate_pairs = pairs or []
        self._redraw()
        if self.document is not None and self.mode == 'junction':
            region = next((j for j in self.document.analysis_snapshot.junctions if j.id == junction_id), None)
            if region is not None:
                self.centerOn(region.x, region.y)

    def _add(self, item):
        self._overlays.append(item)
        return item

    def _dot(self, x, y, color, radius=4, outline=None):
        item = self.scene().addEllipse(-radius, -radius, 2*radius, 2*radius,
            QPen(QColor(outline or color), 1), QColor(color))
        item.setPos(x, y)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        item.setZValue(5)
        return self._add(item)

    def _label(self, x, y, text, color):
        item = self.scene().addSimpleText(text)
        item.setBrush(QColor(color))
        item.setPos(x, y)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        item.setZValue(6)
        return self._add(item)

    def port_display_position(self, port):
        """Place the label outside the shared junction pixel, along its branch."""
        segment = next((s for s in self.document.analysis_snapshot.segments if s.id == port.segment_id), None)
        if segment is None or not segment.nodes:
            return QPointF(port.x, port.y)
        nodes = segment.nodes if segment.start_port == port.id else list(reversed(segment.nodes))
        target = max(28.0 / self._scale(), 2 * (port.diameter_px or 1.0))
        # Do not place a port label at a neighbouring junction on a short segment.
        length = sum(math.hypot(b.x-a.x, b.y-a.y) for a,b in zip(nodes, nodes[1:]))
        target = min(target, length * .45)
        remaining = target
        for a, b in zip(nodes, nodes[1:]):
            step = math.hypot(b.x-a.x, b.y-a.y)
            if step > 0 and remaining <= step:
                t = remaining / step
                return QPointF(a.x+t*(b.x-a.x), a.y+t*(b.y-a.y))
            remaining -= step
        return QPointF(nodes[-1].x, nodes[-1].y)

    def _redraw(self):
        for item in self._overlays:
            self.scene().removeItem(item)
        self._overlays.clear()
        doc = self.document
        if doc is not None:
            for root in doc.roots:
                color = get_facade().vector_root_color(root.id, doc)
                selected = root.id == self.selected_root_id
                if self.show_diameters:
                    for a, b in zip(root.nodes, root.nodes[1:]):
                        if a.diameter_px is not None and b.diameter_px is not None:
                            c = QColor(color)
                            c.setAlpha(50)
                            pen = QPen(c, (a.diameter_px+b.diameter_px)/2)
                            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                            self._add(self.scene().addLine(a.x, a.y, b.x, b.y, pen))
                path = QPainterPath()
                if not root.nodes:
                    continue
                path.moveTo(root.nodes[0].x, root.nodes[0].y)
                for node in root.nodes[1:]:
                    path.lineTo(node.x, node.y)
                order = doc.root_order(root.id)
                pen = QPen(QColor(color), 5 if selected else (3.5 if order == 0 else 2.5))
                pen.setCosmetic(True)
                if root.direction == "unknown" or root.review_state == "pending":
                    pen.setStyle(Qt.PenStyle.DashLine)
                self._add(self.scene().addPath(path, pen)).setZValue(2)
                if selected and self.mode == "nodes":
                    for node in root.nodes:
                        self._dot(node.x, node.y, "#ffffff" if node.id != self.selected_node_id else "#ffd34d", 4, color)
                if root.parent_attachment is not None:
                    self._dot(root.nodes[0].x, root.nodes[0].y, color, 3, "#ffffff")
            states = {d.id: d.state for d in doc.junction_decisions}
            for j in doc.analysis_snapshot.junctions:
                chosen = j.id == self._junction_id and self.mode == 'junction'
                if states.get(j.id) != "accepted" or chosen:
                    self._dot(j.x, j.y, "#ffab32", 7 if chosen else 5, "#000000")
                    self._label(j.x, j.y, " ?" if not chosen else " 交汇", "#ffab32")
            ports = {p.id: p for p in doc.analysis_snapshot.ports}
            active = next((j for j in doc.analysis_snapshot.junctions if j.id == self._junction_id), None) if self.mode == 'junction' else None
            if active is not None:
                for i, pid in enumerate(active.port_ids):
                    port = ports.get(pid)
                    if port is not None:
                        position = self.port_display_position(port)
                        self._dot(position.x(), position.y(), "#f4e988", 3)
                        self._label(position.x(), position.y(), f" {i+1}", "#f4e988")
                for i, pair in enumerate(self._candidate_pairs):
                    if len(pair) == 2 and all(p in ports for p in pair):
                        a, b = (ports[p] for p in pair)
                        pen = QPen(QColor(("#f5ff42", "#52ffff", "#ff91fc")[i % 3]), 4)
                        pen.setCosmetic(True)
                        pen.setStyle(Qt.PenStyle.DashLine)
                        pa, pb = self.port_display_position(a), self.port_display_position(b)
                        path = QPainterPath(pa)
                        path.lineTo(a.x, a.y)
                        path.lineTo(b.x, b.y)
                        path.lineTo(pb)
                        self._add(self.scene().addPath(path, pen)).setZValue(7)
        for i, mark in enumerate(self._marks):
            x = mark["x"] if isinstance(mark, dict) else mark.x
            y = mark["y"] if isinstance(mark, dict) else mark.y
            self._dot(x, y, "#43e9b4", 6, "#05281b")
            self._label(x, y, f" 选点{i+1}", "#43e9b4")

    def _scale(self):
        return max(1e-9, abs(self.transform().m11()))

    def _nearest_node(self, point):
        if self.document is None or self.selected_root_id is None:
            return None
        root = self.document.root_by_id(self.selected_root_id)
        choices = [(math.hypot(n.x-point.x(), n.y-point.y()) * self._scale(), n) for n in root.nodes]
        if not choices:
            return None
        distance, node = min(choices, key=lambda x: x[0])
        return node if distance <= 10 else None

    def pick_roots(self, point, tolerance=8):
        """Return all coincident root hits; repeated clicks cycle through them."""
        hits = []
        if self.document is None:
            return hits
        for root in self.document.roots:
            closest = float("inf")
            for a, b in zip(root.nodes, root.nodes[1:]):
                dx, dy = b.x-a.x, b.y-a.y
                denom = dx*dx+dy*dy
                t = max(0.0, min(1.0, ((point.x()-a.x)*dx+(point.y()-a.y)*dy)/denom)) if denom else 0
                closest = min(closest, math.hypot(point.x()-a.x-t*dx, point.y()-a.y-t*dy))
            if closest*self._scale() <= tolerance:
                hits.append((closest, root.id))
        return [rid for _, rid in sorted(hits)]

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._pan = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        point = self.mapToScene(event.position().toPoint())
        if not self._image_rect.contains(point):
            return
        if self.mode == "base":
            self.baseClicked.emit(point.x(), point.y())
            return
        if self.mode == "nodes":
            node = self._nearest_node(point)
            if node is not None:
                self.selected_node_id = node.id
                self.nodeSelected.emit(self.selected_root_id, node.id)
                self._drag = (self.selected_root_id, node.id, point)
                self._redraw()
                self._drag_marker = self._dot(node.x, node.y, "#ffd34d", 5)
                pen = QPen(QColor("#ffd34d"), 2)
                pen.setCosmetic(True)
                pen.setStyle(Qt.PenStyle.DashLine)
                self._drag_path = self._add(self.scene().addPath(QPainterPath(), pen))
                self._drag_path.setZValue(8)
                return
        if self.mode == "junction" and self.document is not None:
            regions = self.document.analysis_snapshot.junctions
            if regions:
                j = min(regions, key=lambda j: math.hypot(j.x-point.x(), j.y-point.y()))
                if math.hypot(j.x-point.x(), j.y-point.y())*self._scale() < 16:
                    self.junctionSelected.emit(j.id)
                    return
        hits = self.pick_roots(point)
        if hits:
            repeated = self._last_pick is not None and math.hypot(self._last_pick.x()-point.x(), self._last_pick.y()-point.y())*self._scale() < 4
            index = (hits.index(self.selected_root_id)+1) % len(hits) if repeated and self.selected_root_id in hits else 0
            self._last_pick = point
            self.select_root(hits[index])
            self.rootSelected.emit(hits[index])
        event.accept()

    def mouseMoveEvent(self, event):
        if self._pan is not None:
            pos = event.position().toPoint()
            delta = pos-self._pan
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()-delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value()-delta.y())
            self._pan = pos
            event.accept()
            return
        if self._drag is not None and self.document is not None:
            root_id, node_id, _ = self._drag
            point = self.mapToScene(event.position().toPoint())
            x = max(0, min(point.x(), self._image_rect.width()-1))
            y = max(0, min(point.y(), self._image_rect.height()-1))
            self._drag_marker.setPos(x, y)
            root = self.document.root_by_id(root_id)
            index = next(i for i, n in enumerate(root.nodes) if n.id == node_id)
            path = QPainterPath()
            if index > 0:
                path.moveTo(root.nodes[index-1].x, root.nodes[index-1].y)
                path.lineTo(x, y)
            else:
                path.moveTo(x, y)
            if index+1 < len(root.nodes):
                path.lineTo(root.nodes[index+1].x, root.nodes[index+1].y)
            self._drag_path.setPath(path)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pan is not None:
            self._pan = None
            self.set_mode(self.mode)
            return
        if self._drag is not None:
            root_id, node_id, start = self._drag
            self._drag = None
            self._drag_marker = None
            self._drag_path = None
            self._redraw()
            point = self.mapToScene(event.position().toPoint())
            if math.hypot(point.x()-start.x(), point.y()-start.y())*self._scale() >= 2:
                x = max(0, min(point.x(), self._image_rect.width()-1))
                y = max(0, min(point.y(), self._image_rect.height()-1))
                self.nodeMoved.emit(root_id, node_id, x, y)
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        if .005 <= self._scale()*factor <= 200:
            self.scale(factor, factor)
            if self._junction_id is not None:
                self._redraw()
        event.accept()

    def export_png(self, path):
        width, height = math.ceil(self._image_rect.width()), math.ceil(self._image_rect.height())
        if width * height > 100_000_000:
            raise ValueError("图像超过 1 亿像素，请使用 SVG 导出")
        image = QImage(width, height, QImage.Format.Format_ARGB32)
        if image.isNull():
            raise ValueError("无法分配导出图像")
        image.fill(QColor("#17202d"))
        selection = (self.selected_root_id, self.selected_node_id, self._junction_id, self._candidate_pairs)
        self.selected_root_id = self.selected_node_id = self._junction_id = None
        self._candidate_pairs = []
        self._redraw()
        painter = QPainter(image)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.scene().render(painter, QRectF(0, 0, width, height), self._image_rect)
        finally:
            painter.end()
            self.selected_root_id, self.selected_node_id, self._junction_id, self._candidate_pairs = selection
            self._redraw()
        if not image.save(str(path), "PNG"):
            raise ValueError("PNG 保存失败")
