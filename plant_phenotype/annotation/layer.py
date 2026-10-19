"""标注层：与图像同尺寸的 RGBA 图层，支持笔刷/橡皮/油漆桶。"""
from __future__ import annotations

import numpy as np
from collections import deque
from enum import Enum
from typing import Tuple, Optional


class BrushShape(Enum):
    CIRCLE = "circle"
    SQUARE = "square"


def _circle_mask(radius: int) -> np.ndarray:
    """半径 radius 的圆形 mask，True 为内部。"""
    d = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y <= radius * radius).astype(np.uint8)


def _square_mask(size: int) -> np.ndarray:
    """边长 size 的方形 mask。"""
    return np.ones((size, size), dtype=np.uint8)


def _otsu_threshold(gray: np.ndarray) -> int:
    """大津法求二值化阈值，gray 为 uint8 灰度图。"""
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 128
    sum_total = np.dot(np.arange(256), hist)
    best_var = 0.0
    best_t = 0
    w0 = 0.0
    sum0 = 0.0
    for t in range(256):
        w0 += hist[t]
        if w0 <= 0:
            continue
        w1 = total - w0
        if w1 <= 0:
            break
        sum0 += t * hist[t]
        u0 = sum0 / w0
        u1 = (sum_total - sum0) / w1
        var_between = w0 * w1 * (u0 - u1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_t = t
    return best_t


def otsu_threshold_for_image(image: np.ndarray) -> int:
    """根据图像（H,W）或（H,W,3）计算大津阈值，供 UI 设置阈值滑块。"""
    if image is None or image.size == 0:
        return 128
    if image.ndim == 2:
        gray = image.astype(np.uint8)
    else:
        gray = (
            image[:, :, 0].astype(np.uint32) * 299
            + image[:, :, 1].astype(np.uint32) * 587
            + image[:, :, 2].astype(np.uint32) * 114
        ) // 1000
        gray = gray.astype(np.uint8)
    return _otsu_threshold(gray)


class AnnotationLayer:
    """与图像同尺寸的标注层，未标注为透明(0)，标注处为给定颜色+不透明。"""

    def __init__(self, width: int, height: int, color: Tuple[int, int, int] = (0, 255, 0)):
        self.width = width
        self.height = height
        self.color = color
        self._data = np.zeros((height, width, 4), dtype=np.uint8)

    def clear(self) -> None:
        self._data.fill(0)

    def copy(self) -> AnnotationLayer:
        other = AnnotationLayer(self.width, self.height, self.color)
        other._data = self._data.copy()
        return other

    def get_foreground_mask(self) -> np.ndarray:
        return self._data[:, :, 3] > 0

    def get_mask_rgb(self) -> np.ndarray:
        out = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        fg = self.get_foreground_mask()
        out[fg] = self.color
        return out

    def _clamp(self, x: int, y: int) -> Tuple[int, int]:
        return (max(0, min(self.width - 1, x)), max(0, min(self.height - 1, y)))

    def _get_kernel(self, brush_size: int, shape: BrushShape) -> np.ndarray:
        radius = brush_size // 2
        if shape == BrushShape.CIRCLE:
            return _circle_mask(radius)
        return _square_mask(2 * radius + 1)

    def _apply_stamp_with_kernel(
        self, cx: int, cy: int, kernel: np.ndarray, is_erase: bool
    ) -> None:
        kh, kw = kernel.shape
        x0 = cx - kw // 2
        y0 = cy - kh // 2
        x1 = x0 + kw
        y1 = y0 + kh
        dx0 = max(0, -x0)
        dy0 = max(0, -y0)
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self.width, x1)
        y1 = min(self.height, y1)
        kx0 = dx0
        ky0 = dy0
        kx1 = kx0 + (x1 - x0)
        ky1 = ky0 + (y1 - y0)
        if x0 >= x1 or y0 >= y1:
            return
        k = kernel[ky0:ky1, kx0:kx1]
        if is_erase:
            self._data[y0:y1, x0:x1, 3] = np.where(k, 0, self._data[y0:y1, x0:x1, 3])
            self._data[y0:y1, x0:x1, :3] = np.where(
                k[..., None], 0, self._data[y0:y1, x0:x1, :3]
            )
        else:
            self._data[y0:y1, x0:x1, :3] = np.where(
                k[..., None], self.color, self._data[y0:y1, x0:x1, :3]
            )
            self._data[y0:y1, x0:x1, 3] = np.where(
                k, 255, self._data[y0:y1, x0:x1, 3]
            )

    def stroke(
        self,
        from_xy: Tuple[int, int],
        to_xy: Tuple[int, int],
        brush_size: int,
        shape: BrushShape,
        is_erase: bool,
    ) -> None:
        x0, y0 = from_xy
        x1, y1 = to_xy
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        kernel = self._get_kernel(brush_size, shape)
        for i in range(steps + 1):
            t = i / steps
            x = int(round(x0 + t * (x1 - x0)))
            y = int(round(y0 + t * (y1 - y0)))
            self._apply_stamp_with_kernel(x, y, kernel, is_erase)

    def single_stamp(
        self,
        x: int,
        y: int,
        brush_size: int,
        shape: BrushShape,
        is_erase: bool,
    ) -> None:
        kernel = self._get_kernel(brush_size, shape)
        self._apply_stamp_with_kernel(x, y, kernel, is_erase)

    def fill_at(
        self,
        x: int,
        y: int,
        is_erase: bool,
        base_image: Optional[np.ndarray] = None,
        tolerance: int = 32,
        max_pixels: int = 2_000_000,
    ) -> None:
        x, y = self._clamp(x, y)
        if base_image is None or base_image.shape[:2] != (self.height, self.width):
            return
        if base_image.ndim == 2:
            ref = np.stack([base_image.astype(np.int32)] * 3, axis=-1)
        else:
            ref = base_image[:, :, :3].astype(np.int32)
        target = ref[y, x].copy()
        visited = np.zeros((self.height, self.width), dtype=bool)
        q: deque = deque([(x, y)])
        visited[y, x] = True
        to_fill: list = [(x, y)]
        while q and len(to_fill) < max_pixels:
            cx, cy = q.popleft()
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = cx + dx, cy + dy
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue
                if visited[ny, nx]:
                    continue
                diff = np.abs(ref[ny, nx] - target)
                if np.any(diff > tolerance):
                    continue
                visited[ny, nx] = True
                to_fill.append((nx, ny))
                q.append((nx, ny))
        if not to_fill:
            return
        if is_erase:
            for cx, cy in to_fill:
                self._data[cy, cx, :] = 0
        else:
            for cx, cy in to_fill:
                self._data[cy, cx, :3] = self.color
                self._data[cy, cx, 3] = 255

    def fill_at_otsu(
        self,
        x: int,
        y: int,
        is_erase: bool,
        base_image: Optional[np.ndarray] = None,
        threshold: int = 128,
        max_pixels: int = 2_000_000,
    ) -> None:
        """基于阈值的填充：先对底图二值化（阈值为 0–255，可由大津法得到），再从点击处连通填充。"""
        x, y = self._clamp(x, y)
        threshold = max(0, min(255, threshold))
        if base_image is None or base_image.shape[:2] != (self.height, self.width):
            return
        if base_image.ndim == 2:
            gray = base_image.astype(np.uint8)
        else:
            gray = (
                base_image[:, :, 0].astype(np.uint32) * 299
                + base_image[:, :, 1].astype(np.uint32) * 587
                + base_image[:, :, 2].astype(np.uint32) * 114
            ) // 1000
            gray = gray.astype(np.uint8)
        # 与点击点同侧的像素：若点击处 >= 阈值则填 >= 阈值的连通域，否则填 <= 阈值的连通域
        click_val = gray[y, x]
        if click_val >= threshold:
            region = gray >= threshold
        else:
            region = gray <= threshold
        visited = np.zeros_like(region, dtype=bool)
        q: deque = deque([(x, y)])
        visited[y, x] = True
        to_fill: list = [(x, y)]
        while q and len(to_fill) < max_pixels:
            cx, cy = q.popleft()
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = cx + dx, cy + dy
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue
                if visited[ny, nx] or not region[ny, nx]:
                    continue
                visited[ny, nx] = True
                to_fill.append((nx, ny))
                q.append((nx, ny))
        if not to_fill:
            return
        if is_erase:
            for cx, cy in to_fill:
                self._data[cy, cx, :] = 0
        else:
            for cx, cy in to_fill:
                self._data[cy, cx, :3] = self.color
                self._data[cy, cx, 3] = 255

    def fill_at_region_growing(
        self,
        x: int,
        y: int,
        is_erase: bool,
        base_image: Optional[np.ndarray] = None,
        tolerance: int = 32,
        max_pixels: int = 2_000_000,
    ) -> None:
        """区域增长法填充：从种子点出发，将灰度与种子点差值在 tolerance 内的连通像素填入。"""
        x, y = self._clamp(x, y)
        tolerance = max(0, min(255, tolerance))
        if base_image is None or base_image.shape[:2] != (self.height, self.width):
            return
        if base_image.ndim == 2:
            gray = base_image.astype(np.int32)
        else:
            gray = (
                base_image[:, :, 0].astype(np.int32) * 299
                + base_image[:, :, 1].astype(np.int32) * 587
                + base_image[:, :, 2].astype(np.int32) * 114
            ) // 1000
        seed_val = int(gray[y, x])
        visited = np.zeros((self.height, self.width), dtype=bool)
        q: deque = deque([(x, y)])
        visited[y, x] = True
        to_fill: list = [(x, y)]
        while q and len(to_fill) < max_pixels:
            cx, cy = q.popleft()
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = cx + dx, cy + dy
                if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                    continue
                if visited[ny, nx]:
                    continue
                if abs(int(gray[ny, nx]) - seed_val) > tolerance:
                    continue
                visited[ny, nx] = True
                to_fill.append((nx, ny))
                q.append((nx, ny))
        if not to_fill:
            return
        if is_erase:
            for cx, cy in to_fill:
                self._data[cy, cx, :] = 0
        else:
            for cx, cy in to_fill:
                self._data[cy, cx, :3] = self.color
                self._data[cy, cx, 3] = 255

    def apply_mask(self, mask: np.ndarray, is_erase: bool = False) -> None:
        if mask.shape[:2] != (self.height, self.width):
            return
        m = mask.astype(bool) if mask.ndim == 2 else (mask.any(axis=-1) if mask.ndim == 3 else np.zeros((self.height, self.width), dtype=bool))
        if is_erase:
            self._data[m, 3] = 0
            self._data[m, :3] = 0
        else:
            self._data[m, :3] = self.color
            self._data[m, 3] = 255

    def replace_with(self, other: AnnotationLayer) -> None:
        if other.width != self.width or other.height != self.height:
            raise ValueError("size mismatch")
        self._data = other._data.copy()

    def get_rgba(self) -> np.ndarray:
        return self._data.copy()

    def get_data_readonly(self) -> np.ndarray:
        return self._data
