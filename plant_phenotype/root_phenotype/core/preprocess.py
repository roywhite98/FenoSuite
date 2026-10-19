# -*- coding: utf-8 -*-
"""去噪与小连通域剔除、连通域提取（来自 rootmask）。"""
import cv2
import numpy as np
from typing import Tuple, Optional


def get_component_at_pixel(
    mask: np.ndarray,
    row: int,
    col: int,
) -> Optional[np.ndarray]:
    """
    取包含像素 (row, col) 的 8 连通域的二值 mask，与原图同尺寸。
    """
    if mask.size == 0:
        return None
    h, w = mask.shape
    if not (0 <= row < h and 0 <= col < w):
        return None
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    label_at = labels[row, col]
    if label_at == 0:
        return None
    out = np.zeros_like(binary, dtype=np.uint8)
    out[labels == label_at] = 255
    return out


def denoise_and_remove_small_components(
    mask: np.ndarray,
    morph_kernel_size: int = 3,
    min_component_area: int = 50,
    morph_open: bool = False,
) -> np.ndarray:
    """
    形态学去噪后剔除小连通域。
    """
    if mask.dtype != np.uint8:
        mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
    else:
        mask = (mask > 0).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if morph_open:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    out = np.zeros_like(cleaned)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            out[labels == i] = 255
    return out
