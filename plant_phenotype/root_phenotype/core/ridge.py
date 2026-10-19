# -*- coding: utf-8 -*-
"""多尺度 ridge 强度场：用于边沿弧长的积分重要性（轻量持久化思想）。"""
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


def compute_multiscale_r_max(
    mask: np.ndarray,
    sigmas: Iterable[float] = (1.0, 2.0, 4.0),
) -> np.ndarray:
    """
    对二值前景在多尺度高斯平滑后取距离变换，再对尺度取逐像素 max，得到 R(x)。
    mask: 任意非零为前景。
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    h, w = binary.shape
    if not np.any(binary):
        return np.zeros((h, w), dtype=np.float32)
    R = np.zeros((h, w), dtype=np.float32)
    for sigma in sigmas:
        s = float(max(sigma, 0.5))
        blur = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), s)
        m = (blur > 0.5).astype(np.uint8)
        if not np.any(m):
            continue
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)
        R = np.maximum(R, dt / (s + 1e-6))
    return R


def integrate_ridge_along_polyline(
    ridge: np.ndarray,
    pixels: list,
) -> Tuple[float, float]:
    """
    沿像素折线离散 ∫ R ds：相邻像素段中点 R 乘欧氏步长。
    返回 (integral, mean_R 沿路径采样均值)。
    """
    if ridge is None or ridge.size == 0 or len(pixels) < 1:
        return 0.0, 0.0
    h, w = ridge.shape
    rs = []
    for r, c in pixels:
        if 0 <= r < h and 0 <= c < w:
            rs.append(float(ridge[r, c]))
        else:
            rs.append(0.0)
    mean = float(np.mean(rs)) if rs else 0.0
    if len(pixels) < 2:
        return mean, mean
    integral = 0.0
    for i in range(len(pixels) - 1):
        r0, c0 = pixels[i]
        r1, c1 = pixels[i + 1]
        ds = float(np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2))
        rm = 0.5 * (rs[i] + rs[i + 1])
        integral += rm * ds
    return integral, mean
