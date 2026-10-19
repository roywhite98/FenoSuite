"""Deterministic raster evidence for vector roots, with no root identities."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import cv2
import numpy as np
from skimage.morphology import skeletonize

from .models import VectorAnalysisConfig


def load_mask(path):
    image = cv2.imdecode(np.fromfile(Path(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'无法读取根系掩码: {path}')
    return image


def mask_digest(mask):
    """Hash original normalized binary data, before configured filtering."""
    data = np.asarray(mask)
    if data.ndim == 3 and data.shape[2] in (3, 4):
        data = np.max(data[:, :, :3], axis=2)
    if data.ndim != 2 or not data.size or not np.isfinite(data).all():
        raise ValueError('无效掩码')
    binary = np.ascontiguousarray(data > 0, dtype=np.uint8)
    return sha256(str(binary.shape).encode('ascii') + binary.tobytes()).hexdigest()


@dataclass
class RasterFeatures:
    mask: np.ndarray
    skeleton: np.ndarray
    diameter: np.ndarray
    cos2: np.ndarray
    sin2: np.ndarray
    coherence: np.ndarray
    field_valid: np.ndarray
    diagnostics: list[str]


def extract_features(mask, config: VectorAnalysisConfig, check_cancel=None) -> RasterFeatures:
    data = np.asarray(mask)
    if data.ndim == 3 and data.shape[2] in (3, 4):
        data = np.max(data[:, :, :3], axis=2)
    if data.ndim != 2 or not data.size:
        raise ValueError('根系掩码必须是非空二维图像')
    if not np.isfinite(data).all():
        raise ValueError('掩码含有非有限数值')
    binary = np.ascontiguousarray(data > 0, dtype=np.uint8)
    notes = []
    if config.min_component_area:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= config.min_component_area
        keep[0] = False
        removed = int(np.count_nonzero(~keep[1:]))
        binary = np.ascontiguousarray(keep[labels], dtype=np.uint8)
        if removed:
            notes.append(f'按面积参数过滤了 {removed} 个连通域')
    if check_cancel:
        check_cancel()
    # Padding makes roots touching an image edge measurable, including full masks.
    diameter = 2.0 * cv2.distanceTransform(np.pad(binary, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    skel = skeletonize(binary.astype(bool))
    values = binary.astype(np.float32)
    gx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    xx, xy, yy = gx * gx, gx * gy, gy * gy
    cos2 = np.zeros(binary.shape, np.float32)
    sin2 = np.zeros(binary.shape, np.float32)
    coherence = np.zeros(binary.shape, np.float32)
    valid = np.zeros(binary.shape, bool)
    selected = np.zeros(binary.shape, bool)
    scales = sorted(set(config.field_scales))
    for scale in scales:
        if check_cancel:
            check_cancel()
        jxx = cv2.boxFilter(xx, -1, (scale, scale), normalize=True)
        jxy = cv2.boxFilter(xy, -1, (scale, scale), normalize=True)
        jyy = cv2.boxFilter(yy, -1, (scale, scale), normalize=True)
        energy = jxx + jyy
        separation = np.hypot(jxx - jyy, 2 * jxy)
        quality = np.divide(separation, energy, out=np.zeros_like(energy), where=energy > config.energy_min)
        # Select the first window covering both boundaries, not a fixed window
        # whose zero-gradient interior would falsely imply a horizontal axis.
        choose = (~selected) & ((scale >= diameter + 2) | (scale == scales[-1]))
        reliable = (energy > config.energy_min) & (quality >= config.coherence_min)
        # The dominant gradient eigenvector is the NORMAL. Rotating by pi/2
        # negates both doubled-angle components and gives the root tangent.
        c = np.divide(-(jxx - jyy), separation, out=np.zeros_like(energy), where=separation > config.energy_min)
        s = np.divide(-2 * jxy, separation, out=np.zeros_like(energy), where=separation > config.energy_min)
        cos2[choose], sin2[choose] = c[choose], s[choose]
        coherence[choose] = quality[choose]
        valid[choose] = reliable[choose]
        selected[choose] = True
    valid &= binary.astype(bool)
    if not binary.any():
        notes.append('掩码中没有前景根系')
    return RasterFeatures(binary, skel, diameter, cos2, sin2, coherence, valid, notes)
