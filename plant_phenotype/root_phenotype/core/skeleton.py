# -*- coding: utf-8 -*-
"""Zhang-Suen 骨架化（来自 rootmask）。"""
import numpy as np
from skimage.morphology import skeletonize


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """
    mask: (H, W) 二值图 0/255 或 0/1
    返回: (H, W) bool 单像素骨架
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    skel = skeletonize(binary, method="zhang")
    return np.asarray(skel, dtype=bool)
