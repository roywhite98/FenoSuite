# -*- coding: utf-8 -*-
"""合成根系 mask，供 Golden 回归与脚本生成 baseline。"""
from __future__ import annotations

import numpy as np


def straight_horizontal(h: int = 48, w: int = 96, thickness: int = 5) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    r0 = h // 2 - thickness // 2
    mask[r0 : r0 + thickness, w // 8 : 7 * w // 8] = 255
    return mask


def t_junction(h: int = 64, w: int = 64, thickness: int = 5) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    mask[cy - thickness // 2 : cy + thickness // 2 + 1, cx - 20 : cx + 21] = 255
    mask[cy : h - 8, cx - thickness // 2 : cx + thickness // 2 + 1] = 255
    return mask


def y_fork(h: int = 80, w: int = 80, thickness: int = 5) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, base = w // 2, h - 10
    mask[base - 30 : base, cx - thickness // 2 : cx + thickness // 2 + 1] = 255
    for dr, dc in [(-25, -18), (-25, 18)]:
        r, c = base - 30, cx
        steps = 28
        for t in range(steps):
            rr = int(r + dr * t / steps)
            cc = int(c + dc * t / steps)
            mask[
                max(0, rr - thickness // 2) : min(h, rr + thickness // 2 + 1),
                max(0, cc - thickness // 2) : min(w, cc + thickness // 2 + 1),
            ] = 255
    return mask


def main_with_lateral(h: int = 72, w: int = 48, thickness: int = 6) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = w // 2
    mask[8:h - 8, cx - thickness // 2 : cx + thickness // 2 + 1] = 255
    mid = h // 2
    mask[mid - 2 : mid + 3, cx : cx + 18] = 255
    return mask


def two_components(h: int = 60, w: int = 100, thickness: int = 5) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 2 - thickness // 2 : h // 2 + thickness // 2 + 1, 10:40] = 255
    mask[h // 2 - thickness // 2 : h // 2 + thickness // 2 + 1, 60:90] = 255
    return mask


def broken_gap(h: int = 40, w: int = 100, thickness: int = 5, gap: int = 6) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    r0 = h // 2 - thickness // 2
    mid = w // 2
    mask[r0 : r0 + thickness, 8 : mid - gap // 2] = 255
    mask[r0 : r0 + thickness, mid + gap // 2 : w - 8] = 255
    return mask


FIXTURE_BUILDERS = {
    "straight_horizontal": straight_horizontal,
    "t_junction": t_junction,
    "y_fork": y_fork,
    "main_with_lateral": main_with_lateral,
    "two_components": two_components,
    "broken_gap": broken_gap,
}
