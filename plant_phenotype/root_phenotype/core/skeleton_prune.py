# -*- coding: utf-8 -*-
"""骨架端部短毛刺剪枝：迭代删除从叶到分叉前长度 ≤ L 的链。"""
from __future__ import annotations

import logging
from typing import List, Optional, Set, Tuple

import numpy as np

from . import topology as topo

log = logging.getLogger(__name__)


def _find_degree1_pixels(skel: np.ndarray) -> List[Tuple[int, int]]:
    """向量化找度为 1 的叶像素。"""
    sk = np.asarray(skel, dtype=bool)
    if not np.any(sk):
        return []
    try:
        from scipy import ndimage

        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        neighbor_count = ndimage.convolve(sk.astype(np.uint8), kernel, mode="constant", cval=0)
        ys, xs = np.where(sk & (neighbor_count == 1))
        return [(int(r), int(c)) for r, c in zip(ys.tolist(), xs.tolist())]
    except ImportError:
        h, w = sk.shape
        out = []
        for r in range(h):
            for c in range(w):
                if sk[r, c] and topo._neighbor_count(sk, r, c) == 1:
                    out.append((r, c))
        return out


def _trace_spur_pixels_to_junction(
    skel: np.ndarray,
    leaf: Tuple[int, int],
    junction_min_neighbors: int = 3,
) -> Optional[List[Tuple[int, int]]]:
    """
    从度为 1 的叶像素沿唯一链走向分叉；若先遇到另一叶端则返回 None（不剪整条细线）。
    返回：分叉前的一段像素列表（不含分叉像素）。
    """
    if topo._neighbor_count(skel, leaf[0], leaf[1]) != 1:
        return None
    prev: Optional[Tuple[int, int]] = None
    cur = leaf
    path: List[Tuple[int, int]] = [leaf]
    while True:
        nbrs = topo._skeleton_neighbors(skel, cur[0], cur[1], exclude=prev)
        if len(nbrs) != 1:
            return None
        nxt = nbrs[0]
        nc = topo._neighbor_count(skel, nxt[0], nxt[1])
        if nc >= junction_min_neighbors:
            return path
        if nc == 1:
            return None
        prev = cur
        cur = nxt
        path.append(cur)


def prune_skeleton_terminal_spurs(
    skel: np.ndarray,
    max_spur_length: int,
    junction_min_neighbors: int = 3,
) -> np.ndarray:
    """
    删除从叶到最近分叉、长度（像素数）≤ max_spur_length 的毛刺链。
    每轮尽量批量删除互不重叠的短毛刺，减少稠密根毛下的迭代次数。
    max_spur_length <= 0 时不修改。
    """
    if max_spur_length <= 0:
        return np.asarray(skel, dtype=bool)
    sk = np.asarray(skel, dtype=bool).copy()
    total_removed = 0
    max_rounds = 64
    for _ in range(max_rounds):
        leaves = _find_degree1_pixels(sk)
        if not leaves:
            break
        to_remove: Set[Tuple[int, int]] = set()
        occupied: Set[Tuple[int, int]] = set()
        for leaf in leaves:
            if leaf in occupied:
                continue
            spur = _trace_spur_pixels_to_junction(sk, leaf, junction_min_neighbors)
            if spur is None or len(spur) > max_spur_length:
                continue
            if any(p in occupied for p in spur):
                continue
            for p in spur:
                occupied.add(p)
                to_remove.add(p)
        if not to_remove:
            break
        for r, c in to_remove:
            sk[r, c] = False
        total_removed += len(to_remove)
    if total_removed:
        log.info("骨架短毛刺剪枝: 删除 %d 个像素, max_spur_length=%d", total_removed, max_spur_length)
    return sk
