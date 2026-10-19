# -*- coding: utf-8 -*-
"""图节点空间最近邻（cKDTree）。"""
from __future__ import annotations

from typing import List, Optional, Tuple, TypeVar

import numpy as np

Node = Tuple[int, int]
T = TypeVar("T")

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None  # type: ignore


def build_kdtree(nodes: List[T]) -> Tuple[Optional[object], List[T]]:
    if not nodes:
        return None, []
    if cKDTree is None:
        return None, list(nodes)
    coords = np.array([[float(n[0]), float(n[1])] for n in nodes], dtype=np.float64)
    return cKDTree(coords), list(nodes)


def nearest_in_nodes(
    nodes: List[T],
    row: int,
    col: int,
    *,
    radius_px: Optional[float] = None,
    tree: Optional[object] = None,
) -> Optional[T]:
    if not nodes:
        return None
    q = np.array([float(row), float(col)], dtype=np.float64)
    if tree is not None and cKDTree is not None:
        if radius_px is not None and radius_px > 0:
            idxs = tree.query_ball_point(q, float(radius_px))
            if idxs:
                best_i = min(
                    idxs,
                    key=lambda i: (nodes[i][0] - row) ** 2 + (nodes[i][1] - col) ** 2,
                )
                return nodes[best_i]
        _, idx = tree.query(q)
        return nodes[int(idx)]
    best: Optional[T] = None
    best_d = 1e18
    r_sq = (radius_px * radius_px) if radius_px and radius_px > 0 else None
    for n in nodes:
        d = (n[0] - row) ** 2 + (n[1] - col) ** 2
        if r_sq is not None and d > r_sq:
            continue
        if d < best_d:
            best_d = d
            best = n
    if best is None and r_sq is not None:
        for n in nodes:
            d = (n[0] - row) ** 2 + (n[1] - col) ** 2
            if d < best_d:
                best_d = d
                best = n
    return best


def nearest_nodes_for_seeds(
    nodes: List[Node],
    seed_pixels: List[Tuple[int, int]],
) -> List[Node]:
    tree, node_list = build_kdtree(nodes)
    out: List[Node] = []
    seen: set = set()
    for r, c in seed_pixels:
        n = nearest_in_nodes(node_list, r, c, tree=tree)
        if n is not None and n not in seen:
            seen.add(n)
            out.append(n)
    return out
