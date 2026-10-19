# -*- coding: utf-8 -*-
"""拓扑简化：按边重要性删除叶边（短枝 / 低 ridge 积分）；簇状根尖合并。"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .decross import node_xy

log = logging.getLogger(__name__)


def _edge_score(
    data: dict,
    score_attr: str,
    length_fallback: float,
) -> float:
    v = data.get(score_attr)
    if v is not None:
        return float(v)
    if score_attr == "ridge_integral" and data.get("length") is not None:
        return float(data["length"])
    return length_fallback


def prune_leaf_edges_by_score(
    G: nx.Graph,
    min_score: float,
    score_attr: str = "ridge_integral",
) -> nx.Graph:
    """
    迭代删除：仅当叶节点所连边 score < min_score 时删除该边及叶节点。
    不合并 degree-2（MVP）。
    """
    if G.number_of_nodes() == 0:
        return G
    H = G.copy()
    changed = True
    while changed:
        changed = False
        leaves = [n for n in H.nodes() if H.degree(n) == 1]
        for leaf in leaves:
            if H.degree(leaf) != 1:
                continue
            nb = next(iter(H.neighbors(leaf)))
            data = H[leaf][nb]
            L = float(data.get("length", 0.0) or 0.0)
            sc = _edge_score(data, score_attr, L)
            if sc < min_score:
                H.remove_node(leaf)
                changed = True
    return H


def prune_leaf_edges_by_length(
    G: nx.Graph,
    min_length: float,
) -> nx.Graph:
    """删除长度小于阈值的叶边。"""
    return prune_leaf_edges_by_score(G, min_length, score_attr="length")


def _euclidean_node(a: Any, b: Any) -> float:
    pa, pb = node_xy(a), node_xy(b)
    return float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))


def _merge_two_edge_dicts(d1: dict, d2: dict) -> dict:
    """取较长 pixels 链（或同长时 length 更大）对应的属性，并对标量取 max。"""
    p1 = d1.get("pixels") or []
    p2 = d2.get("pixels") or []
    l1, l2 = float(d1.get("length", 0) or 0), float(d2.get("length", 0) or 0)
    if len(p1) > len(p2) or (len(p1) == len(p2) and l1 >= l2):
        base = d1
    else:
        base = d2
    out = dict(base)
    out["length"] = max(l1, l2)
    out["ridge_integral"] = max(float(d1.get("ridge_integral", 0) or 0), float(d2.get("ridge_integral", 0) or 0))
    out["ridge_mean"] = max(float(d1.get("ridge_mean", 0) or 0), float(d2.get("ridge_mean", 0) or 0))
    out["weight"] = max(float(d1.get("weight", 0) or 0), float(d2.get("weight", 0) or 0))
    out["pixels"] = list(base.get("pixels") or [])
    out["diameter"] = float(base.get("diameter", 1.0))
    out["curvature"] = float(base.get("curvature", 0.0))
    return out


def _cluster_leaves_by_distance(leaves: List[Any], max_dist: float) -> List[List[Any]]:
    """单链聚类：欧氏距离 ≤ max_dist 的叶端点归为一簇。"""
    n = len(leaves)
    if n < 2:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _euclidean_node(leaves[i], leaves[j]) <= max_dist:
                union(i, j)
    buckets: Dict[int, List[Any]] = {}
    for i in range(n):
        r = find(i)
        buckets.setdefault(r, []).append(leaves[i])
    return [b for b in buckets.values() if len(b) >= 2]


def merge_clustered_endpoints(
    G: nx.Graph,
    max_dist_px: float,
    dist_transform: Optional[np.ndarray] = None,
) -> nx.Graph:
    """
    将空间上彼此接近（≤ max_dist_px）且共连同一邻接分叉的叶节点合并为一条边，
    代表点取离该分叉欧氏距离最远的叶（最外侧根尖），以消除毛刺导致的「一尖多点」。
    dist_transform 保留参数供后续扩展，当前未用于选点。
    max_dist_px <= 0 时不修改。
    """
    _ = dist_transform
    if max_dist_px <= 0 or G.number_of_nodes() == 0:
        return G
    H = G.copy()
    leaves = [n for n in H.nodes() if H.degree(n) == 1]
    clusters = _cluster_leaves_by_distance(leaves, max_dist_px)
    merged = 0
    for cluster in clusters:
        nbs = {next(iter(H.neighbors(n))) for n in cluster if n in H and H.degree(n) == 1}
        if len(nbs) != 1:
            continue
        nb = nbs.pop()
        nb_xy = np.array(node_xy(nb), dtype=float)
        best = None
        best_d = -1.0
        for e in cluster:
            if e not in H or H.degree(e) != 1:
                continue
            exy = np.array(node_xy(e), dtype=float)
            d = float(np.linalg.norm(exy - nb_xy))
            if d > best_d:
                best_d = d
                best = e
        if best is None:
            continue
        canonical = best
        for e in list(cluster):
            if e == canonical or e not in H:
                continue
            if H.degree(e) != 1:
                continue
            enb = next(iter(H.neighbors(e)))
            if enb != nb:
                continue
            d_e = H[e][enb].copy()
            H.remove_edge(e, enb)
            if H.has_node(e) and H.degree(e) == 0:
                H.remove_node(e)
            merged += 1
            if H.has_edge(canonical, nb):
                H[canonical][nb].update(_merge_two_edge_dicts(H[canonical][nb], d_e))
            else:
                H.add_edge(canonical, nb, **d_e)
        if canonical in H:
            H.nodes[canonical]["pos"] = node_xy(canonical)
    if merged:
        log.info("端点簇合并: 吸收 %d 个伪叶节点, max_dist_px=%.2f", merged, max_dist_px)
    return H
