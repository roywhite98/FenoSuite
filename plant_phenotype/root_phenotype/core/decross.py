# -*- coding: utf-8 -*-
"""伪交叉解交叉：Structure Tensor 方向 + 分叉处邻向聚类，将 X 型交叉拆为两对节点。"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
from scipy.cluster.vq import kmeans2

from . import topology as topo

log = logging.getLogger(__name__)

NodeId = Union[Tuple[int, int], Tuple[int, int, int]]


def node_xy(n: NodeId) -> Tuple[int, int]:
    """节点坐标 (row, col)，兼容 (r,c) 与 (r,c,sub)。"""
    return int(n[0]), int(n[1])


def compute_structure_tensor_orientation(
    binary: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    对二值前景计算结构张量主方向场 θ(r,c) ∈ [-π/2, π/2]（弧度）。
    使用 skimage.feature.structure_tensor。
    """
    try:
        from skimage.feature import structure_tensor
    except ImportError:
        log.warning("skimage 不可用，decross 方向场回退为全零")
        return np.zeros(binary.shape, dtype=np.float32)

    img = (binary.astype(np.float32) > 0).astype(np.float32)
    if not np.any(img):
        return np.zeros(binary.shape, dtype=np.float32)
    Axx, Axy, Ayy = structure_tensor(img, sigma=sigma, order="rc")
    # 主方向（线条切向）
    theta = 0.5 * np.arctan2(2.0 * Axy, Axx - Ayy + 1e-12)
    return theta.astype(np.float32)


def _neighbor_direction_angles(
    jr: int,
    jc: int,
    neighbors: List[Tuple[int, int]],
) -> np.ndarray:
    """每个邻接骨架像素相对分叉点的方向角（弧度），用于双角聚类。"""
    angles = []
    for nr, nc in neighbors:
        dr, dc = nr - jr, nc - jc
        angles.append(math.atan2(dr, dc))
    return np.asarray(angles, dtype=np.float64)


def _is_crossing_two_two(
    pts: np.ndarray,
    min_cluster_sep: float,
) -> Optional[np.ndarray]:
    """
    在 (cos2θ, sin2θ) 空间 k=2 聚类；若 2+2 且两簇中心分离足够则判为 crossing。
    pts: shape (4, 2)。
    """
    if pts.shape != (4, 2):
        return None
    try:
        _, labels = kmeans2(pts, 2, minit="points", iter=30)
    except Exception:
        return None
    c0 = np.sum(labels == 0)
    c1 = np.sum(labels == 1)
    if c0 != 2 or c1 != 2:
        return None
    c0m = np.mean(pts[labels == 0], axis=0)
    c1m = np.mean(pts[labels == 1], axis=0)
    if np.linalg.norm(c0m - c1m) < float(min_cluster_sep):
        return None
    return labels.astype(np.int32)


def find_crossing_junctions(
    skel: np.ndarray,
    junctions: List[Tuple[int, int]],
    theta_map: Optional[np.ndarray] = None,
    min_cluster_sep: float = 0.35,
    geom_tensor_blend: float = 0.5,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    返回 {分叉像素: 邻接像素列表对应的 cluster 标签 (0/1)}，仅包含判为 crossing 的 junction。
    特征 = (1-α)·几何邻向 + α·邻点处结构张量方向（均在 cos2θ,sin2θ 空间）。
    """
    crossings: Dict[Tuple[int, int], np.ndarray] = {}
    h, w = skel.shape[:2]
    for jr, jc in junctions:
        nbrs = topo._skeleton_neighbors(skel, jr, jc, exclude=None)
        if len(nbrs) != 4:
            continue
        angles = _neighbor_direction_angles(jr, jc, nbrs)
        t = 2.0 * angles
        pts_g = np.column_stack([np.cos(t), np.sin(t)])
        alpha = float(np.clip(geom_tensor_blend, 0.0, 1.0))
        if theta_map is not None and theta_map.size and alpha < 1.0:
            pts_t = np.zeros((4, 2), dtype=np.float64)
            for i, (nr, nc) in enumerate(nbrs):
                if 0 <= nr < h and 0 <= nc < w:
                    th = float(theta_map[nr, nc])
                else:
                    th = 0.0
                pts_t[i, 0] = math.cos(2.0 * th)
                pts_t[i, 1] = math.sin(2.0 * th)
            pts = (1.0 - alpha) * pts_g + alpha * pts_t
        else:
            pts = pts_g
        labels = _is_crossing_two_two(pts, min_cluster_sep)
        if labels is None:
            continue
        crossings[(jr, jc)] = labels
        log.debug("decross: 判定 crossing @ (%d,%d)", jr, jc)
    return crossings


def split_crossing_junctions_in_graph(
    G: nx.Graph,
    crossings: Dict[Tuple[int, int], np.ndarray],
    skel: np.ndarray,
) -> nx.Graph:
    """
    将 crossing 处的节点 (r,c) 拆为 (r,c,0) 与 (r,c,1)，按邻向聚类重连边。
    """
    if not crossings:
        return G

    neighbor_lists: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for j, labels in crossings.items():
        jr, jc = j
        nbrs = topo._skeleton_neighbors(skel, jr, jc, exclude=None)
        if len(nbrs) != 4:
            continue
        neighbor_lists[j] = nbrs

    H = G.copy()
    for j, labels in crossings.items():
        if j not in H:
            continue
        nbrs = neighbor_lists.get(j)
        if not nbrs or len(nbrs) != 4:
            continue
        j0: NodeId = (j[0], j[1], 0)
        j1: NodeId = (j[0], j[1], 1)
        nbr_to_cluster = {nbrs[i]: int(labels[i]) for i in range(4)}

        # 收集与 j 相连的边
        incident = list(H.edges(j, data=True))
        if not incident:
            continue
        H.remove_node(j)
        H.add_node(j0, pos=(j[0], j[1]))
        H.add_node(j1, pos=(j[0], j[1]))
        for u, v, data in incident:
            other = u if v == j else v
            cluster = nbr_to_cluster.get(other)
            if cluster is None:
                # 非骨架邻接（不应发生）；挂到 j0
                target = j0
            else:
                target = j0 if cluster == 0 else j1
            H.add_edge(target, other, **data)
    return H


def apply_decross(
    G: nx.Graph,
    skel: np.ndarray,
    binary_mask: np.ndarray,
    junctions: List[Tuple[int, int]],
    enabled: bool = True,
    tensor_sigma: float = 1.0,
    min_cluster_sep: float = 0.35,
    geom_tensor_blend: float = 0.5,
) -> Tuple[nx.Graph, List[Tuple[int, int]], List[NodeId], Dict[NodeId, int]]:
    """
    对图做 crossing 拆分，并更新 junctions / junction_ids（端点不变）。
    返回: G2, split_points, new_junctions, new_junction_ids
    """
    jid0 = {p: i for i, p in enumerate(junctions)}
    if not enabled or not junctions:
        return G, [], list(junctions), jid0

    theta_map = compute_structure_tensor_orientation(binary_mask, sigma=tensor_sigma)
    crossings = find_crossing_junctions(
        skel,
        junctions,
        theta_map=theta_map,
        min_cluster_sep=min_cluster_sep,
        geom_tensor_blend=geom_tensor_blend,
    )
    if not crossings:
        return G, [], list(junctions), jid0

    G2 = split_crossing_junctions_in_graph(G, crossings, skel)

    split_pts = list(crossings.keys())
    new_junctions: List[NodeId] = []
    junction_ids: Dict[NodeId, int] = {}
    idx = 0
    for p in junctions:
        if p in crossings:
            j0 = (p[0], p[1], 0)
            j1 = (p[0], p[1], 1)
            new_junctions.append(j0)
            junction_ids[j0] = idx
            idx += 1
            new_junctions.append(j1)
            junction_ids[j1] = idx
            idx += 1
        else:
            new_junctions.append(p)
            junction_ids[p] = idx
            idx += 1

    log.info("decross: 拆分 %d 个伪交叉点", len(crossings))
    return G2, split_pts, new_junctions, junction_ids
