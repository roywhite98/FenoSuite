# -*- coding: utf-8 -*-
"""加权拓扑图：节点=关键点，边=根段（来自 rootmask）。"""
import numpy as np
import networkx as nx
import cv2
from typing import List, Tuple, Dict, Any, Optional

from .topology import build_topology
from . import decross
from . import ridge as ridge_mod
from . import simplify_graph


def _segment_length(pixels: List[Tuple[int, int]]) -> float:
    if len(pixels) < 2:
        return 0.0
    L = 0.0
    for i in range(1, len(pixels)):
        r0, c0 = pixels[i - 1]
        r1, c1 = pixels[i]
        L += np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
    return L


def _segment_curvature(pixels: List[Tuple[int, int]]) -> float:
    if len(pixels) < 3:
        return 0.0
    angles = []
    for i in range(1, len(pixels) - 1):
        r0, c0 = pixels[i - 1]
        r1, c1 = pixels[i]
        r2, c2 = pixels[i + 1]
        v1 = (r1 - r0, c1 - c0)
        v2 = (r2 - r1, c2 - c1)
        n1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2) or 1
        n2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2) or 1
        cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        cos = np.clip(cos, -1, 1)
        angles.append(np.arccos(cos))
    return np.mean(angles) if angles else 0.0


def _segment_avg_diameter(pixels: List[Tuple[int, int]], dist_transform: np.ndarray) -> float:
    if not pixels or dist_transform is None:
        return 1.0
    vals = [dist_transform[r, c] for r, c in pixels if 0 <= r < dist_transform.shape[0] and 0 <= c < dist_transform.shape[1]]
    if not vals:
        return 1.0
    return 2.0 * np.mean(vals)


def _infer_keypoints_from_graph(G: nx.Graph):
    """由最终图推断端点与分叉（兼容 (r,c) 与 (r,c,sub) 节点）。"""
    endpoints = [n for n in G.nodes() if G.degree(n) == 1]
    junctions = [n for n in G.nodes() if G.degree(n) >= 3]
    return endpoints, junctions


def build_weighted_graph(
    skel: np.ndarray,
    junction_min_neighbors: int = 3,
    weight_length: float = 1.0,
    weight_diameter: float = 0.5,
    weight_curvature: float = -0.2,
    root_mask: Optional[np.ndarray] = None,
    ridge_field: Optional[np.ndarray] = None,
    curvature_angle_deg: Optional[float] = None,
    decross_enabled: bool = False,
    decross_tensor_sigma: float = 1.0,
    decross_min_cluster_sep: float = 0.35,
    decross_geom_tensor_blend: float = 0.5,
    simplify_enabled: bool = False,
    simplify_min_score: float = 0.0,
    simplify_score_attr: str = "ridge_integral",
    endpoint_merge_enabled: bool = False,
    endpoint_merge_max_dist: float = 0.0,
) -> Tuple[nx.Graph, List, List, Dict, Dict]:
    endpoints, junctions, segments, endpoint_ids, junction_ids = build_topology(
        skel, junction_min_neighbors, curvature_angle_deg=curvature_angle_deg
    )
    if root_mask is not None and root_mask.size == skel.size and np.any(root_mask):
        binary = (root_mask.astype(np.uint8) != 0).astype(np.uint8) * 255
        dt = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    else:
        binary = (skel.astype(np.uint8) * 255)
        if binary.any():
            dt = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        else:
            dt = np.ones(skel.shape, dtype=np.float32)

    all_nodes = list(set(endpoints) | set(junctions))
    G = nx.Graph()
    for n in all_nodes:
        xy = decross.node_xy(n)
        G.add_node(n, pos=xy)

    for node_a, node_b, pixels in segments:
        length = _segment_length(pixels)
        diameter = _segment_avg_diameter(pixels, dt)
        curvature = _segment_curvature(pixels)
        weight = (
            weight_length * length
            + weight_diameter * diameter
            + weight_curvature * curvature
        )
        weight = max(weight, 1e-6)
        if ridge_field is not None:
            ridge_integral, ridge_mean = ridge_mod.integrate_ridge_along_polyline(ridge_field, pixels)
        else:
            ridge_integral, ridge_mean = float(length), 1.0
        G.add_edge(
            node_a,
            node_b,
            pixels=pixels,
            length=length,
            diameter=diameter,
            curvature=curvature,
            weight=weight,
            ridge_integral=float(ridge_integral),
            ridge_mean=float(ridge_mean),
        )

    if decross_enabled:
        j_list = [j for j in junctions if isinstance(j, tuple) and len(j) == 2]
        G, _split_pts, new_junctions, junction_ids = decross.apply_decross(
            G,
            skel,
            binary.astype(np.uint8),
            j_list,
            enabled=True,
            tensor_sigma=decross_tensor_sigma,
            min_cluster_sep=decross_min_cluster_sep,
            geom_tensor_blend=decross_geom_tensor_blend,
        )
        junctions = new_junctions
        for n in G.nodes():
            G.nodes[n]["pos"] = decross.node_xy(n)

    if simplify_enabled and simplify_min_score > 0:
        if simplify_score_attr == "length":
            G = simplify_graph.prune_leaf_edges_by_length(G, min_length=simplify_min_score)
        else:
            G = simplify_graph.prune_leaf_edges_by_score(
                G, min_score=simplify_min_score, score_attr=simplify_score_attr
            )

    if endpoint_merge_enabled and endpoint_merge_max_dist > 0:
        G = simplify_graph.merge_clustered_endpoints(G, endpoint_merge_max_dist, dist_transform=dt)

    if (
        decross_enabled
        or (simplify_enabled and simplify_min_score > 0)
        or (endpoint_merge_enabled and endpoint_merge_max_dist > 0)
    ):
        endpoints, junctions = _infer_keypoints_from_graph(G)
        endpoint_ids = {p: i for i, p in enumerate(endpoints)}
        junction_ids = {p: i for i, p in enumerate(junctions)}

    return G, endpoints, junctions, endpoint_ids, junction_ids
