# -*- coding: utf-8 -*-
"""各级根统计、数据清洗、Z-score（来自 rootmask）。"""
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional


def _local_edge_direction(G: nx.Graph, node: tuple, neighbor: tuple) -> Optional[np.ndarray]:
    """取根段靠近连接处的首个非零像素步长，不使用远端节点的连线。"""
    pixels = G[node][neighbor].get("pixels")
    origin = np.asarray(node[:2], dtype=float)
    if pixels is not None and len(pixels) >= 2:
        points = np.asarray(pixels, dtype=float)[:, :2]
        if np.linalg.norm(points[-1] - origin) < np.linalg.norm(points[0] - origin):
            points = points[::-1]
        origin = points[0]
        for point in points[1:]:
            direction = point - origin
            norm = np.linalg.norm(direction)
            if norm > 0:
                return direction / norm
    direction = np.asarray(neighbor[:2], dtype=float) - np.asarray(node[:2], dtype=float)
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 0 else None


def _branch_angle_at_node(
    G: nx.Graph,
    node: tuple,
    branch_neighbor: tuple,
    primary_neighbors: List[tuple],
) -> Optional[float]:
    """二级根局部切线相对一级根无向局部轴线的锐角（弧度，0–π/2）。

    主根没有统一的生长方向，因此使用无向轴线；交点有多个一级根段时取最小夹角。
    每个一级/二级连接只统计一次，0° 是有效角度；无法确定方向返回 None。
    """
    branch_direction = _local_edge_direction(G, node, branch_neighbor)
    if branch_direction is None:
        return None
    angles = []
    for neighbor in primary_neighbors:
        primary_direction = _local_edge_direction(G, node, neighbor)
        if primary_direction is not None:
            cosine = float(np.clip(abs(np.dot(branch_direction, primary_direction)), 0.0, 1.0))
            angles.append(float(np.arccos(cosine)))
    return min(angles) if angles else None


def _main_nodes_from_edge_level(
    edge_level: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int],
) -> set:
    main_nodes = set()
    for (u, v), lev in edge_level.items():
        if lev == 1:
            main_nodes.add(u)
            main_nodes.add(v)
    return main_nodes


def _count_secondary_roots_by_tips(
    G: nx.Graph,
    edge_level: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int],
    main_nodes: set,
) -> int:
    visited = set(main_nodes)
    queue = list(main_nodes)
    while queue:
        u = queue.pop(0)
        for v in G.neighbors(u):
            key = (min(u, v), max(u, v))
            if key not in edge_level or edge_level[key] != 2:
                continue
            if v in visited:
                continue
            visited.add(v)
            queue.append(v)
    return sum(1 for n in visited if n not in main_nodes and G.degree(n) == 1)


def compute_level_stats(
    G: nx.Graph,
    edge_level: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int],
    min_root_length: float = 5.0,
    min_root_diameter: float = 1.0,
) -> pd.DataFrame:
    """分叉角只计二级根在主根上的连接；一级根无上级根，角度列为 0。"""
    main_nodes = _main_nodes_from_edge_level(edge_level)
    rows = []
    for level in [1, 2]:
        lengths = []
        diameters = []
        angles = []
        for (u, v), lev in edge_level.items():
            if lev != level:
                continue
            if not G.has_edge(u, v):
                continue
            d = G[u][v]
            L = d.get("length", 0)
            D = d.get("diameter", 0)
            if L < min_root_length or D < min_root_diameter:
                continue
            lengths.append(L)
            diameters.append(D)
            if level == 2:
                for n, branch_neighbor in ((u, v), (v, u)):
                    primary_neighbors = [
                        neighbor for neighbor in G.neighbors(n)
                        if edge_level.get((min(n, neighbor), max(n, neighbor))) == 1
                    ]
                    ang = _branch_angle_at_node(G, n, branch_neighbor, primary_neighbors)
                    if ang is not None:
                        angles.append(ang)
        n_roots = 1 if level == 1 else _count_secondary_roots_by_tips(G, edge_level, main_nodes)
        total_length = sum(lengths)
        avg_length = total_length / n_roots if n_roots else 0
        avg_diameter = np.mean(diameters) if diameters else 0
        avg_angle = np.degrees(np.mean(angles)) if angles else 0
        rows.append({
            "level": level,
            "name": ["主干", "二级根"][level - 1],
            "count": n_roots,
            "total_length": total_length,
            "avg_length": avg_length,
            "avg_diameter": avg_diameter,
            "avg_branch_angle_deg": avg_angle,
        })
    df = pd.DataFrame(rows)
    return df


def compute_detailed_edges(
    G: nx.Graph,
    edge_level: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int],
    min_root_length: float = 5.0,
    min_root_diameter: float = 1.0,
) -> pd.DataFrame:
    rows = []
    for (u, v), lev in edge_level.items():
        if not G.has_edge(u, v):
            continue
        d = G[u][v]
        L = d.get("length", 0)
        D = d.get("diameter", 0)
        if L < min_root_length or D < min_root_diameter:
            continue
        rows.append({
            "level": lev,
            "level_name": "主干" if lev == 1 else ("二级根" if lev == 2 else "其它"),
            "length": L,
            "diameter": D,
            "curvature": d.get("curvature", 0),
        })
    df = pd.DataFrame(rows)
    return df


def zscore_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns or not np.issubdtype(out[col].dtype, np.number):
            continue
        mean = out[col].mean()
        std = out[col].std()
        if std and std > 0:
            out[col] = (out[col] - mean) / std
    return out


def compute_all_stats(
    G: nx.Graph,
    edge_level: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int],
    min_root_length: float = 5.0,
    min_root_diameter: float = 1.0,
    apply_zscore: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = compute_level_stats(G, edge_level, min_root_length, min_root_diameter)
    detail = compute_detailed_edges(G, edge_level, min_root_length, min_root_diameter)
    if apply_zscore and not detail.empty:
        detail = zscore_columns(detail, ["length", "diameter", "curvature"])
    return summary, detail
