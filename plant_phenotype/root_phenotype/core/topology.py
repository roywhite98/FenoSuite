# -*- coding: utf-8 -*-
"""关键点抽象：端点、分叉点，线性骨架段压缩（来自 rootmask）。"""
import logging
import math
import numpy as np
from typing import List, Tuple, Dict, Set, Optional

log = logging.getLogger(__name__)

_NEIGHBORS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def _neighbor_count(skel: np.ndarray, r: int, c: int) -> int:
    h, w = skel.shape
    n = 0
    for dr, dc in _NEIGHBORS_8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and skel[nr, nc]:
            n += 1
    return n


def _skeleton_neighbors(skel: np.ndarray, r: int, c: int, exclude: Tuple[int, int] = None) -> List[Tuple[int, int]]:
    h, w = skel.shape
    out = []
    for dr, dc in _NEIGHBORS_8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and skel[nr, nc] and (exclude is None or (nr, nc) != exclude):
            out.append((nr, nc))
    return out


def find_keypoints(
    skel: np.ndarray,
    junction_min_neighbors: int = 3,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], np.ndarray]:
    """8 邻域度数：端点=1，分叉=度≥junction_min_neighbors（向量化）。"""
    sk = np.asarray(skel, dtype=bool)
    keypoint_mask = np.zeros_like(sk, dtype=bool)
    if not np.any(sk):
        return [], [], keypoint_mask
    try:
        from scipy import ndimage

        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        neighbor_count = ndimage.convolve(sk.astype(np.uint8), kernel, mode="constant", cval=0)
        ys, xs = np.where(sk)
        counts = neighbor_count[ys, xs]
        endpoints: List[Tuple[int, int]] = []
        junctions: List[Tuple[int, int]] = []
        for i in range(len(ys)):
            nc = int(counts[i])
            r, c = int(ys[i]), int(xs[i])
            if nc == 1:
                endpoints.append((r, c))
                keypoint_mask[r, c] = True
            elif nc >= junction_min_neighbors:
                junctions.append((r, c))
                keypoint_mask[r, c] = True
        return endpoints, junctions, keypoint_mask
    except ImportError:
        pass
    endpoints = []
    junctions = []
    h, w = sk.shape
    for r in range(h):
        for c in range(w):
            if not sk[r, c]:
                continue
            nc = _neighbor_count(sk, r, c)
            if nc == 1:
                endpoints.append((r, c))
                keypoint_mask[r, c] = True
            elif nc >= junction_min_neighbors:
                junctions.append((r, c))
                keypoint_mask[r, c] = True
    return endpoints, junctions, keypoint_mask


def _edge(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return (min(a, b), max(a, b))


def segments_between_keypoints(
    skel: np.ndarray,
    keypoint_mask: np.ndarray,
    endpoints: List[Tuple[int, int]],
    junctions: List[Tuple[int, int]],
) -> List[Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]]:
    all_keypoints = set(endpoints) | set(junctions)
    segments = []
    used_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    h, w = skel.shape
    for start in all_keypoints:
        sr, sc = start
        for nb in _skeleton_neighbors(skel, sr, sc, exclude=None):
            e = _edge(start, nb)
            if e in used_edges:
                continue
            path = [start, nb]
            used_edges.add(e)
            cur = nb
            prev = start
            while True:
                if cur in all_keypoints and cur != start:
                    segments.append((start, cur, path))
                    break
                neighbors = _skeleton_neighbors(skel, cur[0], cur[1], exclude=prev)
                if len(neighbors) == 0:
                    break
                if len(neighbors) != 1:
                    if cur in all_keypoints and cur != start:
                        segments.append((start, cur, path))
                    break
                nxt = neighbors[0]
                e2 = _edge(cur, nxt)
                if e2 in used_edges:
                    break
                used_edges.add(e2)
                path.append(nxt)
                prev = cur
                cur = nxt
    return segments


def _turn_angle_rad(
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    p2: Tuple[int, int],
) -> float:
    """p1 处折转角（弧度），共线约 0，直角约 π/2。"""
    v1 = (p1[0] - p0[0], p1[1] - p0[1])
    v2 = (p2[0] - p1[0], p2[1] - p1[1])
    n1 = math.hypot(v1[0], v1[1]) or 1.0
    n2 = math.hypot(v2[0], v2[1]) or 1.0
    c = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    c = max(-1.0, min(1.0, c))
    return math.acos(c)


def _curvature_breakpoints_along_segments(
    segments: List[Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]],
    angle_deg: float,
) -> List[Tuple[int, int]]:
    """在段链内角点超过阈值处插入新关键点（作为 junction 参与分段）。"""
    if angle_deg <= 0:
        return []
    thr = math.radians(angle_deg)
    extra: List[Tuple[int, int]] = []
    for _a, _b, path in segments:
        if len(path) < 3:
            continue
        for k in range(1, len(path) - 1):
            ang = _turn_angle_rad(path[k - 1], path[k], path[k + 1])
            if ang >= thr:
                extra.append(path[k])
    return extra


def build_topology(
    skel: np.ndarray,
    junction_min_neighbors: int = 3,
    curvature_angle_deg: Optional[float] = None,
) -> Tuple[
    List[Tuple[int, int]],
    List[Tuple[int, int]],
    List[Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]],
    Dict[Tuple[int, int], int],
    Dict[Tuple[int, int], int],
]:
    endpoints, junctions, keypoint_mask = find_keypoints(skel, junction_min_neighbors)
    segments = segments_between_keypoints(skel, keypoint_mask, endpoints, junctions)
    if curvature_angle_deg is not None and curvature_angle_deg > 0:
        extra_pts = _curvature_breakpoints_along_segments(segments, float(curvature_angle_deg))
        if extra_pts:
            for p in extra_pts:
                keypoint_mask[p[0], p[1]] = True
            merged: List[Tuple[int, int]] = list(dict.fromkeys(list(junctions) + extra_pts))
            junctions = merged
            segments = segments_between_keypoints(skel, keypoint_mask, endpoints, junctions)
            log.info("曲率关键点: 新增 %d 个, 拓扑段数=%d", len(extra_pts), len(segments))
        else:
            log.info("拓扑段数=%d", len(segments))
    else:
        log.info("拓扑段数=%d", len(segments))
    endpoint_ids = {p: i for i, p in enumerate(endpoints)}
    junction_ids = {p: i for i, p in enumerate(junctions)}
    return endpoints, junctions, segments, endpoint_ids, junction_ids
