# -*- coding: utf-8 -*-
"""断根连接：骨架多分量时在 mask 上画桥再重新骨架化（来自 rootmask）。"""
import numpy as np
import cv2
from typing import Tuple, List, Set

from . import topology
from . import skeleton

def _normalize(dr: float, dc: float):
    n = np.sqrt(dr * dr + dc * dc)
    if n <= 1e-9:
        return 0.0, 0.0
    return dr / n, dc / n


def _endpoint_direction(skel: np.ndarray, r: int, c: int) -> Tuple[float, float]:
    """沿端点的短唯一链估计朝外切向，减小骨架末端单像素偏斜的影响。"""
    neighbors = topology._skeleton_neighbors(skel, r, c, exclude=None)
    if len(neighbors) != 1:
        return 0.0, 0.0
    previous = (r, c)
    current = neighbors[0]
    for _ in range(7):
        following = topology._skeleton_neighbors(skel, *current, exclude=previous)
        if len(following) != 1:
            break  # 不跨过分叉或另一个端点。
        previous, current = current, following[0]
    nr, nc = current
    # 从根段内部指向端点外侧，才能沿断口搜索另一根段。
    return _normalize(r - nr, c - nc)


def _ray_hit_foreground(
    mask: np.ndarray,
    pr: int, pc: int,
    dr: float, dc: float,
    max_search: int,
) -> Tuple[bool, int, int]:
    h, w = mask.shape
    fg = (np.asarray(mask) > 0)
    in_gap = False
    for t in range(1, max_search + 1):
        r = int(round(pr + t * dr))
        c = int(round(pc + t * dc))
        if r < 0 or r >= h or c < 0 or c >= w:
            break
        if fg[r, c]:
            if in_gap:
                return True, r, c
        else:
            in_gap = True
    return False, 0, 0


def _angle_between_rad(d1: Tuple[float, float], d2: Tuple[float, float]) -> float:
    a = np.arccos(np.clip(d1[0] * d2[0] + d1[1] * d2[1], -1.0, 1.0))
    return float(a)


def connect_broken_roots(
    mask: np.ndarray,
    skel: np.ndarray,
    max_search: int = 120,
    angle_tol_deg: float = 25.0,
    min_gap: float = 3.0,
    max_gap: float = 200.0,
    max_iter: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask)
    skel = np.asarray(skel, dtype=bool)
    h, w = skel.shape
    if mask.shape != (h, w):
        return np.array(mask, copy=True), np.array(skel, copy=True)

    if mask.dtype != np.uint8:
        work_mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
    else:
        work_mask = np.array(mask, copy=True)

    did_any_connection = False
    for _ in range(max_iter):
        num_labels, comp_id, _, _ = cv2.connectedComponentsWithStats(
            (work_mask > 0).astype(np.uint8), connectivity=8
        )
        num_components = num_labels - 1
        if num_components <= 1:
            break

        endpoints, _, _ = topology.find_keypoints(skel, junction_min_neighbors=3)
        comp_endpoints: List[List[Tuple[int, int]]] = [[] for _ in range(num_labels)]
        for (r, c) in endpoints:
            if 0 <= r < h and 0 <= c < w and skel[r, c]:
                lid = int(comp_id[r, c])
                if lid > 0:
                    comp_endpoints[lid].append((r, c))

        angle_tol_rad = np.deg2rad(angle_tol_deg)
        connected_pairs: Set[Tuple[int, int]] = set()
        any_bridge = False

        for (pr, pc) in endpoints:
            if pr < 0 or pr >= h or pc < 0 or pc >= w or not skel[pr, pc]:
                continue
            ci = int(comp_id[pr, pc])
            if ci <= 0:
                continue
            dr, dc = _endpoint_direction(skel, pr, pc)
            if abs(dr) < 1e-9 and abs(dc) < 1e-9:
                continue

            hit, qr, qc = _ray_hit_foreground(work_mask, pr, pc, dr, dc, max_search)
            if not hit:
                continue
            # 射线命中的是根的外边界，厚根的骨架可能缩在多个像素之外。
            # 用前景连通域直接识别命中的根段，不依赖边界处有相邻骨架。
            cj = int(comp_id[qr, qc])
            if cj <= 0 or cj == ci:
                continue
            pair = (min(ci, cj), max(ci, cj))
            if pair in connected_pairs:
                continue

            candidates = comp_endpoints[cj]
            if not candidates:
                continue
            best_r, best_c = min(candidates, key=lambda rc: (rc[0] - qr) ** 2 + (rc[1] - qc) ** 2)
            dist_pr = np.sqrt((best_r - pr) ** 2 + (best_c - pc) ** 2)
            if dist_pr < min_gap:
                continue
            if dist_pr > max_gap:
                continue

            d2r, d2c = _endpoint_direction(skel, best_r, best_c)
            if abs(d2r) < 1e-9 and abs(d2c) < 1e-9:
                continue
            angle = _angle_between_rad((dr, dc), (d2r, d2c))
            if abs(angle - np.pi) > angle_tol_rad:
                continue
            bridge_direction = _normalize(best_r - pr, best_c - pc)
            if _angle_between_rad((dr, dc), bridge_direction) > angle_tol_rad:
                continue
            if _angle_between_rad((d2r, d2c), (-bridge_direction[0], -bridge_direction[1])) > angle_tol_rad:
                continue

            cv2.line(work_mask, (pc, pr), (best_c, best_r), 255, 1)
            connected_pairs.add(pair)
            any_bridge = True
            did_any_connection = True

        if not any_bridge:
            break
        skel = skeleton.skeletonize_mask(work_mask)

    if not did_any_connection:
        if mask.dtype != np.uint8:
            out_mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
        else:
            out_mask = np.array(mask, copy=True)
        return out_mask, np.array(skel, copy=True)

    return work_mask, skel
