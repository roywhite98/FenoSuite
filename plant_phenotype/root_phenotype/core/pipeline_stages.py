# -*- coding: utf-8 -*-
"""pipeline 分阶段纯函数（图分析、主根、统计）。"""
from __future__ import annotations

import logging
import hashlib
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from . import connect_roots
from . import features
from . import graph
from . import main_root
from . import skeleton
from . import skeleton_prune
from . import stats
from . import tree
from .phenotyping_config import PhenotypingConfig
from .skeleton_cache import get_graph_cache, get_skeleton_cache, mask_mtime_ns

log = logging.getLogger(__name__)


def make_skeleton_cache_key(
    mask_path: str,
    cfg: PhenotypingConfig,
    seed_key: Optional[Tuple[Tuple[int, int], ...]] = None,
    union_mask: Optional[np.ndarray] = None,
) -> tuple:
    preprocess_params = (
        cfg.morph_kernel_size,
        cfg.min_component_area,
        cfg.morph_open,
        cfg.connect_broken_roots,
        cfg.connect_max_search,
        cfg.connect_angle_tol_deg,
        cfg.connect_min_gap,
        cfg.connect_max_gap,
        cfg.junction_min_neighbors,
        cfg.spur_max_length,
        mask_mtime_ns(mask_path),
    )
    # 同一路径可能对应不同的内存掩码（尤其是统一标识为 <array> 的 API 输入）。
    # 文件时间戳和种子无法标识这类变化，必须把实际待分析内容纳入骨架/图缓存键。
    fingerprint = None
    if union_mask is not None:
        binary = np.ascontiguousarray(np.asarray(union_mask) > 0)
        fingerprint = (binary.shape, hashlib.sha256(binary).digest())
    return (mask_path,) + preprocess_params + (seed_key, fingerprint)


def make_graph_cache_key(
    skel_cache_key: tuple,
    cfg: PhenotypingConfig,
) -> tuple:
    return skel_cache_key + (
        cfg.weight_length,
        cfg.weight_diameter,
        cfg.weight_curvature,
        bool(cfg.compute_betweenness),
    )


def skeletonize_union_mask(
    mask_path: str,
    union_mask: np.ndarray,
    cfg: PhenotypingConfig,
    seed_key: Optional[Tuple[Tuple[int, int], ...]],
) -> Tuple[np.ndarray, np.ndarray, tuple]:
    """返回 (union_mask, skel, skeleton_cache_key)。"""
    cache_key = make_skeleton_cache_key(mask_path, cfg, seed_key, union_mask)
    cached = get_skeleton_cache().get(cache_key)
    if cached is not None:
        log.info("骨架缓存命中: %s", mask_path)
        return cached[0].copy(), cached[1].copy(), cache_key
    log.info("骨架缓存未命中: %s，执行骨架化", mask_path)
    t0 = time.perf_counter()
    skel = skeleton.skeletonize_mask(union_mask)
    if cfg.connect_broken_roots:
        union_mask, skel = connect_roots.connect_broken_roots(
            union_mask,
            skel,
            max_search=cfg.connect_max_search,
            angle_tol_deg=cfg.connect_angle_tol_deg,
            min_gap=cfg.connect_min_gap,
            max_gap=cfg.connect_max_gap,
        )
    if cfg.spur_max_length > 0:
        skel = skeleton_prune.prune_skeleton_terminal_spurs(
            skel,
            max_spur_length=int(cfg.spur_max_length),
            junction_min_neighbors=cfg.junction_min_neighbors,
        )
    # Analysis results and source arrays are mutable; keep cache-owned copies.
    get_skeleton_cache().put(cache_key, (union_mask.copy(), skel.copy()))
    log.info(
        "骨架结果已缓存: %s (%.0fms, 骨架像素=%d)",
        mask_path,
        (time.perf_counter() - t0) * 1000,
        int(np.count_nonzero(skel)),
    )
    return union_mask, skel, cache_key


def build_graph_from_skeleton(
    skel: np.ndarray,
    union_mask: np.ndarray,
    cfg: PhenotypingConfig,
    skel_cache_key: Optional[tuple] = None,
) -> Tuple[nx.Graph, list, list, dict, dict]:
    gkey = make_graph_cache_key(skel_cache_key, cfg) if skel_cache_key is not None else None
    if gkey is not None:
        cached = get_graph_cache().get(gkey)
        if cached is not None:
            log.info("拓扑图缓存命中")
            # A shallow graph copy still shares edge pixel lists and arrays.
            return deepcopy(cached)
    t0 = time.perf_counter()
    G, endpoints, junctions, endpoint_ids, junction_ids = graph.build_weighted_graph(
        skel,
        junction_min_neighbors=cfg.junction_min_neighbors,
        weight_length=cfg.weight_length,
        weight_diameter=cfg.weight_diameter,
        weight_curvature=cfg.weight_curvature,
        root_mask=union_mask,
    )
    features.compute_graph_features(G, compute_betweenness=cfg.compute_betweenness)
    result = (G, endpoints, junctions, endpoint_ids, junction_ids)
    if gkey is not None:
        get_graph_cache().put(gkey, deepcopy(result))
    log.info(
        "拓扑图构建完成: %.0fms, 节点=%d, 交点=%d",
        (time.perf_counter() - t0) * 1000,
        G.number_of_nodes(),
        len(junctions),
    )
    return result


def analyze_components(
    G: nx.Graph,
    union_mask: np.ndarray,
    cfg: PhenotypingConfig,
    seeds: List[Tuple[int, int]],
    batch_mode: bool,
) -> Tuple[List[Dict[str, Any]], Dict, List[Tuple[int, int]]]:
    components = list(nx.connected_components(G))
    roots_list: List[Dict[str, Any]] = []
    edge_level: Dict = {}
    main_path_first: List[Tuple[int, int]] = []

    for comp in components:
        sub = G.subgraph(comp).copy()
        use_primary = cfg.primary_root_enabled
        E_primary_set = None
        subtree_S_map = None
        if use_primary:
            from . import primary_fork

            sn = primary_fork.resolve_seed_nodes(sub, union_mask, seeds if seeds else None, batch_mode)
            if not sn:
                sn = primary_fork.batch_seed_nodes_from_mask_top(sub, union_mask)
            E_primary_set, subtree_S_map, _, _ = primary_fork.compute_e_primary(
                sub,
                sn,
                alpha=cfg.primary_alpha,
                beta=cfg.primary_beta,
                gamma=cfg.primary_gamma,
                tau=cfg.primary_tau,
                d_attr=cfg.primary_d_attr,
                normalize_score=cfg.primary_normalize_score,
                diameter_ratio_min=cfg.primary_diameter_ratio_min,
                diameter_floor=cfg.primary_diameter_floor,
            )
        if use_primary and E_primary_set:
            main_path_comp = main_root.find_representative_path_in_primary_subgraph(
                sub,
                E_primary_set,
                subtree_S=subtree_S_map,
                lambda1=cfg.primary_lambda1,
                lambda2=cfg.primary_lambda2,
                d_attr=cfg.primary_d_attr,
            )
            if not main_path_comp:
                e0 = next(iter(E_primary_set))
                u, v = e0[0], e0[1]
                main_path_comp = [u, v] if sub.has_edge(u, v) else [u]
            edge_level_comp = tree.assign_edge_levels_from_primary_edges(sub, E_primary_set)
        else:
            if len(comp) >= 2:
                if cfg.inverse_weight_for_longest:
                    main_path_comp = main_root.find_longest_path_approximate(sub)
                    if not main_path_comp:
                        main_path_comp = main_root.find_smooth_main_path(
                            sub, curvature_penalty=cfg.main_root_curvature_penalty
                        )
                else:
                    main_path_comp = main_root.find_smooth_main_path(
                        sub, curvature_penalty=cfg.main_root_curvature_penalty
                    )
            else:
                main_path_comp = []
            if not main_path_comp:
                main_path_comp = list(sub.nodes())[:1] if sub.nodes() else []
            else:
                main_path_comp = main_root.ensure_simple_path_on_graph(sub, main_path_comp)
                if not main_path_comp:
                    main_path_comp = list(sub.nodes())[:1] if sub.nodes() else []
            edge_level_comp = tree.build_rooted_tree(
                sub,
                main_path_comp,
                min_root_length=cfg.min_root_length,
                min_root_diameter=cfg.min_root_diameter,
            )
        if not main_path_first:
            main_path_first = main_path_comp
        edge_level.update(edge_level_comp)
        summary_comp, detail_comp = stats.compute_all_stats(
            sub,
            edge_level_comp,
            min_root_length=cfg.min_root_length,
            min_root_diameter=cfg.min_root_diameter,
            apply_zscore=cfg.zscore,
        )
        roots_list.append({
            "summary_df": summary_comp,
            "detail_df": detail_comp,
            "main_path": main_path_comp,
            "edge_level": edge_level_comp,
            "primary_edges": E_primary_set,
            "subtree_support": subtree_S_map,
        })

    if not main_path_first and roots_list:
        main_path_first = roots_list[0].get("main_path", [])
    return roots_list, edge_level, main_path_first
