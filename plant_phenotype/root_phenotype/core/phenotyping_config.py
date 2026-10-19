# -*- coding: utf-8 -*-
"""根系表型流水线配置（dataclass + 默认 dict 合并）。"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional


def _load_defaults() -> Dict[str, Any]:
    try:
        from plant_phenotype.root_phenotype.config import (
            CONNECT_BROKEN_ROOTS,
            EDGE_WEIGHT,
            GRAPH_FEATURES,
            MAIN_ROOT,
            PREPROCESS,
            PRIMARY_ROOT,
            STATS,
            TOPOLOGY,
        )
    except ImportError:
        return {}
    return {
        "morph_kernel_size": PREPROCESS.get("morph_kernel_size", 3),
        "min_component_area": PREPROCESS.get("min_component_area", 50),
        "morph_open": PREPROCESS.get("morph_open", False),
        "junction_min_neighbors": TOPOLOGY.get("junction_min_neighbors", 3),
        "spur_max_length": int(TOPOLOGY.get("spur_max_length", 8)),
        "weight_length": EDGE_WEIGHT.get("length", 1.0),
        "weight_diameter": EDGE_WEIGHT.get("diameter", 0.5),
        "weight_curvature": EDGE_WEIGHT.get("curvature", -0.2),
        "max_main_roots": MAIN_ROOT.get("max_main_roots", 1),
        "inverse_weight_for_longest": MAIN_ROOT.get("inverse_weight_for_longest", True),
        "main_root_curvature_penalty": MAIN_ROOT.get("curvature_penalty", 1.0),
        "stroke_search_radius": MAIN_ROOT.get("stroke_search_radius", 24.0),
        "stroke_use_graph_weight": MAIN_ROOT.get("stroke_use_graph_weight", True),
        "min_root_length": STATS.get("min_root_length", 5.0),
        "min_root_diameter": STATS.get("min_root_diameter", 1.0),
        "zscore": STATS.get("zscore", True),
        "connect_broken_roots": CONNECT_BROKEN_ROOTS.get("enabled", False),
        "connect_max_search": CONNECT_BROKEN_ROOTS.get("max_search", 120),
        "connect_angle_tol_deg": CONNECT_BROKEN_ROOTS.get("angle_tol_deg", 25.0),
        "connect_min_gap": CONNECT_BROKEN_ROOTS.get("min_gap", 3.0),
        "connect_max_gap": CONNECT_BROKEN_ROOTS.get("max_gap", 200.0),
        "compute_betweenness": GRAPH_FEATURES.get("compute_betweenness", False),
        "primary_root_enabled": PRIMARY_ROOT.get("enabled", False),
        "primary_alpha": float(PRIMARY_ROOT.get("alpha", 1.0)),
        "primary_beta": float(PRIMARY_ROOT.get("beta", 0.5)),
        "primary_gamma": float(PRIMARY_ROOT.get("gamma", 0.8)),
        "primary_tau": float(PRIMARY_ROOT.get("tau", 0.75)),
        "primary_d_attr": str(PRIMARY_ROOT.get("d_attr", "diameter")),
        "primary_lambda1": float(PRIMARY_ROOT.get("lambda1", 1.0)),
        "primary_lambda2": float(PRIMARY_ROOT.get("lambda2", 0.5)),
        "primary_max_level": int(PRIMARY_ROOT.get("max_level", 3)),
        "primary_diameter_ratio_min": float(PRIMARY_ROOT.get("diameter_ratio_min", 0.3)),
        "primary_diameter_floor": float(PRIMARY_ROOT.get("diameter_floor", 1.0)),
        "primary_normalize_score": bool(PRIMARY_ROOT.get("normalize_score", True)),
    }


@dataclass
class PhenotypingConfig:
    morph_kernel_size: int = 3
    min_component_area: int = 50
    morph_open: bool = False
    junction_min_neighbors: int = 3
    spur_max_length: int = 8
    weight_length: float = 1.0
    weight_diameter: float = 0.5
    weight_curvature: float = -0.2
    max_main_roots: int = 1
    inverse_weight_for_longest: bool = True
    main_root_curvature_penalty: float = 1.0
    stroke_search_radius: float = 24.0
    stroke_use_graph_weight: bool = True
    min_root_length: float = 5.0
    min_root_diameter: float = 1.0
    zscore: bool = True
    connect_broken_roots: bool = False
    connect_max_search: int = 120
    connect_angle_tol_deg: float = 25.0
    connect_min_gap: float = 3.0
    connect_max_gap: float = 200.0
    compute_betweenness: bool = False
    primary_root_enabled: bool = False
    primary_alpha: float = 1.0
    primary_beta: float = 0.5
    primary_gamma: float = 0.8
    primary_tau: float = 0.75
    primary_d_attr: str = "diameter"
    primary_lambda1: float = 1.0
    primary_lambda2: float = 0.5
    primary_max_level: int = 3
    primary_diameter_ratio_min: float = 0.3
    primary_diameter_floor: float = 1.0
    primary_normalize_score: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, overrides: Optional[Dict[str, Any]] = None) -> PhenotypingConfig:
        base = _load_defaults()
        if overrides:
            base.update(overrides)
        valid = {f.name for f in fields(cls)}
        kwargs = {k: base[k] for k in base if k in valid}
        return cls(**kwargs)
