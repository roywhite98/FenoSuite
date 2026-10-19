# -*- coding: utf-8 -*-
"""Golden 序列化与对比工具。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

from plant_phenotype.root_phenotype.core.pipeline import PipelineResult


def _norm_edge(u: Tuple[int, int], v: Tuple[int, int]) -> List[int]:
    a, b = (u, v) if u < v else (v, u)
    return [int(a[0]), int(a[1]), int(b[0]), int(b[1])]


def _path_to_json(path: List[Tuple[int, int]]) -> List[List[int]]:
    return [[int(r), int(c)] for r, c in path]


def _level1_edges(edge_level: Dict) -> List[List[int]]:
    out: List[List[int]] = []
    for (u, v), lev in edge_level.items():
        if lev == 1:
            out.append(_norm_edge(u, v))
    return sorted(out)


def main_root_length_from_graph(G: nx.Graph, edge_level: Dict) -> float:
    if G is None:
        return 0.0
    total = 0.0
    for (u, v), lev in edge_level.items():
        if lev != 1 or not G.has_edge(u, v):
            continue
        total += float(G[u][v].get("length", 0.0))
    return total


def result_to_golden_dict(
    name: str,
    *,
    batch_mode: bool,
    selected_seeds: List[Tuple[int, int]] | None,
    res: PipelineResult,
) -> Dict[str, Any]:
    seeds_json = None
    if selected_seeds:
        seeds_json = [[int(r), int(c)] for r, c in selected_seeds]
    return {
        "name": name,
        "batch_mode": batch_mode,
        "selected_seeds": seeds_json,
        "error": res.error,
        "main_path": _path_to_json(res.main_path or []),
        "level1_edges": _level1_edges(res.edge_level or {}),
        "main_root_length": main_root_length_from_graph(res.G, res.edge_level or {}),
        "n_components": len(list(nx.connected_components(res.G))) if res.G is not None else 0,
    }


def save_golden(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_golden(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def level1_jaccard(expected: List[List[int]], actual: List[List[int]]) -> float:
    exp: Set[Tuple[int, int, int, int]] = {tuple(e) for e in expected}
    act: Set[Tuple[int, int, int, int]] = {tuple(e) for e in actual}
    if not exp and not act:
        return 1.0
    union = exp | act
    if not union:
        return 1.0
    return len(exp & act) / len(union)


def compare_to_golden(actual: Dict[str, Any], expected: Dict[str, Any]) -> None:
    if actual.get("error") != expected.get("error"):
        raise AssertionError(f"error mismatch: {actual.get('error')} vs {expected.get('error')}")
    if expected.get("error"):
        return
    exp_len = float(expected.get("main_root_length", 0.0))
    act_len = float(actual.get("main_root_length", 0.0))
    if exp_len > 1e-6:
        rel = abs(act_len - exp_len) / exp_len
        if rel >= 0.01:
            raise AssertionError(f"main_root_length rel err {rel:.4f}: {act_len} vs {exp_len}")
    j = level1_jaccard(expected.get("level1_edges", []), actual.get("level1_edges", []))
    if j < 0.98:
        raise AssertionError(f"level1 Jaccard {j:.4f} < 0.98")
