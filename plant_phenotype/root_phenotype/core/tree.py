# -*- coding: utf-8 -*-
"""有根树分级：主干=一级根，整条分支=二级根（来自 rootmask）。"""
import networkx as nx
from typing import List, Tuple, Set, Dict


def get_main_edges_set(path: List[Tuple[int, int]]) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
    edges = set()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edges.add((min(u, v), max(u, v)))
    return edges


def _norm(u: Tuple[int, int], v: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return (min(u, v), max(u, v))


def _collect_branch_from_main(
    G: nx.Graph,
    main_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
    main_nodes: Set[Tuple[int, int]],
    start_u: Tuple[int, int],
    start_v: Tuple[int, int],
) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
    branch = {_norm(start_u, start_v)}
    queue = [start_v]
    visited = {start_v}
    while queue:
        u = queue.pop(0)
        for v in G.neighbors(u):
            key = _norm(u, v)
            if key in main_edges or key in branch:
                continue
            if v in main_nodes:
                continue
            branch.add(key)
            if v not in visited:
                visited.add(v)
                queue.append(v)
    return branch


def assign_edge_levels_from_primary_edges(
    G: nx.Graph,
    E_primary: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], int]:
    """主根边集合为一级，挂接分支为二级，其余为四级（与 assign_edge_levels 语义一致）。"""
    main_edges = set(E_primary)
    main_nodes: Set[Tuple[int, int]] = set()
    for u, v in main_edges:
        main_nodes.add(u)
        main_nodes.add(v)
    edge_level: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}
    for e in main_edges:
        edge_level[e] = 1
    for n in main_nodes:
        for nb in G.neighbors(n):
            key = _norm(n, nb)
            if key in edge_level:
                continue
            branch = _collect_branch_from_main(G, main_edges, main_nodes, n, nb)
            for e in branch:
                edge_level[e] = 2
    for u, v in G.edges():
        key = _norm(u, v)
        if key not in edge_level:
            edge_level[key] = 4
    return edge_level


def assign_edge_levels(
    G: nx.Graph,
    main_path: List[Tuple[int, int]],
) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], int]:
    main_edges = get_main_edges_set(main_path)
    main_nodes = set(main_path)
    edge_level = {}

    for i in range(len(main_path) - 1):
        u, v = main_path[i], main_path[i + 1]
        edge_level[_norm(u, v)] = 1

    for n in main_nodes:
        for nb in G.neighbors(n):
            key = _norm(n, nb)
            if key in edge_level:
                continue
            branch = _collect_branch_from_main(G, main_edges, main_nodes, n, nb)
            for e in branch:
                edge_level[e] = 2

    for u, v in G.edges():
        key = _norm(u, v)
        if key not in edge_level:
            edge_level[key] = 4

    return edge_level


def build_rooted_tree(
    G: nx.Graph,
    main_path: List[Tuple[int, int]],
    min_root_length: float = 5.0,
    min_root_diameter: float = 1.0,
) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], int]:
    return assign_edge_levels(G, main_path)
