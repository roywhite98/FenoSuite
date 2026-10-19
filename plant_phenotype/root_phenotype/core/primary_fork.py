# -*- coding: utf-8 -*-
"""分叉点局部竞争主根：D（粗细）+ S（子树支持）+ C（方向连续性），多赢家 τ。"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx

Node = Tuple[int, int]
EdgeKey = Tuple[Node, Node]


def norm_edge(u: Node, v: Node) -> EdgeKey:
    return (u, v) if u < v else (v, u)


def _node_pos(G: nx.Graph, n: Node) -> np.ndarray:
    p = G.nodes[n].get("pos")
    if p is not None:
        return np.array([float(p[0]), float(p[1])], dtype=np.float64)
    return np.array([float(n[0]), float(n[1])], dtype=np.float64)


def _edge_D(G: nx.Graph, u: Node, v: Node, d_attr: str) -> float:
    d = G[u][v].get(d_attr, G[u][v].get("diameter", 1.0))
    return float(d)


def _unit_vec(G: nx.Graph, fr: Node, to: Node) -> np.ndarray:
    a = _node_pos(G, fr)
    b = _node_pos(G, to)
    w = b - a
    n = float(np.linalg.norm(w)) or 1.0
    return w / n


def cos_continuity(G: nx.Graph, parent: Optional[Node], v: Node, child: Node) -> float:
    if parent is None:
        return 1.0
    v1 = _unit_vec(G, parent, v)
    v2 = _unit_vec(G, v, child)
    return float(np.clip(np.dot(v1, v2), -1.0, 1.0))


def nearest_graph_nodes_to_seeds(G: nx.Graph, seed_pixels: List[Tuple[int, int]]) -> List[Node]:
    from .nearest import nearest_nodes_for_seeds

    nodes = list(G.nodes())
    if not nodes or not seed_pixels:
        return []
    return nearest_nodes_for_seeds(nodes, seed_pixels)


def resolve_seed_nodes(
    G: nx.Graph,
    union_mask: np.ndarray,
    seeds: Optional[List[Tuple[int, int]]],
    batch_mode: bool,
) -> List[Node]:
    if batch_mode or not seeds:
        return []
    return nearest_graph_nodes_to_seeds(G, seeds)


def batch_seed_nodes_from_mask_top(G: nx.Graph, union_mask: np.ndarray) -> List[Node]:
    ys, xs = np.where(union_mask > 0)
    if len(ys) == 0:
        return []
    top = int(ys.min())
    cols = np.where(union_mask[top] > 0)[0]
    if len(cols) == 0:
        return []
    step = max(1, len(cols) // 48)
    pixels = [(top, int(c)) for c in cols[::step]][:50]
    return nearest_graph_nodes_to_seeds(G, pixels)


def multi_source_bfs_tree(G: nx.Graph, seed_nodes: List[Node]) -> Dict[Node, Optional[Node]]:
    parent: Dict[Node, Optional[Node]] = {}
    q: deque = deque()
    for s in seed_nodes:
        if s in G:
            parent[s] = None
            q.append(s)
    while q:
        u = q.popleft()
        for v in G.neighbors(u):
            if v not in parent:
                parent[v] = u
                q.append(v)
    return parent


def compute_subtree_support(
    G: nx.Graph,
    parent: Dict[Node, Optional[Node]],
    d_attr: str,
) -> Dict[EdgeKey, float]:
    memo: Dict[EdgeKey, float] = {}

    def dfs(v: Node, par: Optional[Node]) -> float:
        tot = 0.0
        for w in G.neighbors(v):
            if w == par:
                continue
            if parent.get(w) != v:
                continue
            D = _edge_D(G, v, w, d_attr)
            L = float(G[v][w].get("length", 1.0))
            sub = dfs(w, v)
            sm = D * L + sub
            memo[norm_edge(v, w)] = sm
            tot += sm
        return tot

    for s in parent:
        if parent.get(s) is None and s in G:
            dfs(s, None)
    return memo


def _fork_winners_at(
    G: nx.Graph,
    parent: Dict[Node, Optional[Node]],
    v: Node,
    children: List[Node],
    subtree_S: Dict[EdgeKey, float],
    d_attr: str,
    alpha: float,
    beta: float,
    gamma: float,
    tau: float,
    normalize_score: bool,
    diameter_ratio_min: float,
    diameter_floor: float,
) -> List[EdgeKey]:
    pv = parent.get(v)
    if len(children) <= 1:
        out: List[EdgeKey] = []
        for w in children:
            D = _edge_D(G, v, w, d_attr)
            if D < diameter_floor:
                continue
            if pv is not None and diameter_ratio_min > 0:
                pd = _edge_D(G, pv, v, d_attr)
                if D < diameter_ratio_min * max(pd, 1e-9):
                    continue
            out.append(norm_edge(v, w))
        return out

    cands: List[Tuple[Node, float, float, float, bool]] = []
    for w in children:
        ne = norm_edge(v, w)
        D = _edge_D(G, v, w, d_attr)
        S = float(subtree_S.get(ne, 0.0))
        C = cos_continuity(G, pv, v, w)
        ok = True
        if D < diameter_floor:
            ok = False
        if ok and pv is not None and diameter_ratio_min > 0:
            pd = _edge_D(G, pv, v, d_attr)
            if D < diameter_ratio_min * max(pd, 1e-9):
                ok = False
        cands.append((w, D, S, C, ok))

    valid = [(w, D, S, C) for w, D, S, C, ok in cands if ok]
    if not valid:
        return []

    Ds = [t[1] for t in valid]
    Ss = [t[2] for t in valid]
    dmax = max(Ds) if Ds else 1.0
    smax = max(Ss) if Ss else 1.0
    if dmax <= 0:
        dmax = 1.0
    if smax <= 0:
        smax = 1.0

    scores: List[Tuple[Node, float]] = []
    for w, D, S, C in valid:
        if normalize_score:
            Dn = D / dmax
            Sn = S / smax
        else:
            Dn, Sn = D, S
        sc = alpha * Dn + beta * Sn + gamma * C
        scores.append((w, sc))
    best = max(sc for _, sc in scores)
    thr = tau * best if best > 0 else 0.0
    win: List[EdgeKey] = []
    for w, sc in scores:
        if sc >= thr - 1e-9:
            win.append(norm_edge(v, w))
    return win


def compute_e_primary(
    G: nx.Graph,
    seed_nodes: List[Node],
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.8,
    tau: float = 0.75,
    d_attr: str = "diameter",
    normalize_score: bool = True,
    diameter_ratio_min: float = 0.3,
    diameter_floor: float = 1.0,
) -> Tuple[Set[EdgeKey], Dict[EdgeKey, float], Dict[str, Any], Dict[str, Any]]:
    if G.number_of_edges() == 0 or not seed_nodes:
        return set(), {}, {}, {}

    parent = multi_source_bfs_tree(G, seed_nodes)
    subtree_S = compute_subtree_support(G, parent, d_attr)

    E_raw: Set[EdgeKey] = set()
    for v in G.nodes():
        ch = [w for w in G.neighbors(v) if parent.get(w) == v]
        if not ch:
            continue
        wins = _fork_winners_at(
            G,
            parent,
            v,
            ch,
            subtree_S,
            d_attr,
            alpha,
            beta,
            gamma,
            tau,
            normalize_score,
            diameter_ratio_min,
            diameter_floor,
        )
        E_raw.update(wins)

    E_filt = _filter_reachable_from_seeds(E_raw, G, seed_nodes, parent)
    return E_filt, subtree_S, {}, {}


def _filter_reachable_from_seeds(
    E_primary: Set[EdgeKey],
    G: nx.Graph,
    seed_nodes: List[Node],
    parent: Dict[Node, Optional[Node]],
) -> Set[EdgeKey]:
    if not E_primary:
        return set()
    seeds = set(seed_nodes)
    reached: Set[EdgeKey] = set()
    q: deque = deque()
    for s in seed_nodes:
        if s not in G:
            continue
        for v in G.neighbors(s):
            if parent.get(v) != s:
                continue
            ne = norm_edge(s, v)
            if ne in E_primary:
                reached.add(ne)
                q.append(v)
    if not reached:
        for e in E_primary:
            u, v = e[0], e[1]
            if u in seeds and v in seeds:
                reached.add(e)
            elif u in seeds and parent.get(v) == u:
                reached.add(e)
                q.append(v)
            elif v in seeds and parent.get(u) == v:
                reached.add(e)
                q.append(u)
    if not reached:
        return set(E_primary)

    seen: Set[Node] = set()
    while q:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        for w in G.neighbors(u):
            if parent.get(w) != u:
                continue
            ne = norm_edge(u, w)
            if ne not in E_primary or ne in reached:
                continue
            reached.add(ne)
            q.append(w)
    return reached
