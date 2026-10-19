# -*- coding: utf-8 -*-
"""主干识别：复合权 w 上的近似最长路径，及可选的平滑最短路（length + λ·曲率）。"""
from collections import deque
from typing import Dict, Hashable, Iterable, List, Optional, Set, Tuple

import networkx as nx

Node = Tuple[int, int]

# 端点对数超过此阈值时不再做全组合 Dijkstra（O(E²)），改用少量种子源点。
_MAX_EXACT_ENDPOINTS = 24
# 大图上每个连通分量最多跑这么多次单源 Dijkstra。
_MAX_DIJKSTRA_SOURCES = 12


def _inverse_weight(G: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        H.add_edge(u, v, weight=max(1e-9, 1.0 / w))
    return H


def _endpoints(G: nx.Graph) -> List[Node]:
    return [n for n in G.nodes() if G.degree(n) == 1]


def _path_raw_weight(G: nx.Graph, path: List[Node]) -> float:
    return float(
        sum(G[path[k]][path[k + 1]].get("weight", 0.0) for k in range(len(path) - 1))
    )


def _select_spatial_seeds(nodes: List[Node], k: int) -> List[Node]:
    """按包围盒/对角线极值选取至多 k 个空间种子（O(n)，不看图距离）。"""
    if not nodes:
        return []
    if len(nodes) <= k:
        return list(nodes)

    def row(n: Node) -> float:
        return float(n[0])

    def col(n: Node) -> float:
        return float(n[1])

    picks: List[Node] = []
    seen: Set[Node] = set()

    def add(n: Node) -> None:
        if n not in seen:
            seen.add(n)
            picks.append(n)

    add(min(nodes, key=row))
    add(max(nodes, key=row))
    add(min(nodes, key=col))
    add(max(nodes, key=col))
    add(min(nodes, key=lambda n: row(n) + col(n)))
    add(max(nodes, key=lambda n: row(n) + col(n)))
    add(min(nodes, key=lambda n: row(n) - col(n)))
    add(max(nodes, key=lambda n: row(n) - col(n)))

    if len(picks) < k:
        cy = sum(row(n) for n in nodes) / len(nodes)
        cx = sum(col(n) for n in nodes) / len(nodes)
        ranked = sorted(
            nodes,
            key=lambda n: -(row(n) - cy) ** 2 - (col(n) - cx) ** 2,
        )
        for n in ranked:
            add(n)
            if len(picks) >= k:
                break
    return picks[:k]


def _farthest_endpoint_pair(G: nx.Graph) -> Optional[Tuple[Node, Node]]:
    endpoints = _endpoints(G)
    if len(endpoints) < 2:
        nodes = list(G.nodes())
        if len(nodes) < 2:
            return None
        endpoints = nodes[: min(10, len(nodes))]
    # 端点很多时用空间极值代替全组合欧氏扫描
    if len(endpoints) > 64:
        seeds = _select_spatial_seeds(endpoints, 16)
        best_pair = None
        best_d2 = -1.0
        for i, a in enumerate(seeds):
            for b in seeds[i + 1 :]:
                d2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                if d2 > best_d2:
                    best_d2 = d2
                    best_pair = (a, b)
        return best_pair
    best_pair = None
    best_d2 = -1.0
    for i, a in enumerate(endpoints):
        for b in endpoints[i + 1 :]:
            d2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            if d2 > best_d2:
                best_d2 = d2
                best_pair = (a, b)
    return best_pair


def find_smooth_main_path(
    G: nx.Graph,
    curvature_penalty: float = 1.0,
) -> List[Node]:
    if G.number_of_nodes() < 2:
        return list(G.nodes())
    pair = _farthest_endpoint_pair(G)
    if pair is None:
        return list(G.nodes())[:1]
    s, t = pair
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        length = d.get("length", 1.0)
        curvature = d.get("curvature", 0.0)
        smooth = length + curvature_penalty * curvature
        H.add_edge(u, v, weight=max(1e-9, smooth))
    try:
        path = nx.shortest_path(H, s, t, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [s]
    return path


def _is_tree_graph(G: nx.Graph) -> bool:
    n, m = G.number_of_nodes(), G.number_of_edges()
    if n == 0:
        return False
    if m != n - 1:
        return False
    try:
        return nx.is_connected(G)
    except Exception:
        return False


def _tree_farthest(G: nx.Graph, start: Node) -> Tuple[Node, dict]:
    """树中路径唯一；一次遍历累计实际边权即可求最远点。"""
    dist = {start: 0.0}
    queue: deque = deque([start])
    farthest = start
    while queue:
        u = queue.popleft()
        for v in G.neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + float(G[u][v].get("weight", 1.0))
                queue.append(v)
                if dist[v] >= dist.get(farthest, 0):
                    farthest = v
    return farthest, dist


def _longest_path_tree_diameter(G: nx.Graph) -> Optional[List[Node]]:
    if G.number_of_nodes() < 2 or not _is_tree_graph(G):
        return None
    nodes = list(G.nodes())
    start = nodes[0]
    u, _ = _tree_farthest(G, start)
    v, _ = _tree_farthest(G, u)
    try:
        return nx.shortest_path(G, u, v)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _best_path_from_sources(
    G: nx.Graph,
    H: nx.Graph,
    sources: Iterable[Node],
    targets: Optional[Set[Node]] = None,
) -> List[Node]:
    """对每个源点做一次单源 Dijkstra，在目标集合中挑 raw-weight 最大的路径。

    复杂度 O(|sources| · Dijkstra)，而非端点全组合 O(|E|² · Dijkstra)。
    """
    target_set = targets
    best_path: List[Node] = []
    best_raw = -1.0
    for s in sources:
        if s not in H:
            continue
        try:
            paths = nx.single_source_dijkstra_path(H, s, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        for t, path in paths.items():
            if t == s or len(path) < 2:
                continue
            if target_set is not None and t not in target_set:
                continue
            raw = _path_raw_weight(G, path)
            if raw > best_raw:
                best_raw = raw
                best_path = path
    return best_path


def find_longest_path_approximate(G: nx.Graph) -> List[Node]:
    """复合边权 w 上的近似最长简单路径。

    - 树：两次遍历累计非负复合边权，精确求加权直径。
    - 一般图：端点较少时对每个端点做一次单源 Dijkstra（等价原全组合，但少一半常数）；
      端点很多时只用空间极值种子 + 一次 2-sweep 精炼，避免 O(E²)。
    """
    if G.number_of_nodes() == 0:
        return []
    tree_path = _longest_path_tree_diameter(G)
    if tree_path:
        return tree_path

    endpoints = _endpoints(G)
    if len(endpoints) < 2:
        nodes = list(G.nodes())
        if len(nodes) <= 1:
            return nodes
        endpoints = nodes[: min(10, len(nodes))]

    H = _inverse_weight(G)
    endpoint_set = set(endpoints)

    if len(endpoints) <= _MAX_EXACT_ENDPOINTS:
        # 每个端点一次单源 = 覆盖全部端点对，远快于逐对 shortest_path
        return _best_path_from_sources(G, H, endpoints, endpoint_set)

    seeds = _select_spatial_seeds(endpoints, _MAX_DIJKSTRA_SOURCES)
    best = _best_path_from_sources(G, H, seeds, endpoint_set)
    # 2-sweep：从当前最优路径端点再扩一次，逼近加权直径
    if best:
        tip = best[-1]
        refined = _best_path_from_sources(G, H, [tip], endpoint_set)
        if refined:
            if _path_raw_weight(G, refined) >= _path_raw_weight(G, best):
                best = refined
            other = refined[0] if refined[-1] == tip else refined[-1]
            refined2 = _best_path_from_sources(G, H, [other], endpoint_set)
            if refined2 and _path_raw_weight(G, refined2) >= _path_raw_weight(G, best):
                best = refined2
    return best


def ensure_simple_path_on_graph(G: nx.Graph, path: List[Node]) -> List[Node]:
    """
    将「主根路径」规范为图上的简单路径（无重复顶点），相邻顶点在原图中有边相连。
    用于未开启分叉竞争模式时，避免行走中出现回路或缺口导致一级根在视觉上非单链。

    - 先去重连续重复点，再按「若顶点已出现则截断到首次出现」去掉行走中的环。
    - 若相邻两点无边，用加权最短路拼接（与 find_longest_path 使用同一 weight）。
    """
    if not path or G.number_of_nodes() == 0:
        return list(path) if path else []
    # 连续重复
    nodes: List[Node] = [path[0]]
    for x in path[1:]:
        if x != nodes[-1]:
            nodes.append(x)
    # 行走去环 → 简单顶点序列
    out: List[Node] = []
    for x in nodes:
        if x in out:
            out = out[: out.index(x) + 1]
        else:
            out.append(x)
    if len(out) < 2:
        return out
    # 用图边或最短路缝合
    chain: List[Node] = [out[0]]
    for i in range(1, len(out)):
        target = out[i]
        u = chain[-1]
        if u == target:
            continue
        if G.has_edge(u, target):
            chain.append(target)
            continue
        try:
            seg = nx.shortest_path(G, u, target, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            break
        chain.extend(seg[1:])
    # 缝合后可能又出现重复顶点，再压一次
    out2: List[Node] = []
    for x in chain:
        if x in out2:
            out2 = out2[: out2.index(x) + 1]
        else:
            out2.append(x)
    # 去掉连续重复
    if not out2:
        return [path[0]]
    final = [out2[0]]
    for x in out2[1:]:
        if x != final[-1]:
            final.append(x)
    return final if len(final) >= 1 else [path[0]]


def get_main_root_edges(path: List[Node], G: nx.Graph) -> Set[Tuple[Node, Node]]:
    edges = set()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if G.has_edge(u, v):
            edges.add((min(u, v), max(u, v)))
    return edges


def _best_score_path_from_sources(
    score_graph: nx.Graph,
    H: nx.Graph,
    sources: Iterable[Hashable],
    targets: Optional[Set[Hashable]] = None,
) -> List[Hashable]:
    """H 为 score 的反权图；在目标中挑 score 之和最大的路径。"""
    best_path: List[Hashable] = []
    best_score = -1.0
    for s in sources:
        if s not in H:
            continue
        try:
            paths = nx.single_source_dijkstra_path(H, s, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        for t, path in paths.items():
            if t == s or len(path) < 2:
                continue
            if targets is not None and t not in targets:
                continue
            sc = float(
                sum(
                    score_graph[path[i]][path[i + 1]].get("score", 0.0)
                    for i in range(len(path) - 1)
                )
            )
            if sc > best_score:
                best_score = sc
                best_path = list(path)
    return best_path


def find_representative_path_in_primary_subgraph(
    G: nx.Graph,
    E_primary_set: Set[Tuple[Node, Node]],
    subtree_S: Optional[Dict[Tuple[Node, Node], float]] = None,
    lambda1: float = 1.0,
    lambda2: float = 0.5,
    d_attr: str = "diameter",
) -> List[Node]:
    """在 E_primary 子图上按 λ1·D + λ2·S 近似最长简单路径（用于可视化/API）。

    用 score 反权做 Dijkstra（正权），再按 score 之和择优；大分量仅用少量空间种子，
    避免节点全组合 O(N²)。
    """
    if not E_primary_set:
        return []
    Gp = nx.Graph()
    for u, v in E_primary_set:
        if G.has_edge(u, v):
            D = float(G[u][v].get(d_attr, G[u][v].get("diameter", 1.0)))
            key = (min(u, v), max(u, v))
            S = float(subtree_S.get(key, 0.0)) if subtree_S else 0.0
            score = max(1e-9, lambda1 * D + lambda2 * S)
            Gp.add_edge(u, v, score=score, weight=1.0 / score)
    if Gp.number_of_edges() == 0:
        return []

    best_path: List[Node] = []
    best_score = -1.0
    for comp in nx.connected_components(Gp):
        sub = Gp.subgraph(comp)
        nodes = list(sub.nodes())
        if len(nodes) < 2:
            continue

        leaves = [n for n in nodes if sub.degree(n) == 1]
        targets = set(leaves) if len(leaves) >= 2 else set(nodes)
        if len(targets) <= _MAX_EXACT_ENDPOINTS:
            sources: List[Node] = list(targets)
        else:
            sources = _select_spatial_seeds(list(targets), _MAX_DIJKSTRA_SOURCES)

        cand = _best_score_path_from_sources(sub, sub, sources, targets)
        if cand and len(cand) >= 2:
            tip = cand[-1]
            refined = _best_score_path_from_sources(sub, sub, [tip], targets)
            if refined:
                cand = refined
            sc = float(
                sum(sub[a][b].get("score", 0.0) for a, b in zip(cand[:-1], cand[1:]))
            )
            if sc > best_score:
                best_score = sc
                best_path = list(cand)

    if best_path:
        return best_path
    nlist = list(Gp.nodes())
    return [nlist[0]] if nlist else []
