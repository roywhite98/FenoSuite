"""Seed-scoped repair and main-axis-first assembly for interactive analysis.

Clicks choose components, never root orientation. The segment graph is only an
analysis index; the persisted result remains vector roots with parent attachments.
"""
from __future__ import annotations

import math
import cv2
import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .models import BaseMark, JunctionAlternative, VectorAnalysisConfig
from .resolve import _pair_cost
from .vectorize import skeleton_adjacency
from ..core.connect_roots import _endpoint_direction


def normalize_selections(points, mask):
    result = []
    for i, value in enumerate(points):
        x, y = (value['x'], value['y']) if isinstance(value, dict) else value
        x, y = float(x), float(y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('选点坐标必须是有限数值')
        col, row = int(round(x)), int(round(y))
        if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]) or not mask[row, col]:
            raise ValueError('请点击根内的前景像素；背景位置不能选定根系')
        result.append({'id': f'seed{i+1}', 'x': x, 'y': y})
    if not result:
        raise ValueError('请先点击要识别的根，根内任意位置均可')
    return result


def prepare_selection(data, points, config, check_cancel=None):
    binary = np.asarray(data)
    if binary.ndim == 3:
        binary = np.max(binary[:, :, :3], axis=2)
    binary = np.ascontiguousarray(binary > 0, dtype=np.uint8)
    points = normalize_selections(points, binary)
    original = binary.copy()
    notes, bridges = [], 0

    def selected_labels(labels):
        return {int(labels[round(p['y']), round(p['x'])]) for p in points} - {0}

    # Keep all fragments as candidates, but only grow from a selected component.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= max(2, config.min_component_area)
    keep[list(selected_labels(labels))] = True
    keep[0] = False
    binary = np.ascontiguousarray(keep[labels], dtype=np.uint8)
    for _ in range(max(0, count-2) if config.connect_gap_px else 0):
        if check_cancel:
            check_cancel()
        _, labels = cv2.connectedComponents(binary, connectivity=8)
        chosen = selected_labels(labels)
        skel = skeletonize(binary.astype(bool))
        adjacency = skeleton_adjacency(skel)
        ends = sorted(p for p, neighbors in adjacency.items() if len(neighbors) == 1)
        if len(ends) < 2:
            break
        radii = cv2.distanceTransform(np.pad(binary, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        directions = [_endpoint_direction(skel, y, x) for x, y in ends]
        candidates = []
        # Skeleton tips retreat inside thick cut ends. The user limit measures
        # missing foreground, not the larger distance between skeleton tips.
        search_distance = config.connect_gap_px + 2*float(radii.max())
        for a, b in cKDTree(ends).query_pairs(search_distance):
            ax, ay = ends[a]
            bx, by = ends[b]
            ca, cb = int(labels[ay, ax]), int(labels[by, bx])
            if ca == cb or not ({ca, cb} & chosen):
                continue
            gap = math.hypot(bx-ax, by-ay)
            if gap == 0:
                continue
            dr, dc = (by-ay)/gap, (bx-ax)/gap
            da, db = directions[a], directions[b]
            if da[0]*dr+da[1]*dc < .906 or -db[0]*dr-db[1]*dc < .906:
                continue
            ra, rb = float(radii[ay, ax]), float(radii[by, bx])
            if max(ra, rb) > 2.5*max(1, min(ra, rb)):
                continue
            samples = max(2, int(math.ceil(gap))+1)
            xs = np.rint(np.linspace(ax, bx, samples)).astype(int)
            ys = np.rint(np.linspace(ay, by, samples)).astype(int)
            missing_length = np.count_nonzero(binary[ys, xs] == 0)*gap/(samples-1)
            if missing_length > config.connect_gap_px:
                continue
            candidates.append((gap, a, b, ca, cb))
        used, added = set(), 0
        for _, a, b, ca, cb in sorted(candidates):
            if a in used or b in used:
                continue
            # Never cross a third component while drawing a bridge.
            line = np.zeros_like(binary)
            cv2.line(line, ends[a], ends[b], 1, 1)
            if set(np.unique(labels[line > 0])) - {0, ca, cb}:
                continue
            cv2.line(binary, ends[a], ends[b], 1, 1)
            used.update((a, b))
            added += 1
        bridges += added
        if not added:
            break
    _, labels = cv2.connectedComponents(binary, connectivity=8)
    selected = np.isin(labels, list(selected_labels(labels)))
    # Fill only small enclosed holes; do not close gaps between neighbouring roots.
    holes, count = ndimage.label(~selected)
    sizes = np.bincount(holes.ravel())
    small = sizes <= 16
    small[0] = False
    small[np.unique(np.concatenate((holes[0], holes[-1], holes[:, 0], holes[:, -1])))] = False
    selected |= small[holes]
    synthetic = selected & ~original.astype(bool)
    notes.append(f'仅分析选中的根系；连接了 {bridges} 处短断口')
    return selected.astype(np.uint8), synthetic, points, notes


def prune_spurs(skeleton, limit):
    """Remove short terminal chains using the vectorizer's corner-safe adjacency."""
    skel = skeleton.copy()
    if limit <= 0:
        return skel, 0
    adjacency = skeleton_adjacency(skel)
    removed = set()
    for leaf, neighbors in adjacency.items():
        if len(neighbors) != 1:
            continue
        path, previous, current, length = [leaf], leaf, neighbors[0], 0.0
        while True:
            length += math.dist(previous, current)
            if length > limit:
                break
            if len(adjacency[current]) >= 3:
                removed.update(path)
                break
            following = [p for p in adjacency[current] if p != previous]
            if len(following) != 1:
                break
            path.append(current)
            previous, current = current, following[0]
    for x, y in removed:
        skel[y, x] = False
    return skel, len(removed)


def _best_tree_axis(graph, tips, evidence, penalties, check_cancel=None):
    """Exact endpoint path with local transition costs, without endpoint sampling.

    For each subtree retain its best route to a real tip, conditional on entering
    from its parent. Combining two such routes considers every possible axis.
    Only endpoint references are retained; the winning path is reconstructed once.
    """
    parent, order = {tips[0]: None}, [tips[0]]
    for node in order:
        if check_cancel:
            check_cancel()
        for other in graph[node]:
            if other not in parent:
                parent[other] = node
                order.append(other)
    tip_set, down = set(tips), {}
    best_score, best_ends = None, None

    def combine(node, first, second, score):
        if node[0] != 'j':
            return score
        pair = frozenset((first, second))
        return score-penalties[(node, pair)]

    for node in reversed(order):
        if check_cancel:
            check_cancel()
        children = [n for n in graph[node] if parent.get(n) == node]
        weight = evidence.get(node, 0.0)
        if not children:
            if node in tip_set:
                down[node] = (weight, node)
            continue
        # Best route entering this subtree from the parent.
        for child in children:
            child_score, end = down[child]
            score = child_score+weight
            if parent[node] is not None:
                score = combine(node, parent[node], child, score)
            if node not in down or score > down[node][0]:
                down[node] = (score, end)
        # A path either ends at this real tip or joins two child subtrees.
        routes = [(n, *down[n]) for n in children]
        if node in tip_set:
            routes.insert(0, (None, 0.0, node))
        for i, (first, a_score, a_end) in enumerate(routes):
            for second, b_score, b_end in routes[i+1:]:
                score = a_score+b_score+weight
                if first is not None:
                    score = combine(node, first, second, score)
                if best_score is None or score > best_score:
                    best_score, best_ends = score, (a_end, b_end)
    if best_ends is None:
        return None, None
    return nx.shortest_path(graph, *best_ends), best_score


def infer_selected_axes(snapshot, decisions, check_cancel=None, config=None):
    """Choose a complete primary axis, then orient each remaining branch outward."""
    config = config or VectorAnalysisConfig()
    graph = nx.Graph()
    segments = {s.id: s for s in snapshot.segments}
    ports = {p.id: p for p in snapshot.ports}
    decision_by_id = {d.id: d for d in decisions}
    ends, port_at, evidence, lengths, side_evidence = {}, {}, {}, {}, {}
    for segment in snapshot.segments:
        if segment.closed:
            continue
        sid = ('s', segment.id)
        length = sum(math.dist((a.x, a.y), (b.x, b.y)) for a, b in zip(segment.nodes, segment.nodes[1:]))
        lengths[sid] = length
        widths = [n.diameter_px for n in segment.nodes if n.diameter_px]
        diameter = float(np.median(widths)) if widths else 1
        # Strong diameter weighting prevents many pixels of a thin lateral
        # from outweighing a shorter thick axis. This is a ranking heuristic,
        # not a volume measurement, and never vetoes a path at one junction.
        evidence[sid] = length * diameter**4
        side_evidence[sid] = length * math.sqrt(diameter)
        for side, pid, node in ((0, segment.start_port, segment.nodes[0]), (1, segment.end_port, segment.nodes[-1])):
            end = ('j', ports[pid].junction_id) if pid else ('e', segment.id, side)
            graph.add_edge(sid, end, weight=max(.001, length/2),
                           axis_cost=max(.001, length/(2*diameter**2)))
            if pid:
                port_at[(end, sid)] = pid
            else:
                ends[end] = node

    # Cache local evidence once, rather than recomputing it for every endpoint
    # pair. A junction may affect only its sampling window on either side. Half
    # a segment per end bounds overlap when junctions are densely packed.
    penalties = {}
    for junction in snapshot.junctions:
        key = ('j', junction.id)
        neighbours = sorted(graph[key]) if key in graph else []
        for i, first in enumerate(neighbours):
            for second in neighbours[i+1:]:
                a, b = ports[port_at[(key, first)]], ports[port_at[(key, second)]]
                cost, _ = _pair_cost(a, b)
                pair = frozenset((first, second))
                local_support = 0.0
                for segment, port in ((first, a), (second, b)):
                    window = min(lengths[segment]/2, max(4.0, config.sample_length_factor*(port.diameter_px or 2.0)))
                    local_support += evidence[segment]*window/max(lengths[segment], 1e-9)
                penalties[(key, pair)] = cost*local_support

    def path_score(path):
        support = sum(evidence.get(n, 0) for n in path)
        penalty = 0.0
        for i, junction in enumerate(path[1:-1], 1):
            if junction[0] != 'j':
                continue
            pair = frozenset((path[i-1], path[i+1]))
            penalty += penalties[(junction, pair)]
        # Shared upstream support cancels when comparing alternative tips;
        # it is never discounted again by downstream junction count.
        return support-penalty

    def choose(junction, first, second):
        decision = decision_by_id[junction[1]]
        pair = [port_at[(junction, first)], port_at[(junction, second)]]
        kind = 'continuation' if len(decision.port_ids) == 2 else 'branch'
        option = next((a for a in decision.alternatives if a.kind == kind and len(a.pairs) == 1 and set(a.pairs[0]) == set(pair)), None)
        if option is None:
            option = JunctionAlternative(f'{decision.id}:axis', kind, [pair], _pair_cost(*(ports[p] for p in pair))[0])
            decision.alternatives.insert(0, option)
        decision.chosen, decision.state = option.id, 'accepted'
        decision.score_terms = {**option.score_terms, 'main_axis_first': 1.0,
                                'local_penalty': penalties[(junction, frozenset((first, second)))]}

    marks, notes = [], []
    for component in nx.connected_components(graph):
        if check_cancel:
            check_cancel()
        sub = graph.subgraph(component)
        tips = sorted(n for n in component if n in ends)
        if len(tips) < 2:
            notes.append('存在无完整端点的闭环，保留待确认根段')
            continue
        def base_rank(endpoint):
            segment = segments[endpoint[1]]
            nodes = segment.nodes if endpoint[2] == 0 else list(reversed(segment.nodes))
            widths = [n.diameter_px for n in nodes[:max(3, len(nodes)//3)] if n.diameter_px]
            width = float(np.median(widths)) if widths else 0
            return (round(width, 1), -ends[endpoint].y, -ends[endpoint].x)

        ranked = sorted(tips, key=base_rank, reverse=True)
        tip_widths = [base_rank(t)[0] for t in ranked]
        # Two similarly thick ends, clearly separated from all other tips,
        # jointly identify a plausible complete axis. A single widest tip may
        # merely be a broken stub or swollen junction and cannot anchor it.
        anchors = (ranked[:2] if len(ranked) >= 3 and tip_widths[2] > 0 and
                   tip_widths[1] >= 1.5*tip_widths[2] and
                   tip_widths[0] <= 1.25*tip_widths[1] else [])
        is_tree = nx.is_tree(sub)
        if is_tree and anchors:
            best = nx.shortest_path(sub, *anchors)
            best_score = path_score(best)
        elif is_tree:
            best, best_score = _best_tree_axis(sub, tips, evidence, penalties, check_cancel)
        else:
            # Cycles have no unique path; keep a bounded candidate search and
            # report the loop below so its inferred ancestry can be reviewed.
            sources = (anchors[:1] if anchors else
                       tips if len(tips) <= 24 else [tips[i] for i in np.linspace(0, len(tips)-1, 12, dtype=int)])
            best, best_score = None, None
            for start in sources:
                if check_cancel:
                    check_cancel()
                # A thin shortcut across a loop must not displace the thicker
                # axis solely because it is shorter in pixel distance.
                paths = nx.single_source_dijkstra_path(sub, start, weight='axis_cost')
                for end in (anchors[1:] if anchors else tips):
                    if start == end:
                        continue
                    path = paths[end]
                    score = path_score(path)
                    if best_score is None or score > best_score:
                        best, best_score = path, score
        # Estimate the proximal end from the terminal segment's median width;
        # ties use upper/left image coordinates, independent of click location.
        if base_rank(best[-1]) > base_rank(best[0]):
            best.reverse()
        base = ends[best[0]]
        marks.append(BaseMark(f'auto{len(marks)+1}', base.x, base.y))
        main = set(best)
        for i, n in enumerate(best[1:-1], 1):
            if n[0] == 'j':
                choose(n, best[i-1], best[i+1])
        # A shortest-path tree gives a unique parent at each off-axis junction.
        distances, paths = nx.single_source_dijkstra(sub, best[0], weight='weight')
        for n in sorted(component, key=lambda n: (distances[n], n)):
            if n[0] != 'j' or n in main:
                continue
            incoming = paths[n][-2]
            children = [v for v in sub[n] if len(paths[v]) > 1 and paths[v][-2] == n]
            if not children:
                continue
            # At a side-root fork favour a smooth continuation, supported by size.
            def cost(child):
                a, b = ports[port_at[(n, incoming)]], ports[port_at[(n, child)]]
                return _pair_cost(a, b)[0] - .05*math.log1p(side_evidence.get(child, 0))
            choose(n, incoming, min(children, key=lambda v: (cost(v), v)))
        if not is_tree:
            notes.append('选中根系含闭环或粘连，请核对交汇解释')
    notes.append('主根按长度、粗细和连续性自动推断；基部粗细相近时按图像上端定向，可人工反向')
    return marks, notes
