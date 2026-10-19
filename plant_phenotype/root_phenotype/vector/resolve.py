"""Local junction hypotheses and rooted assembly without a global path graph."""
from __future__ import annotations

from collections import deque
import copy
import itertools
import math

from .models import JunctionAlternative, JunctionDecision, ParentAttachment, RootNode, VectorRoot, validate_document
from .vectorize import simplify_nodes


def _pair_cost(a, b):
    dot = a.tangent[0]*b.tangent[0] + a.tangent[1]*b.tangent[1]
    valid = math.hypot(*a.tangent) > .5 and math.hypot(*b.tangent) > .5
    turn = math.acos(max(-1.0, min(1.0, -dot))) if valid else math.pi
    values = {'direction': turn/math.pi, 'turn_deg': math.degrees(turn)}
    total, weight = .60*values['direction'], .60
    if a.diameter_px and b.diameter_px:
        values['diameter'] = min(1.0, abs(math.log(a.diameter_px/b.diameter_px))/math.log(2))
        total += .25*values['diameter']
        weight += .25
    if a.field_axis is not None and b.field_axis is not None:
        quality = min(a.coherence, b.coherence)
        values['field'] = 1-abs(a.field_axis[0]*b.field_axis[0] + a.field_axis[1]*b.field_axis[1])
        total += .15*quality*values['field']
        weight += .15*quality
    values['coverage'] = weight
    return total/weight, values


def propose_decisions(snapshot, config):
    ports = {p.id: p for p in snapshot.ports}
    decisions = []
    for junction in snapshot.junctions:
        ids = junction.port_ids
        candidates = []
        if len(ids) == 4:
            for i, pairing in enumerate((((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))):
                pairs = [[ids[a], ids[b]] for a, b in pairing]
                costs = [_pair_cost(ports[a], ports[b])[0] for a, b in pairs]
                candidates.append(JunctionAlternative(f'{junction.id}:x{i}', 'crossing', pairs, sum(costs)/2,
                                                      f'穿越：{pairs[0][0]} ↔ {pairs[0][1]}；{pairs[1][0]} ↔ {pairs[1][1]}'))
        kind = 'continuation' if len(ids) == 2 else 'branch'
        for i, (a, b) in enumerate(itertools.combinations(ids, 2)):
            cost, terms = _pair_cost(ports[a], ports[b])
            candidates.append(JunctionAlternative(f'{junction.id}:b{i}', kind, [[a, b]], cost,
                                                  f'{"延续" if kind == "continuation" else "分叉，主轴延续"}：{a} ↔ {b}', terms))
        # Crowded junctions remain reviewable without an exponential matching search.
        candidates.sort(key=lambda a: (a.cost, a.id))
        candidates = candidates[:12]
        decision = JunctionDecision(junction.id, list(ids), candidates)
        if len(ids) in (2, 3) and candidates:
            best = candidates[0]
            margin = candidates[1].cost-best.cost if len(candidates) > 1 else 1.0
            decision.score_terms = {**best.score_terms, 'best_cost': best.cost, 'margin': margin}
            t_contact = False
            if len(ids) == 3:
                used = set(best.pairs[0])
                side = ports[next(p for p in ids if p not in used)]
                a = ports[best.pairs[0][0]]
                # A perpendicular touching tip is indistinguishable from a T fork.
                t_contact = abs(a.tangent[0]*side.tangent[0] + a.tangent[1]*side.tangent[1]) < .25
            if best.cost <= config.max_pair_cost and margin >= config.min_pair_margin and best.score_terms.get('turn_deg', 180) <= 75 and not t_contact:
                decision.chosen, decision.state = best.id, 'accepted'
        # Four ports alone cannot establish crossing versus two adjacent forks.
        decision.alternatives.append(JunctionAlternative(f'{junction.id}:stop', 'stop', cost=1.0, label='在此停止，不建立连接'))
        decisions.append(decision)
    return decisions


def _connector(junction, start, end):
    """Shortest skeleton walk inside this junction; never cut through background."""
    first, last = (round(start[0]), round(start[1])), (round(end[0]), round(end[1]))
    if first == last:
        return [first]
    pixels = {tuple(p) for p in junction.pixels}
    queue, previous = deque([first]), {first: None}
    while queue:
        point = queue.popleft()
        if point == last:
            break
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                other = (point[0]+dx, point[1]+dy)
                if not (dx or dy) or other not in pixels or other in previous:
                    continue
                if dx and dy and ((point[0]+dx, point[1]) in pixels or (point[0], point[1]+dy) in pixels):
                    continue
                previous[other] = point
                queue.append(other)
    if last not in previous:
        raise ValueError(f'交汇 {junction.id} 内部骨架不连通')
    path, current = [], last
    while current is not None:
        path.append(current)
        current = previous[current]
    return list(reversed(path))


def _junction_node(junction_id, point):
    x, y = point
    return RootNode(f'{junction_id}:n{x}_{y}', float(x), float(y), None, 'missing', ['junction'],
                    {'junction_id': junction_id, 'pixel': [x, y]})


def rebuild_document(document, check_cancel=None):
    """Return a new assembly. Accepted choices and manual edits are authoritative."""
    doc = document.clone()
    snapshot = doc.analysis_snapshot
    excluded = set(doc.excluded_segment_ids)
    segments = {s.id: s for s in snapshot.segments if s.id not in excluded}
    ports = {p.id: p for p in snapshot.ports}
    junctions = {j.id: j for j in snapshot.junctions}
    decisions = {d.id: d for d in doc.junction_decisions}
    accepted = {}
    for decision in doc.junction_decisions:
        if decision.state == 'accepted':
            accepted[decision.id] = next((a for a in decision.alternatives if a.id == decision.chosen), None)
    # Rebuilding should not repeat stale diagnostics from a previous assembly.
    doc.diagnostics = [s for s in doc.diagnostics if not s.startswith(('基部 ', '接续 ', '闭环 ', '人工 '))]
    for mark in doc.base_marks:
        mark.root_id = None
    doc.roots = []
    assigned = set()
    node_ids_by_list = {}

    def other_end(segment, side):
        return segment.end_port if side == 0 else segment.start_port

    def outgoing(incoming):
        port = ports[incoming]
        option = accepted.get(port.junction_id)
        if option is None or option.kind == 'stop':
            return None, [], option
        pair = next((pair for pair in option.pairs if incoming in pair), None)
        if pair is None:
            return None, [], option
        continuation = pair[1] if pair[0] == incoming else pair[0]
        sides = [p for p in junctions[port.junction_id].port_ids if p not in pair] if option.kind == 'branch' else []
        return continuation, sides, option

    def entering(port_id):
        p = ports[port_id]
        segment = segments.get(p.segment_id)
        return (p.segment_id, 0 if segment.start_port == port_id else 1) if segment is not None else None

    def claim_tree(start):
        result, work = set(), [start]
        while work:
            sid, side = work.pop()
            if sid in result or sid not in segments:
                continue
            result.add(sid)
            port_id = other_end(segments[sid], side)
            if port_id:
                continuation, children, _ = outgoing(port_id)
                for pid in ([continuation] if continuation else []) + children:
                    entry = entering(pid)
                    if entry:
                        work.append(entry)
        return result

    endpoints = []
    for segment in segments.values():
        if segment.closed:
            continue
        for side, port_id, node in ((0, segment.start_port, segment.nodes[0]), (1, segment.end_port, segment.nodes[-1])):
            if port_id is None:
                endpoints.append((segment.id, side, node))
    starts, claims = {}, {}
    for mark in doc.base_marks:
        candidates = sorted(((math.hypot(n.x-mark.x, n.y-mark.y), sid, side, n) for sid, side, n in endpoints), key=lambda v: (v[0], v[1], v[2]))
        if not candidates or candidates[0][0] > doc.config.base_snap_distance:
            doc.diagnostics.append(f'基部 {mark.id} 未靠近可见端点，请在根基部端点重新标记')
            continue
        distance, sid, side, node = candidates[0]
        closest = min((math.hypot(n.x-mark.x, n.y-mark.y) for s in segments.values() for n in s.nodes), default=distance)
        if distance > max(3.0, closest+2.0):
            doc.diagnostics.append(f'基部 {mark.id} 位于根中段，方向未确定；请改为端点标记')
            continue
        if len(candidates) > 1 and candidates[1][0]-distance < 1.0:
            doc.diagnostics.append(f'基部 {mark.id} 同时靠近多个端点，归属待确认')
            continue
        starts[mark.id] = (sid, side)
        claims[mark.id] = claim_tree((sid, side))
    conflicts = set()
    for first, second in itertools.combinations(starts, 2):
        if claims[first] & claims[second]:
            conflicts.update((first, second))
    for mid in sorted(conflicts):
        doc.diagnostics.append(f'基部 {mid} 与其他基部争用同一根段，未按处理顺序强行归属')

    def append(nodes, point):
        point = copy.deepcopy(point)
        if nodes and math.hypot(nodes[-1].x-point.x, nodes[-1].y-point.y) < 1e-10:
            nodes[-1].tags = list(dict.fromkeys([*nodes[-1].tags, *point.tags]))
            return nodes[-1]
        identifiers = node_ids_by_list.setdefault(id(nodes), set())
        if point.id in identifiers:
            point.id = f'{point.id}:v{len(nodes)}'
        nodes.append(point)
        identifiers.add(point.id)
        return point

    def trace(start, root_id, *, base_id=None, attachment=None, prefix=None, known=False):
        root = VectorRoot(root_id, parent_attachment=attachment, base_mark_id=base_id,
                          direction='base_to_tip' if known else 'unknown', review_state='auto' if known else 'pending')
        pending_children = []
        if prefix:
            for node in prefix:
                append(root.nodes, node)
        sid, side = start
        while sid in segments and sid not in assigned:
            if check_cancel:
                check_cancel()
            segment = segments[sid]
            assigned.add(sid)
            root.source_segment_ids.append(sid)
            chain = segment.nodes if side == 0 else list(reversed(segment.nodes))
            for node in chain:
                append(root.nodes, node)
            if segment.closed:
                root.completeness, root.review_state = 'loop', 'pending'
                root.quality_flags.append('closed_loop')
                break
            incoming = other_end(segment, side)
            if incoming is None:
                last = root.nodes[-1]
                if last.x <= 0 or last.y <= 0 or last.x >= doc.source.width-1 or last.y >= doc.source.height-1:
                    root.completeness = 'boundary'
                break
            continuation, children, option = outgoing(incoming)
            if continuation is None:
                root.completeness = 'pending'
                root.review_state = 'pending'
                root.quality_flags.append('unresolved_junction')
                break
            next_entry = entering(continuation)
            if next_entry is None:
                root.completeness = 'pending'
                root.quality_flags.append('excluded_continuation')
                break
            if next_entry[0] in assigned:
                root.completeness, root.review_state = 'loop', 'pending'
                root.quality_flags.append('cycle_or_conflict')
                break
            first, last = ports[incoming], ports[continuation]
            junction = junctions[first.junction_id]
            path = _connector(junction, (first.x, first.y), (last.x, last.y))
            path_nodes = [root.nodes[-1]]
            for point in path[1:]:
                path_nodes.append(append(root.nodes, _junction_node(junction.id, point)))
            if option.kind == 'branch' and known:
                for child_port in children:
                    child_entry = entering(child_port)
                    if child_entry is None or child_entry[0] in assigned:
                        continue
                    child_p = ports[child_port]
                    anchor = min(path_nodes, key=lambda n: (n.x-child_p.x)**2+(n.y-child_p.y)**2)
                    if 'attachment' not in anchor.tags:
                        anchor.tags.append('attachment')
                    child_path = _connector(junction, (anchor.x, anchor.y), (child_p.x, child_p.y))
                    child_prefix = [copy.deepcopy(anchor)] + [_junction_node(junction.id, p) for p in child_path[1:]]
                    pending_children.append((child_entry, ParentAttachment(root.id, anchor.id), child_prefix))
            sid, side = next_entry
        if len(root.nodes) >= 2:
            doc.roots.append(root)
            for entry, child_attachment, child_prefix in pending_children:
                if entry[0] not in assigned:
                    trace(entry, f'root_{entry[0]}', attachment=child_attachment, prefix=child_prefix, known=True)
        return root

    for mark in doc.base_marks:
        if mark.id not in starts or mark.id in conflicts:
            continue
        start = starts[mark.id]
        if start[0] in assigned:
            continue
        root = trace(start, f'root_base_{mark.id}', base_id=mark.id, known=True)
        if root in doc.roots:
            mark.root_id = root.id
    # Start at real tips first, so accepted crossings also form continuous
    # unassigned roots while never becoming accidental biological parents.
    rest = [(sid, side) for sid, side, _ in endpoints]
    rest += [(sid, 0) for sid in segments]
    for start in rest:
        if start[0] not in assigned:
            trace(start, f'root_{start[0]}')
    # Restore protected geometry before simplification so its source anchor
    # cannot disappear. The common editing helper then handles parent/direction.
    for root in doc.roots:
        previous = [r for r in document.roots if r.id == root.id or set(r.source_segment_ids).intersection(root.source_segment_ids)]
        protected = [n for r in previous for n in r.nodes if 'manual_locked' in n.tags]
        for node in root.nodes:
            matches = [old for old in protected if old.id == node.id or (old.source_anchor is not None and old.source_anchor == node.source_anchor)]
            if matches:
                old = matches[0]
                node.x, node.y, node.diameter_px = old.x, old.y, old.diameter_px
                node.diameter_source = old.diameter_source
                node.tags = list(dict.fromkeys([*node.tags, *old.tags]))
        root.nodes = simplify_nodes(root.nodes, doc.config)
    if document.roots:
        from .editing import _preserve_edits
        _preserve_edits(document, doc)
    validate_document(doc)
    from .quality import review_primary_paths
    review_primary_paths(doc)
    return doc
