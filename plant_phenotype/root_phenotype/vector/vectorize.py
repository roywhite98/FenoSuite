"""Compress consistent skeleton adjacency into segments and junction ports."""
from __future__ import annotations

from collections import deque
import math
import numpy as np

from .features import RasterFeatures
from .models import AnalysisSnapshot, JunctionRegion, Port, RootNode, Segment, VectorAnalysisConfig


def skeleton_adjacency(skeleton):
    """8-neighbours, omitting diagonal shortcuts around an orthogonal step."""
    coordinates = {(int(x), int(y)) for y, x in np.argwhere(skeleton)}
    adjacency = {}
    for x, y in sorted(coordinates, key=lambda p: (p[1], p[0])):
        neighbours = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (x + dx, y + dy)
                if not (dx or dy) or q not in coordinates:
                    continue
                if dx and dy and ((x + dx, y) in coordinates or (x, y + dy) in coordinates):
                    continue
                neighbours.append(q)
        adjacency[(x, y)] = neighbours
    return adjacency


def _unit(vector):
    length = math.hypot(*vector)
    return (vector[0] / length, vector[1] / length) if length else (0.0, 0.0)


def _port_evidence(nodes, features, config):
    outside = [n for n in nodes[1:] if n.diameter_source == 'measured']
    if not outside:
        outside = nodes[1:]
    diameters = [n.diameter_px for n in outside if n.diameter_px is not None]
    diameter = float(np.median(diameters[:12])) if diameters else None
    limit = max(4.0, config.sample_length_factor * (diameter or 2.0))
    selected = []
    distance = 0.0
    previous = nodes[0]
    for point in nodes[1:]:
        distance += math.hypot(point.x - previous.x, point.y - previous.y)
        selected.append(point)
        previous = point
        if distance >= limit:
            break
    target = selected[-1] if selected else nodes[-1]
    tangent = _unit((target.x - nodes[0].x, target.y - nodes[0].y))
    # The outside half of the sample avoids the mixed signal at the junction.
    measured = [n for n in selected[len(selected)//2:] if n.diameter_source == 'measured']
    widths = [n.diameter_px for n in measured if n.diameter_px is not None]
    if widths:
        diameter = float(np.median(widths))
    samples = [(int(round(n.y)), int(round(n.x))) for n in measured]
    samples = [(y, x) for y, x in samples if features.field_valid[y, x]]
    if not samples:
        return tangent, diameter, None, 0.0
    weights = np.array([features.coherence[y, x] for y, x in samples])
    c = float(np.average([features.cos2[y, x] for y, x in samples], weights=weights))
    s = float(np.average([features.sin2[y, x] for y, x in samples], weights=weights))
    theta = 0.5 * math.atan2(s, c)
    axis = (math.cos(theta), math.sin(theta))
    quality = float(np.mean(weights) * min(1.0, math.hypot(c, s)))
    # Geometry/field disagreement reduces the field's weight in the resolver.
    quality *= abs(axis[0] * tangent[0] + axis[1] * tangent[1])
    return tangent, diameter, axis, quality


def vectorize(features: RasterFeatures, config: VectorAnalysisConfig, *, check_cancel=None):
    adjacency = skeleton_adjacency(features.skeleton)
    critical = {p for p, neighbours in adjacency.items() if len(neighbours) >= 3}
    region_of = {}
    junctions = []
    for point in adjacency:
        if point not in critical or point in region_of:
            continue
        component, queue = [], deque([point])
        region_id = f'j{point[1]}_{point[0]}'
        region_of[point] = region_id
        while queue:
            current = queue.popleft()
            component.append(current)
            for other in adjacency[current]:
                if other in critical and other not in region_of:
                    region_of[other] = region_id
                    queue.append(other)
        mean_x = sum(p[0] for p in component) / len(component)
        mean_y = sum(p[1] for p in component) / len(component)
        anchor = min(component, key=lambda p: (p[0]-mean_x)**2 + (p[1]-mean_y)**2)
        junctions.append(JunctionRegion(region_id, float(anchor[0]), float(anchor[1]), pixels=component))
    by_region = {j.id: j for j in junctions}
    visited = set()
    segments, ports, diagnostics = [], [], []

    def edge(a, b):
        return (a, b) if a < b else (b, a)

    for p in critical:
        for q in adjacency[p]:
            if region_of.get(q) == region_of[p]:
                visited.add(edge(p, q))

    def add_chain(chain, closed=False):
        if len(chain) < 2:
            return
        sid = f's{len(segments):05d}'
        nodes = []
        for i, (x, y) in enumerate(chain):
            nid = f'{sid}:n{i}'
            in_junction = (x, y) in critical
            nodes.append(RootNode(nid, float(x), float(y), None if in_junction else float(features.diameter[y, x]),
                                  'missing' if in_junction else 'measured',
                                  ['junction'] if in_junction else [], {'segment_id': sid, 'node_id': nid}))
        segment = Segment(sid, nodes, closed=closed, quality_flags=['closed_loop'] if closed else [])
        for side, endpoint in ((0, chain[0]), (1, chain[-1])):
            jid = region_of.get(endpoint)
            if jid is None:
                continue
            pid = f'{sid}:p{side}'
            evidence = _port_evidence(nodes if side == 0 else list(reversed(nodes)), features, config)
            ports.append(Port(pid, sid, jid, float(endpoint[0]), float(endpoint[1]), *evidence))
            by_region[jid].port_ids.append(pid)
            if side == 0:
                segment.start_port = pid
            else:
                segment.end_port = pid
        segments.append(segment)

    def trace(first, second):
        chain = [first, second]
        visited.add(edge(first, second))
        previous, current = first, second
        while current not in critical and len(adjacency[current]) == 2:
            following = next(p for p in adjacency[current] if p != previous)
            if edge(current, following) in visited:
                break
            visited.add(edge(current, following))
            chain.append(following)
            previous, current = current, following
            if current == first:
                break
        return chain

    for index, point in enumerate(adjacency):
        if check_cancel and index % 2048 == 0:
            check_cancel()
        if len(adjacency[point]) == 0:
            diagnostics.append(f'保留孤立像素记录 ({point[0]}, {point[1]})，未生成退化根')
        if point in critical or len(adjacency[point]) != 2:
            for other in adjacency[point]:
                if edge(point, other) not in visited:
                    add_chain(trace(point, other))
    # A pure cycle has no endpoint or junction, so requires a second pass.
    for point in adjacency:
        for other in adjacency[point]:
            if edge(point, other) not in visited:
                chain = trace(point, other)
                add_chain(chain, closed=(chain[0] == chain[-1]))
    return AnalysisSnapshot(segments=segments, ports=ports, junctions=junctions), diagnostics


def simplify_nodes(nodes, config):
    """Bound geometry, diameter interpolation and arc-length loss per interval."""
    if len(nodes) <= 2:
        return nodes
    keep = {0, len(nodes)-1}
    keep.update(i for i, node in enumerate(nodes) if any(tag in node.tags for tag in ('attachment', 'junction', 'manual_locked')))
    # Preserve changes between measured and missing/interpolated diameter data.
    for i in range(1, len(nodes)):
        if nodes[i].diameter_source != nodes[i-1].diameter_source:
            keep.update((i-1, i))
    coordinates = np.array([(n.x, n.y) for n in nodes], dtype=float)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1))]
    limits = sorted(keep)
    work = list(zip(limits, limits[1:]))
    while work:
        start, end = work.pop()
        if end-start <= 1:
            continue
        v = coordinates[end] - coordinates[start]
        denom = float(v @ v)
        t = np.clip((coordinates[start+1:end]-coordinates[start]) @ v / max(denom, 1e-12), 0, 1)
        deviation = np.linalg.norm(coordinates[start+1:end] - (coordinates[start] + t[:, None] * v), axis=1)
        errors = deviation / config.simplify_tolerance
        d0, d1 = nodes[start].diameter_px, nodes[end].diameter_px
        if d0 is not None and d1 is not None:
            for offset, i in enumerate(range(start+1, end)):
                if nodes[i].diameter_px is not None:
                    fraction = (arc[i]-arc[start]) / max(arc[end]-arc[start], 1e-12)
                    errors[offset] = max(errors[offset], abs(nodes[i].diameter_px - (d0+(d1-d0)*fraction))/config.diameter_tolerance)
        local_length = arc[end] - arc[start]
        loss = (local_length-math.sqrt(denom)) / max(local_length, 1e-12)
        if float(errors.max()) > 1 or loss > config.length_error_fraction:
            offset = int(np.argmax(errors))
            split = start + 1 + offset if float(errors.max()) > 1 else (start+end)//2
            keep.add(split)
            work.extend(((start, split), (split, end)))
    return [nodes[i] for i in sorted(keep)]
