"""Measurements from the current vector document, never from pixel graph labels."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .models import RootDocument, VectorRoot, validate_document


ROOT_COLUMNS = [
    'root_id', 'order', 'parent_root_id', 'parent_node_id', 'base_mark_id',
    'specimen_id', 'direction', 'review_state', 'completeness', 'quality_flags', 'length_px',
    'mean_diameter_px', 'diameter_coverage', 'node_count', 'direct_child_count',
    'lateral_density_per_px', 'tip_x', 'tip_y', 'is_visible_tip', 'length_mm',
    'mean_diameter_mm', 'lateral_density_per_mm',
]
NODE_COLUMNS = [
    'root_id', 'node_id', 'node_index', 'x', 'y', 'arclength_px',
    'diameter_px', 'diameter_source', 'tags', 'arclength_mm', 'diameter_mm',
]
ATTACHMENT_COLUMNS = [
    'root_id', 'parent_root_id', 'parent_node_id', 'attachment_s_px',
    'attachment_fraction', 'branch_angle_deg', 'acute_angle_deg', 'attachment_s_mm',
]


def _geometry(root: VectorRoot):
    xy = np.asarray([(n.x, n.y) for n in root.nodes], dtype=float)
    lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(lengths)]
    return xy, lengths, arc


def _diameter(root, lengths):
    weighted, covered = 0.0, 0.0
    for left, right, length in zip(root.nodes, root.nodes[1:], lengths):
        if (left.diameter_source in ('measured', 'manual')
                and right.diameter_source in ('measured', 'manual')
                and left.diameter_px is not None and right.diameter_px is not None):
            weighted += float(length) * (left.diameter_px + right.diameter_px) / 2.0
            covered += float(length)
    return (weighted / covered if covered > 0 else None), covered


def _position(xy, arc, distance):
    return np.asarray([np.interp(distance, arc, xy[:, axis]) for axis in (0, 1)])


def _unit_chord(xy, arc, start, end):
    if end - start <= 1e-10:
        return None
    vector = _position(xy, arc, end) - _position(xy, arc, start)
    magnitude = float(np.linalg.norm(vector))
    return vector / magnitude if magnitude > 1e-10 else None


def _angles(parent, child, attachment_s, geometry):
    """Finite arc-window chords; skip the child's immediate mixed junction zone.

    The window is based on measured width, with a 3 px lower bound. At a
    truncated end it uses the available interval. Unknown direction permits
    only an undirected acute angle, not a biological insertion angle.
    """
    pxy, _, parc = geometry[parent.id]
    cxy, _, carc = geometry[child.id]
    widths = [n.diameter_px for n in child.nodes
              if n.diameter_px is not None and n.diameter_source in ('measured', 'manual')]
    window = max(3.0, 2.5 * float(np.median(widths))) if widths else 5.0
    pvector = _unit_chord(pxy, parc, max(0.0, attachment_s - window),
                          min(float(parc[-1]), attachment_s + window))
    # Reserve a quarter of the available child length so short laterals still
    # have a well-defined tangent instead of a zero-length sampling interval.
    start = min(window * 0.25, float(carc[-1]) * 0.25)
    cvector = _unit_chord(cxy, carc, start, min(float(carc[-1]), start + window))
    if pvector is None or cvector is None:
        return None, None
    angle = math.degrees(math.acos(float(np.clip(np.dot(pvector, cvector), -1, 1))))
    directed = angle if parent.direction == child.direction == 'base_to_tip' else None
    return directed, min(angle, 180.0 - angle)


def measure_vectors(document: RootDocument) -> dict[str, pd.DataFrame]:
    """Return roots/nodes/attachments/summary tables with explicit pixel units.

    The ``summary`` all row includes visible geometry of every object. Its
    ``root_count`` counts only objects with known ancestry, while
    ``unassigned_count`` reports the remaining fragments separately. Group
    rows partition by order, review state and completeness. Missing physical
    calibration and unavailable measurements remain null/NaN.
    """
    validate_document(document)
    roots = {root.id: root for root in document.roots}
    geometry = {root.id: _geometry(root) for root in document.roots}
    base_marks = {mark.id: mark for mark in document.base_marks}
    scale = document.source.mm_per_pixel
    rows, nodes, attachments = [], [], []

    def specimen(root):
        while root.parent_attachment is not None:
            root = roots[root.parent_attachment.parent_root_id]
        mark = base_marks.get(root.base_mark_id)
        return mark.specimen_id if mark else None

    for root in document.roots:
        _, lengths, arc = geometry[root.id]
        length = float(arc[-1])
        diameter, covered = _diameter(root, lengths)
        order = document.root_order(root.id)
        child_count = sum(r.parent_attachment is not None
                          and r.parent_attachment.parent_root_id == root.id
                          for r in document.roots)
        attachment = root.parent_attachment
        rows.append(dict(
            root_id=root.id, order=order,
            parent_root_id=attachment.parent_root_id if attachment else None,
            parent_node_id=attachment.parent_node_id if attachment else None,
            base_mark_id=root.base_mark_id, specimen_id=specimen(root),
            direction=root.direction, review_state=root.review_state,
            completeness=root.completeness, quality_flags=';'.join(root.quality_flags), length_px=length,
            mean_diameter_px=diameter, diameter_coverage=covered / length if length else 0.0,
            node_count=len(root.nodes), direct_child_count=child_count,
            lateral_density_per_px=child_count / length if length > 0 and order is not None else None,
            tip_x=root.nodes[-1].x, tip_y=root.nodes[-1].y,
            is_visible_tip=root.completeness == 'complete' and root.direction == 'base_to_tip',
            length_mm=length * scale if scale else None,
            mean_diameter_mm=diameter * scale if scale and diameter is not None else None,
            lateral_density_per_mm=child_count / (length * scale)
            if scale and length > 0 and order is not None else None,
        ))
        for index, node in enumerate(root.nodes):
            nodes.append(dict(
                root_id=root.id, node_id=node.id, node_index=index, x=node.x, y=node.y,
                arclength_px=float(arc[index]), diameter_px=node.diameter_px,
                diameter_source=node.diameter_source, tags=';'.join(node.tags),
                arclength_mm=float(arc[index]) * scale if scale else None,
                diameter_mm=node.diameter_px * scale if scale and node.diameter_px is not None else None,
            ))
        if attachment:
            parent = roots[attachment.parent_root_id]
            parent_arc = geometry[parent.id][2]
            index = next(i for i, n in enumerate(parent.nodes) if n.id == attachment.parent_node_id)
            attach_s = float(parent_arc[index])
            branch_angle, acute_angle = _angles(parent, root, attach_s, geometry)
            attachments.append(dict(
                root_id=root.id, parent_root_id=parent.id,
                parent_node_id=attachment.parent_node_id, attachment_s_px=attach_s,
                attachment_fraction=attach_s / parent_arc[-1] if parent_arc[-1] else None,
                branch_angle_deg=branch_angle, acute_angle_deg=acute_angle,
                attachment_s_mm=attach_s * scale if scale else None,
            ))

    def aggregate(values, scope, order=None, review_state=None, completeness=None):
        total = sum(r['length_px'] for r in values)
        covered = sum(r['length_px'] * r['diameter_coverage'] for r in values)
        diameter = sum(r['mean_diameter_px'] * r['length_px'] * r['diameter_coverage']
                       for r in values if r['mean_diameter_px'] is not None) / covered if covered else None
        known = [r for r in values if r['order'] is not None]
        return dict(
            scope=scope, order=order, review_state=review_state, completeness=completeness,
            object_count=len(values), root_count=len(known), unassigned_count=len(values) - len(known),
            total_length_px=total, known_root_length_px=sum(r['length_px'] for r in known),
            mean_object_length_px=total / len(values) if values else None,
            mean_root_length_px=sum(r['length_px'] for r in known) / len(known) if known else None,
            mean_diameter_px=diameter, diameter_coverage=covered / total if total else 0.0,
            auto_count=sum(r['review_state'] == 'auto' for r in values),
            confirmed_count=sum(r['review_state'] == 'confirmed' for r in values),
            pending_count=sum(r['review_state'] == 'pending' for r in values),
            incomplete_count=sum(r['completeness'] != 'complete' for r in values),
            pending_junction_count=sum(d.state != 'accepted' for d in document.junction_decisions)
            if scope == 'all' else None,
            document_status=document.status,
            total_length_mm=total * scale if scale else None,
            mean_diameter_mm=diameter * scale if scale and diameter is not None else None,
        )

    summary = [aggregate(rows, 'all')]
    groups = {}
    for row in rows:
        groups.setdefault((row['order'], row['review_state'], row['completeness']), []).append(row)
    for (order, review, completeness), values in groups.items():
        summary.append(aggregate(values, 'group', order, review, completeness))
    return {
        'roots': pd.DataFrame(rows, columns=ROOT_COLUMNS),
        'nodes': pd.DataFrame(nodes, columns=NODE_COLUMNS),
        'attachments': pd.DataFrame(attachments, columns=ATTACHMENT_COLUMNS),
        'summary': pd.DataFrame(summary),
    }
