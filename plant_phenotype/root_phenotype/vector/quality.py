"""Review uncertain primary paths without inventing geometry across mask gaps."""
from __future__ import annotations

import math
from statistics import median


def _has_thin_detour(root):
    measured = [n for n in root.nodes if n.diameter_px and
                n.diameter_source in ('measured', 'manual')]
    if len(measured) < 6:
        return False
    count = max(3, len(measured)//20)
    first = median(n.diameter_px for n in measured[:count])
    last = median(n.diameter_px for n in measured[-count:])
    reference = min(first, last)
    if max(first, last) > 1.25*reference:
        return False
    # A long measured thin interval bounded by thick tissue is suspicious.
    # Junctions and repaired pixels have no measured width: they contribute
    # no thin length and a long unknown interval breaks this inference.
    thick_seen, thin_length, unknown_length = False, 0.0, 0.0
    for a, b in zip(root.nodes, root.nodes[1:]):
        length = math.hypot(a.x-b.x, a.y-b.y)
        widths = [n.diameter_px for n in (a, b) if n.diameter_px and
                  n.diameter_source in ('measured', 'manual')]
        if len(widths) != 2:
            unknown_length += length
            if unknown_length > reference:
                thick_seen, thin_length = False, 0.0
            continue
        unknown_length = 0.0
        if min(widths) >= .6*reference:
            if thick_seen and thin_length >= 3*reference:
                return True
            thick_seen, thin_length = True, 0.0
        elif max(widths) < .6*reference and thick_seen:
            thin_length += length
    return False


def review_primary_paths(document):
    """Recompute after assembly; explicit root confirmations remain authoritative."""
    if not document.selection_points:
        return
    document.diagnostics = [s for s in document.diagnostics if not s.startswith('主根候选 ')]
    uncertain = set()
    for root in document.roots:
        if document.root_order(root.id) != 0 or not _has_thin_detour(root):
            continue
        root.quality_flags.append('possible_trunk_detour')
        if root.review_state == 'confirmed':
            continue
        root.review_state = 'pending'
        uncertain.add(root.id)
        document.diagnostics.insert(0, f'主根候选 {root.id} 可能绕过了主干断口：粗根之间有长细根绕行，'
                                    '已标为待确认虚线；缺失主干尚未恢复')
    roots = {r.id: r for r in document.roots}
    for root in document.roots:
        current = root
        while current.parent_attachment:
            current = roots[current.parent_attachment.parent_root_id]
            if current.id in uncertain:
                root.quality_flags.append('uncertain_primary_ancestry')
                if root.review_state != 'confirmed':
                    root.review_state = 'pending'
                break
