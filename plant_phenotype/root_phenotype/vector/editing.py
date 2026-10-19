"""Atomic vector edits. Never modify a document currently displayed by Qt."""
from __future__ import annotations

import copy
import math

from .models import EditResult, ParentAttachment, RootDocument, validate_document


def _node(root, node_id):
    for node in root.nodes:
        if node.id == node_id:
            return node
    raise ValueError(f'节点不存在: {node_id}')


def _lock(node):
    if 'manual_locked' not in node.tags:
        node.tags.append('manual_locked')


def _move_attachment_children(doc, parent, node):
    """Only the attached first point follows, not the whole child curve."""
    affected = []
    for child in doc.roots:
        a = child.parent_attachment
        if a and a.parent_root_id == parent.id and a.parent_node_id == node.id:
            first = child.nodes[0]
            if math.hypot(first.x-node.x, first.y-node.y) <= 1e-10:
                continue
            first.x, first.y = node.x, node.y
            first.diameter_px = None
            first.diameter_source = 'missing'
            _lock(first)
            affected.append(child.id)
            affected.extend(_move_attachment_children(doc, child, first))
    return affected


def apply_edit(document: RootDocument, command: dict) -> EditResult:
    doc = document.clone()
    kind = command.get('type')
    affected = []
    if kind == 'set_scale':
        value = command.get('mm_per_pixel')
        if value is not None:
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('每像素对应毫米数必须为正数')
        doc.source.mm_per_pixel = value
        doc.config.mm_per_pixel = value
        affected = [r.id for r in doc.roots]
    else:
        root = doc.root_by_id(command.get('root_id', ''))
        affected.append(root.id)
        if kind == 'move_node':
            node = _node(root, command['node_id'])
            x, y = float(command['x']), float(command['y'])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError('节点坐标必须为有限数值')
            if not (0 <= x < doc.source.width and 0 <= y < doc.source.height):
                raise ValueError('节点必须位于图像范围内')
            if root.parent_attachment and node is root.nodes[0]:
                raise ValueError('子根基部由父根附着节点控制；请移动父节点或重新附着')
            node.x, node.y = x, y
            node.diameter_px, node.diameter_source = None, 'missing'
            _lock(node)
            affected.extend(_move_attachment_children(doc, root, node))
        elif kind == 'set_diameter':
            node = _node(root, command['node_id'])
            value = command.get('diameter_px')
            if value is not None:
                value = float(value)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError('直径必须为正数')
            node.diameter_px = value
            node.diameter_source = 'manual' if value is not None else 'missing'
            _lock(node)
        elif kind in ('reparent', 'detach'):
            if kind == 'detach':
                root.parent_attachment = None
                root.base_mark_id = None
                root.direction = 'unknown'
                root.review_state = 'pending'
            else:
                parent = doc.root_by_id(command['parent_root_id'])
                if parent.id == root.id:
                    raise ValueError('根不能附着到自身')
                point = _node(parent, command['parent_node_id'])
                if doc.root_order(parent.id) is None or parent.direction != 'base_to_tip':
                    raise ValueError('请先确定父根基部与方向')
                root.parent_attachment = ParentAttachment(parent.id, point.id)
                root.base_mark_id = None
                root.direction = 'base_to_tip'
                root.review_state = 'confirmed'
                first = root.nodes[0]
                first.x, first.y = point.x, point.y
                first.diameter_px, first.diameter_source = None, 'missing'
                _lock(first)
                _lock(point)
                if 'attachment' not in point.tags:
                    point.tags.append('attachment')
                affected.append(parent.id)
                # Check for cycles before traversing descendants.
                doc.root_order(root.id)
                affected.extend(_move_attachment_children(doc, root, first))
            doc.base_marks = [b for b in doc.base_marks if b.root_id != root.id]
            if 'manual_parent' not in root.quality_flags:
                root.quality_flags.append('manual_parent')
        elif kind == 'reverse':
            if root.parent_attachment:
                raise ValueError('附着子根反向前需要解除附着')
            root.nodes.reverse()
            root.direction = 'base_to_tip' if root.base_mark_id else 'unknown'
            for mark in doc.base_marks:
                if mark.root_id == root.id:
                    mark.x, mark.y = root.nodes[0].x, root.nodes[0].y
            if 'manual_direction' not in root.quality_flags:
                root.quality_flags.append('manual_direction')
        elif kind == 'confirm_root':
            if doc.root_order(root.id) is None:
                raise ValueError('请先指定基部或父根，才能确认主侧根归属')
            root.review_state = 'confirmed'
        elif kind == 'delete_root':
            removed = {root.id}
            if command.get('subtree', False):
                while True:
                    children = {r.id for r in doc.roots if r.parent_attachment and r.parent_attachment.parent_root_id in removed}
                    if children <= removed:
                        break
                    removed |= children
            excluded = set(doc.excluded_segment_ids)
            for r in doc.roots:
                if r.id in removed:
                    excluded.update(r.source_segment_ids)
                elif r.parent_attachment and r.parent_attachment.parent_root_id in removed:
                    r.parent_attachment = None
                    r.base_mark_id = None
                    r.direction = 'unknown'
                    r.review_state = 'pending'
                    if 'manual_parent' not in r.quality_flags:
                        r.quality_flags.append('manual_parent')
                    affected.append(r.id)
            doc.excluded_segment_ids = sorted(excluded)
            doc.roots = [r for r in doc.roots if r.id not in removed]
            doc.base_marks = [b for b in doc.base_marks if b.root_id not in removed]
            affected.extend(removed)
        else:
            raise ValueError(f'未知编辑命令: {kind}')
    doc.revision = document.revision + 1
    validate_document(doc)
    return EditResult(doc, sorted(set(affected)))


def _preserve_edits(previous: RootDocument, candidate: RootDocument) -> None:
    """A rebuild may change identities, but must not erase manual geometry."""
    for old in previous.roots:
        protected = [n for n in old.nodes if 'manual_locked' in n.tags]
        manual_parent = 'manual_parent' in old.quality_flags
        manual_direction = 'manual_direction' in old.quality_flags
        targets = [r for r in candidate.roots if r.id == old.id]
        if not targets:
            targets = [r for r in candidate.roots if set(r.source_segment_ids).intersection(old.source_segment_ids)]
        if old.review_state == 'confirmed':
            for target in targets:
                unchanged_identity = (set(target.source_segment_ids) == set(old.source_segment_ids)
                                      and target.parent_attachment == old.parent_attachment
                                      and target.base_mark_id == old.base_mark_id)
                target.review_state = 'confirmed' if unchanged_identity else 'pending'
                if not unchanged_identity and 'confirmation_stale' not in target.quality_flags:
                    target.quality_flags.append('confirmation_stale')
        if not protected and not manual_parent and not manual_direction:
            continue
        if len(targets) != 1:
            raise ValueError('接续改变涉及人工编辑，无法安全映射；请撤销相关编辑后重试')
        target = targets[0]
        for node in protected:
            matches = [n for n in target.nodes if n.id == node.id]
            if not matches and node.source_anchor:
                matches = [n for n in target.nodes if n.source_anchor == node.source_anchor]
            if len(matches) != 1:
                raise ValueError('接续改变会删除人工节点；请先撤销相关节点编辑')
            point = matches[0]
            point.x, point.y = node.x, node.y
            point.diameter_px, point.diameter_source = node.diameter_px, node.diameter_source
            point.tags = list(dict.fromkeys([*point.tags, *node.tags]))
        if manual_parent:
            target.parent_attachment = copy.deepcopy(old.parent_attachment)
            target.base_mark_id = old.base_mark_id
            target.direction, target.review_state = old.direction, old.review_state
            if 'manual_parent' not in target.quality_flags:
                target.quality_flags.append('manual_parent')
        if manual_direction:
            if target.nodes[0].id == old.nodes[-1].id:
                target.nodes.reverse()
            elif target.nodes[0].id != old.nodes[0].id:
                raise ValueError('接续改变与人工方向冲突')
            if 'manual_direction' not in target.quality_flags:
                target.quality_flags.append('manual_direction')
    # Parent overrides can introduce a cycle into an otherwise valid rebuild.
    # Check before recursively synchronizing attachment coordinates.
    for r in candidate.roots:
        candidate.root_order(r.id)
    for r in candidate.roots:
        for n in r.nodes:
            if 'manual_locked' in n.tags:
                _move_attachment_children(candidate, r, n)


def resolve_decisions(document: RootDocument, decisions: dict, *, check_cancel=None) -> EditResult:
    from .resolve import rebuild_document
    doc = document.clone()
    by_id = {d.id: d for d in doc.junction_decisions}
    for junction_id, alternative_id in decisions.items():
        if junction_id not in by_id:
            raise ValueError('交汇不存在')
        d = by_id[junction_id]
        if alternative_id is not None and not any(a.id == alternative_id for a in d.alternatives):
            raise ValueError('接续候选不存在')
        d.chosen = alternative_id
        d.state = 'accepted' if alternative_id is not None else 'pending'
        d.source = 'manual'
    if check_cancel:
        check_cancel()
    updated = rebuild_document(doc, check_cancel=check_cancel)
    updated.revision = document.revision + 1
    validate_document(updated)
    if check_cancel:
        check_cancel()
    return EditResult(updated, sorted({r.id for r in document.roots} | {r.id for r in updated.roots}))
