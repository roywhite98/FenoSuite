"""Safe JSON persistence and portable table / SVG exports for vector roots."""
from __future__ import annotations

import colorsys
import dataclasses
import hashlib
import json
import math
import os
import tempfile
import types
import typing
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import pandas as pd

from .metrics import measure_vectors
from .models import RootDocument, validate_document


def root_color(root_id: str, document=None) -> str:
    """Stable, process-independent color; unrelated roots need not share an axis."""
    if document is not None:
        order = document.root_order(root_id)
        return '#f05a47' if order == 0 else ('#28b9ef' if order == 1 else ('#b688ff' if order is not None else '#a0a5ae'))
    hue = int.from_bytes(hashlib.sha256(str(root_id).encode('utf-8')).digest()[:4], 'big') / 2**32
    rgb = colorsys.hls_to_rgb(hue, 0.43, 0.78)
    return '#' + ''.join(f'{round(channel * 255):02x}' for channel in rgb)


def _json_value(value, label):
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f'{label} 含非有限数值')
        return
    if isinstance(value, list):
        for entry in value:
            _json_value(entry, label)
        return
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        for entry in value.values():
            _json_value(entry, label)
        return
    raise ValueError(f'{label} 不是合法 JSON 值')


def _check_type(value, expected, label):
    """Validate JSON against the declared schema before dataclass construction."""
    origin, args = get_origin(expected), get_args(expected)
    if expected is Any:
        _json_value(value, label)
    elif origin in (typing.Union, types.UnionType):
        for candidate in args:
            try:
                _check_type(value, candidate, label)
                return
            except ValueError:
                pass
        raise ValueError(f'{label} 的类型不符合文档格式')
    elif expected is type(None):
        if value is not None:
            raise ValueError(f'{label} 应为 null')
    elif dataclasses.is_dataclass(expected):
        if not isinstance(value, dict):
            raise ValueError(f'{label} 应为对象')
        declared = {f.name: f for f in dataclasses.fields(expected)}
        if set(value) - set(declared):
            raise ValueError(f'{label} 包含未知字段: {sorted(set(value) - set(declared))}')
        annotations = get_type_hints(expected)
        for name, field in declared.items():
            if name in value:
                _check_type(value[name], annotations[name], f'{label}.{name}')
            elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
                raise ValueError(f'{label} 缺少字段 {name}')
    elif origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f'{label} 应为数组')
        if origin is tuple and len(args) > 1 and args[-1] is not Ellipsis:
            if len(value) != len(args):
                raise ValueError(f'{label} 的数组长度错误')
            for index, (item, item_type) in enumerate(zip(value, args)):
                _check_type(item, item_type, f'{label}[{index}]')
        else:
            for index, item in enumerate(value):
                _check_type(item, args[0], f'{label}[{index}]')
    elif origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f'{label} 应为对象')
        for key, entry in value.items():
            _check_type(key, args[0], label)
            _check_type(entry, args[1], f'{label}.{key}')
    elif expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{label} 应为有限数值')
    elif expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'{label} 应为整数')
    elif not isinstance(value, expected):
        raise ValueError(f'{label} 的类型错误')


def _validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError('根系文档必须是 JSON 对象')
    required = {'schema_version', 'document_id', 'revision', 'source', 'config',
                'base_marks', 'roots', 'junction_decisions', 'analysis_snapshot'}
    if required - set(payload):
        raise ValueError(f'根系文档缺少字段: {sorted(required - set(payload))}')
    _check_type(payload, RootDocument, 'document')
    if payload['schema_version'] != 1:
        raise ValueError('不支持的根系文档版本')
    document = RootDocument.from_dict(payload)
    if not document.document_id:
        raise ValueError('文档编号不能为空')
    if document.roots and (document.source.width <= 0 or document.source.height <= 0):
        raise ValueError('含根对象的文档必须有有效图像尺寸')
    snap = document.analysis_snapshot
    segments = {s.id: s for s in snap.segments}
    ports = {p.id: p for p in snap.ports}
    junctions = {j.id: j for j in snap.junctions}
    roots = {r.id: r for r in document.roots}
    segment_nodes = {s.id: {n.id for n in s.nodes} for s in snap.segments}
    for node in [n for r in document.roots for n in r.nodes] + [n for s in snap.segments for n in s.nodes]:
        anchor = node.source_anchor
        if not anchor:
            continue
        if 'segment_id' in anchor or 'node_id' in anchor:
            segment_id, node_id = anchor.get('segment_id'), anchor.get('node_id')
            if (not isinstance(segment_id, str) or not isinstance(node_id, str)
                    or node_id not in segment_nodes.get(segment_id, set())):
                raise ValueError('节点的来源根段或来源节点不存在')
        if 'junction_id' in anchor:
            junction_id = anchor['junction_id']
            if not isinstance(junction_id, str) or junction_id not in junctions:
                raise ValueError('节点的来源交汇不存在')
    for root in document.roots:
        if root.completeness not in ('complete', 'pending', 'boundary', 'loop'):
            raise ValueError('未知根完整性状态')
        if len(set(root.source_segment_ids)) != len(root.source_segment_ids):
            raise ValueError('根的来源根段重复')
        if any(s not in segments for s in root.source_segment_ids):
            raise ValueError('根引用的来源根段不存在')
    if len(set(document.excluded_segment_ids)) != len(document.excluded_segment_ids):
        raise ValueError('排除根段重复')
    if any(s not in segments for s in document.excluded_segment_ids):
        raise ValueError('排除的根段不存在')
    assigned = {s for root in document.roots for s in root.source_segment_ids}
    if assigned.intersection(document.excluded_segment_ids):
        raise ValueError('已排除根段不能同时归属于根对象')
    for mark in document.base_marks:
        if mark.root_id is not None and roots[mark.root_id].base_mark_id != mark.id:
            raise ValueError('基部与根的引用必须双向一致')
    for port in snap.ports:
        if port.id not in junctions[port.junction_id].port_ids:
            raise ValueError('端口缺少交汇的反向引用')
        segment = segments[port.segment_id]
        if port.id not in (segment.start_port, segment.end_port):
            raise ValueError('端口缺少根段的反向引用')
        if not 0 <= port.coherence <= 1:
            raise ValueError('方向场一致性必须在 0 到 1 之间')
        if port.diameter_px is not None and port.diameter_px <= 0:
            raise ValueError('端口直径必须为正数')
    for junction in snap.junctions:
        if len(set(junction.port_ids)) != len(junction.port_ids):
            raise ValueError('交汇端口重复')
    for decision in document.junction_decisions:
        if set(decision.port_ids) != set(junctions[decision.id].port_ids):
            raise ValueError('交汇决定的端口与交汇区域不一致')
        if len(decision.port_ids) != len(set(decision.port_ids)):
            raise ValueError('交汇决定端口重复')
        if decision.source not in ('auto', 'manual'):
            raise ValueError('未知交汇决定来源')
        for alternative in decision.alternatives:
            if alternative.kind not in ('branch', 'crossing', 'continuation', 'stop'):
                raise ValueError('未知交汇候选类别')
    return document


def _atomic(path, writer):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.stem}.', suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    try:
        writer(Path(temporary))
        # All format writers have closed their handles before replacement.
        with open(temporary, 'r+b') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_document(document: RootDocument, path) -> None:
    """Atomically save JSON. Paths are relative to the saved document if possible."""
    validate_document(document)
    payload = document.to_dict()
    _validate_payload(payload)
    target = Path(path).expanduser().resolve()
    for key in ('mask_path', 'image_path'):
        value = payload['source'][key]
        if value:
            absolute = Path(value).expanduser().resolve()
            try:
                payload['source'][key] = Path(os.path.relpath(absolute, target.parent)).as_posix()
            except ValueError:  # Different Windows drives cannot form a relative path.
                payload['source'][key] = str(absolute)
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
    _atomic(target, lambda temporary: temporary.write_text(serialized, encoding='utf-8'))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'JSON 字段重复: {key}')
        result[key] = value
    return result


def load_document(path) -> RootDocument:
    """Load and validate references before returning any document to the GUI."""
    target = Path(path).expanduser().resolve()

    def reject_constant(value):
        raise ValueError(f'JSON 不允许 {value}')

    try:
        payload = json.loads(target.read_text(encoding='utf-8-sig'),
                             object_pairs_hook=_unique_object, parse_constant=reject_constant)
        document = _validate_payload(payload)
    except (TypeError, KeyError, AttributeError, OverflowError) as exc:
        raise ValueError(f'无效根系文档: {exc}') from exc
    for key in ('mask_path', 'image_path'):
        value = getattr(document.source, key)
        if value:
            source_path = Path(value).expanduser()
            setattr(document.source, key, str((target.parent / source_path).resolve()
                                             if not source_path.is_absolute() else source_path.resolve()))
    return document


def _spreadsheet_safe(table):
    def safe(value):
        # Keep numeric cells numeric; only user-supplied strings can become formulas.
        if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
            return "'" + value
        return value
    return table.apply(lambda column: column.map(safe))


def export_tables(document: RootDocument, path) -> None:
    """CSV exports the roots table; XLSX contains all four named tables."""
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix not in ('.csv', '.xlsx'):
        raise ValueError('表格导出仅支持 .csv 或 .xlsx')
    tables = {key: _spreadsheet_safe(value) for key, value in measure_vectors(document).items()}
    if suffix == '.csv':
        _atomic(target, lambda temporary: tables['roots'].to_csv(temporary, index=False, encoding='utf-8-sig'))
    else:
        def write(temporary):
            with pd.ExcelWriter(temporary, engine='openpyxl') as workbook:
                for name, table in tables.items():
                    table.to_excel(workbook, sheet_name=name, index=False)
        _atomic(target, write)


def export_svg(document: RootDocument, path) -> None:
    """Export editable vector centerlines with stable root IDs and colors."""
    tables = measure_vectors(document)
    width, height = max(1, document.source.width), max(1, document.source.height)
    svg = ET.Element('svg', {'xmlns': 'http://www.w3.org/2000/svg',
                            'viewBox': f'0 0 {width} {height}',
                            'width': str(width), 'height': str(height)})
    ET.SubElement(svg, 'title').text = '根系表型 beta：矢量根系'
    ET.SubElement(svg, 'metadata').text = json.dumps({
        'document_id': document.document_id, 'revision': document.revision,
        'schema_version': document.schema_version, 'coordinates': 'pixel; x=column,y=row',
        'mm_per_pixel': document.source.mm_per_pixel,
    }, ensure_ascii=False)
    ET.SubElement(svg, 'rect', {'width': '100%', 'height': '100%', 'fill': '#ffffff'})
    row_map = {row['root_id']: row for row in tables['roots'].to_dict('records')}
    for root in document.roots:
        row = row_map[root.id]
        group = ET.SubElement(svg, 'g', {'data-root-id': root.id})
        ET.SubElement(group, 'title').text = (
            f"{root.id}; order={document.root_order(root.id)}; "
            f"length={row['length_px']:.3f} px; {root.review_state}; {root.completeness}")
        attributes = {'points': ' '.join(f'{node.x:.6g},{node.y:.6g}' for node in root.nodes),
                      'fill': 'none', 'stroke': root_color(root.id, document), 'stroke-width': '1.5',
                      'stroke-linecap': 'round', 'stroke-linejoin': 'round'}
        if root.review_state == 'pending':
            attributes['stroke-dasharray'] = '4 2'
        ET.SubElement(group, 'polyline', attributes)
    for mark in document.base_marks:
        ET.SubElement(svg, 'circle', {'cx': str(mark.x), 'cy': str(mark.y), 'r': '3',
                                    'fill': '#ffffff', 'stroke': '#111111', 'stroke-width': '1.5'})
    junctions = {j.id: j for j in document.analysis_snapshot.junctions}
    for decision in document.junction_decisions:
        if decision.state != 'accepted':
            junction = junctions[decision.id]
            ET.SubElement(svg, 'circle', {'cx': str(junction.x), 'cy': str(junction.y), 'r': '4',
                                        'fill': 'none', 'stroke': '#e68a00', 'stroke-width': '1.5'})
    serialized = ET.tostring(svg, encoding='utf-8', xml_declaration=True)
    _atomic(path, lambda temporary: temporary.write_bytes(serialized))
