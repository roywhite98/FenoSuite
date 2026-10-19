"""Qt-independent, serializable vector geometry and root ancestry.

Coordinates are image (x=column, y=row) in pixels. A missing parent does
not establish a primary root: a valid base mark must anchor that root.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any
from uuid import uuid4


@dataclass
class VectorAnalysisConfig:
    min_component_area: int = 0
    field_scales: tuple[int, ...] = (5, 9, 17, 33, 65)
    coherence_min: float = 0.2
    energy_min: float = 1e-6
    sample_length_factor: float = 2.5
    max_pair_cost: float = 0.32
    min_pair_margin: float = 0.08
    simplify_tolerance: float = 0.6
    diameter_tolerance: float = 0.5
    length_error_fraction: float = 0.01
    base_snap_distance: float = 15.0
    connect_gap_px: float = 20.0
    spur_length_px: float = 5.0
    mm_per_pixel: float | None = None

    @classmethod
    def from_dict(cls, value=None):
        if isinstance(value, cls):
            result = copy.deepcopy(value)
        else:
            value = dict(value or {})
            unknown = set(value) - {f.name for f in fields(cls)}
            if unknown:
                raise ValueError(f"未知矢量分析参数: {', '.join(sorted(unknown))}")
            result = cls(**value)
        result.field_scales = tuple(result.field_scales)
        if not result.field_scales or any(not isinstance(n, int) or n < 3 or n % 2 == 0 for n in result.field_scales):
            raise ValueError("方向场窗口必须是至少为 3 的奇数")
        for name in ('coherence_min', 'max_pair_cost', 'min_pair_margin', 'length_error_fraction'):
            v = getattr(result, name)
            if not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError(f"{name} 必须在 0 到 1 之间")
        for name in ('energy_min', 'sample_length_factor', 'simplify_tolerance', 'diameter_tolerance', 'base_snap_distance'):
            v = getattr(result, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} 必须为正数")
        if not isinstance(result.min_component_area, int) or result.min_component_area < 0:
            raise ValueError("最小连通域面积必须为非负整数")
        for name in ('connect_gap_px', 'spur_length_px'):
            if not math.isfinite(getattr(result, name)) or getattr(result, name) < 0:
                raise ValueError(f"{name} 必须为非负数")
        if result.mm_per_pixel is not None and (not math.isfinite(result.mm_per_pixel) or result.mm_per_pixel <= 0):
            raise ValueError("标定必须为正数")
        return result


@dataclass
class SourceRef:
    width: int = 0
    height: int = 0
    mask_path: str | None = None
    image_path: str | None = None
    mask_hash: str = ''
    image_hash: str = ''
    mm_per_pixel: float | None = None
    algorithm_version: str = 'vector-beta-1'


@dataclass
class RootNode:
    id: str
    x: float
    y: float
    diameter_px: float | None = None
    diameter_source: str = 'measured'
    tags: list[str] = field(default_factory=list)
    source_anchor: dict[str, Any] | None = None


@dataclass
class ParentAttachment:
    parent_root_id: str
    parent_node_id: str


@dataclass
class VectorRoot:
    id: str
    nodes: list[RootNode] = field(default_factory=list)
    parent_attachment: ParentAttachment | None = None
    base_mark_id: str | None = None
    direction: str = 'unknown'
    review_state: str = 'pending'
    completeness: str = 'complete'
    source_segment_ids: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)


@dataclass
class BaseMark:
    id: str
    x: float
    y: float
    root_id: str | None = None
    direction_hint: tuple[float, float] | None = None
    specimen_id: str | None = None


@dataclass
class Port:
    id: str
    segment_id: str
    junction_id: str
    x: float
    y: float
    tangent: tuple[float, float] = (0.0, 0.0)
    diameter_px: float | None = None
    field_axis: tuple[float, float] | None = None
    coherence: float = 0.0


@dataclass
class Segment:
    id: str
    nodes: list[RootNode] = field(default_factory=list)
    start_port: str | None = None
    end_port: str | None = None
    closed: bool = False
    quality_flags: list[str] = field(default_factory=list)


@dataclass
class JunctionRegion:
    id: str
    x: float
    y: float
    port_ids: list[str] = field(default_factory=list)
    pixels: list[tuple[int, int]] = field(default_factory=list)  # x,y


@dataclass
class JunctionAlternative:
    id: str
    kind: str  # branch, crossing, continuation, stop
    pairs: list[list[str]] = field(default_factory=list)
    cost: float = 0.0
    label: str = ''
    score_terms: dict[str, float] = field(default_factory=dict)


@dataclass
class JunctionDecision:
    id: str  # same id as the JunctionRegion
    port_ids: list[str] = field(default_factory=list)
    alternatives: list[JunctionAlternative] = field(default_factory=list)
    chosen: str | None = None
    state: str = 'pending'
    source: str = 'auto'
    score_terms: dict[str, float] = field(default_factory=dict)


@dataclass
class AnalysisSnapshot:
    analysis_id: str = field(default_factory=lambda: uuid4().hex)
    segments: list[Segment] = field(default_factory=list)
    ports: list[Port] = field(default_factory=list)
    junctions: list[JunctionRegion] = field(default_factory=list)
    parameter_hash: str = ''


@dataclass
class RootDocument:
    schema_version: int = 1
    document_id: str = field(default_factory=lambda: uuid4().hex)
    revision: int = 0
    source: SourceRef = field(default_factory=SourceRef)
    config: VectorAnalysisConfig = field(default_factory=VectorAnalysisConfig)
    base_marks: list[BaseMark] = field(default_factory=list)
    roots: list[VectorRoot] = field(default_factory=list)
    junction_decisions: list[JunctionDecision] = field(default_factory=list)
    analysis_snapshot: AnalysisSnapshot = field(default_factory=AnalysisSnapshot)
    diagnostics: list[str] = field(default_factory=list)
    excluded_segment_ids: list[str] = field(default_factory=list)
    selection_points: list[dict] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.roots and not self.analysis_snapshot.segments:
            return 'empty'
        if any(d.state != 'accepted' for d in self.junction_decisions):
            return 'needs_review'
        if any(r.review_state == 'pending' or r.direction == 'unknown' for r in self.roots):
            return 'needs_review'
        return 'ready'

    def root_by_id(self, root_id: str) -> VectorRoot:
        for root in self.roots:
            if root.id == root_id:
                return root
        raise ValueError(f"根不存在: {root_id}")

    def root_order(self, root_id: str) -> int | None:
        roots = {r.id: r for r in self.roots}
        seen = set()
        depth = 0
        current = roots.get(root_id)
        while current is not None:
            if current.id in seen:
                raise ValueError("父子关系存在循环")
            seen.add(current.id)
            if current.parent_attachment is None:
                return depth if current.base_mark_id is not None and any(b.id == current.base_mark_id and b.root_id == current.id for b in self.base_marks) else None
            depth += 1
            current = roots.get(current.parent_attachment.parent_root_id)
        return None

    def clone(self):
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        data = copy.deepcopy(data)
        if data.get('schema_version') != 1:
            raise ValueError("不支持的根系文档版本")
        data['source'] = SourceRef(**data.get('source', {}))
        data['config'] = VectorAnalysisConfig.from_dict(data.get('config'))
        data['base_marks'] = [BaseMark(**v) for v in data.get('base_marks', [])]
        def node(v):
            return RootNode(**v)
        roots = []
        for v in data.get('roots', []):
            v['nodes'] = [node(n) for n in v.get('nodes', [])]
            if v.get('parent_attachment'):
                v['parent_attachment'] = ParentAttachment(**v['parent_attachment'])
            roots.append(VectorRoot(**v))
        data['roots'] = roots
        decisions = []
        for v in data.get('junction_decisions', []):
            v['alternatives'] = [JunctionAlternative(**a) for a in v.get('alternatives', [])]
            decisions.append(JunctionDecision(**v))
        data['junction_decisions'] = decisions
        snap = data.get('analysis_snapshot', {})
        segments = []
        for v in snap.get('segments', []):
            v['nodes'] = [node(n) for n in v.get('nodes', [])]
            segments.append(Segment(**v))
        snap['segments'] = segments
        snap['ports'] = [Port(**v) for v in snap.get('ports', [])]
        snap['junctions'] = [JunctionRegion(**v) for v in snap.get('junctions', [])]
        data['analysis_snapshot'] = AnalysisSnapshot(**snap)
        result = cls(**data)
        validate_document(result)
        return result


@dataclass
class VectorAnalysisResult:
    document: RootDocument
    diagnostics: list[str] = field(default_factory=list)
    mask: Any = None
    skeleton: Any = None


@dataclass
class EditResult:
    document: RootDocument
    affected_root_ids: list[str] = field(default_factory=list)


def validate_document(doc: RootDocument) -> None:
    """Reject invalid geometry and ancestry before committing or loading."""
    if doc.schema_version != 1 or not isinstance(doc.revision, int) or doc.revision < 0:
        raise ValueError('无效文档版本')
    VectorAnalysisConfig.from_dict(doc.config)
    if doc.source.width < 0 or doc.source.height < 0:
        raise ValueError('无效图像尺寸')
    scale = doc.source.mm_per_pixel
    if scale is not None and (not math.isfinite(scale) or scale <= 0):
        raise ValueError('无效标定')
    def unique(items, label):
        ids = [x.id for x in items]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f'{label}编号为空或重复')
    unique(doc.roots, '根')
    unique(doc.base_marks, '基部')
    roots = {r.id: r for r in doc.roots}
    for point in doc.selection_points:
        if not isinstance(point, dict) or any(
            not isinstance(point.get(k), (int, float)) or not math.isfinite(point[k])
            for k in ('x', 'y')
        ):
            raise ValueError('无效根系选点坐标')
        if not (0 <= point['x'] < doc.source.width and 0 <= point['y'] < doc.source.height):
            raise ValueError('根系选点超出图像范围')
    for mark in doc.base_marks:
        if not all(math.isfinite(v) for v in (mark.x, mark.y)):
            raise ValueError('无效基部坐标')
        if mark.root_id is not None and mark.root_id not in roots:
            raise ValueError('基部引用的根不存在')
    assigned = set()
    def validate_nodes(nodes):
        unique(nodes, '节点')
        for n in nodes:
            if not all(math.isfinite(v) for v in (n.x, n.y)):
                raise ValueError('节点坐标必须为有限数值')
            if n.diameter_px is not None and (not math.isfinite(n.diameter_px) or n.diameter_px <= 0):
                raise ValueError('节点直径必须为正数或 null')
            if n.diameter_source not in ('measured', 'interpolated', 'manual', 'missing'):
                raise ValueError('未知直径来源')
    for r in doc.roots:
        if len(r.nodes) < 2:
            raise ValueError('根至少需要两个节点')
        validate_nodes(r.nodes)
        if r.direction not in ('unknown', 'base_to_tip') or r.review_state not in ('auto', 'confirmed', 'pending'):
            raise ValueError('无效根状态')
        if any(math.hypot(a.x-b.x, a.y-b.y) <= 1e-10 for a,b in zip(r.nodes, r.nodes[1:])):
            raise ValueError('根包含相邻重复节点')
        if r.base_mark_id is not None and not any(b.id == r.base_mark_id and b.root_id == r.id for b in doc.base_marks):
            raise ValueError('基部与根引用不一致')
        if assigned.intersection(r.source_segment_ids):
            raise ValueError('同一根段归属于多条根')
        assigned.update(r.source_segment_ids)
        if r.parent_attachment is not None:
            a = r.parent_attachment
            parent = roots.get(a.parent_root_id)
            if parent is None:
                raise ValueError('父根不存在')
            parent_node = next((n for n in parent.nodes if n.id == a.parent_node_id), None)
            if parent_node is None:
                raise ValueError('附着节点不存在')
            if math.hypot(parent_node.x-r.nodes[0].x, parent_node.y-r.nodes[0].y) > 1e-5:
                raise ValueError('子根首点必须与附着节点重合')
            if r.base_mark_id is not None:
                raise ValueError('已附着的子根不能同时是基部主轴')
        doc.root_order(r.id)
    snap = doc.analysis_snapshot
    unique(snap.segments, '根段')
    unique(snap.ports, '端口')
    unique(snap.junctions, '交汇')
    unique(doc.junction_decisions, '交汇决定')
    segments = {s.id: s for s in snap.segments}
    ports = {p.id: p for p in snap.ports}
    junctions = {j.id: j for j in snap.junctions}
    for segment in snap.segments:
        validate_nodes(segment.nodes)
        if len(segment.nodes) < 2:
            raise ValueError('根段至少需要两个节点')
        for port_id in (segment.start_port, segment.end_port):
            if port_id is not None and (port_id not in ports or ports[port_id].segment_id != segment.id):
                raise ValueError('根段端口引用无效')
    for p in snap.ports:
        if p.segment_id not in segments or p.junction_id not in junctions:
            raise ValueError('端口引用无效')
        if not all(math.isfinite(v) for v in (p.x, p.y, *p.tangent, p.coherence)):
            raise ValueError('端口特征包含无效数值')
    for j in snap.junctions:
        if not all(math.isfinite(v) for v in (j.x, j.y)) or any(p not in ports or ports[p].junction_id != j.id for p in j.port_ids):
            raise ValueError('交汇端口引用无效')
    for d in doc.junction_decisions:
        if d.id not in junctions or d.state not in ('pending', 'accepted', 'stale'):
            raise ValueError('无效交汇决定')
        unique(d.alternatives, '候选')
        if d.chosen is not None and not any(a.id == d.chosen for a in d.alternatives):
            raise ValueError('所选接续候选不存在')
        if d.state == 'accepted' and d.chosen is None:
            raise ValueError('已接受决定缺少候选')
        for a in d.alternatives:
            used = set()
            if not math.isfinite(a.cost):
                raise ValueError('无效候选分数')
            for pair in a.pairs:
                if len(pair) != 2 or pair[0] == pair[1] or any(p not in d.port_ids or p in used for p in pair):
                    raise ValueError('接续候选的端口配对冲突')
                used.update(pair)
