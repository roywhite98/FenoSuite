"""Public vector analysis entry point, independent of Qt and legacy graph types."""
from __future__ import annotations

import copy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from .features import extract_features, load_mask, mask_digest
from .models import BaseMark, RootDocument, SourceRef, VectorAnalysisConfig, VectorAnalysisResult
from .resolve import propose_decisions, rebuild_document
from .vectorize import vectorize


def analyze_vectors(mask, base_marks=None, *, original_image_path=None, config=None,
                    document_id=None, progress=None, check_cancel=None, selection_points=None):
    settings = VectorAnalysisConfig.from_dict(config)
    mask_path = str(Path(mask).resolve()) if isinstance(mask, (str, Path)) else None
    data = load_mask(mask_path) if mask_path else mask
    source_hash = mask_digest(data)

    def report(value, message):
        if check_cancel:
            check_cancel()
        if progress:
            progress(value, 100, message)

    notes, selections, synthetic = [], [], None
    if selection_points is not None:
        from .selection import prepare_selection, prune_spurs, infer_selected_axes
        report(5, '选定根系范围，连接短断口并清理噪点')
        data, synthetic, selections, notes = prepare_selection(data, selection_points, settings, check_cancel)
    report(25, '提取骨架、直径和方向场')
    feature_settings = copy.deepcopy(settings)
    if selection_points is not None:
        feature_settings.min_component_area = 0  # A deliberately selected small root survives filtering.
    features = extract_features(data, feature_settings, check_cancel=check_cancel)
    if selection_points is not None:
        features.skeleton, removed = prune_spurs(features.skeleton, settings.spur_length_px)
        notes.append(f'去除了 {removed} 个短毛刺骨架像素')
    report(35, '压缩骨架为矢量根段')
    snapshot, diagnostics = vectorize(features, settings, check_cancel=check_cancel)
    if synthetic is not None:
        for segment in snapshot.segments:
            for node in segment.nodes:
                if synthetic[round(node.y), round(node.x)]:
                    node.diameter_px, node.diameter_source = None, 'missing'
                    node.tags.append('repaired')
    parameter_hash = sha256(json.dumps(asdict(settings), sort_keys=True).encode()).hexdigest()
    snapshot.parameter_hash = parameter_hash
    snapshot.analysis_id = sha256((source_hash+parameter_hash+json.dumps(selections, sort_keys=True)).encode()).hexdigest()[:24]
    marks = []
    for index, item in enumerate(base_marks or []):
        if isinstance(item, BaseMark):
            mark = copy.deepcopy(item)
        elif isinstance(item, dict):
            mark = BaseMark(**{'id': f'base{index+1}', **item})
        else:
            if len(item) != 2:
                raise ValueError('基部坐标必须为 (x, y)')
            mark = BaseMark(f'base{index+1}', float(item[0]), float(item[1]))
        mark.root_id = None
        marks.append(mark)
    height, width = features.mask.shape
    image_hash = ''
    if original_image_path is not None:
        original_image_path = str(Path(original_image_path).resolve())
        original = load_mask(original_image_path)
        if original.shape != (height, width):
            raise ValueError('原图与根系掩码尺寸不一致')
        image_hash = sha256(original.tobytes()).hexdigest()
    source = SourceRef(width=width, height=height, mask_path=mask_path,
                       image_path=str(original_image_path) if original_image_path else None,
                       mask_hash=source_hash, image_hash=image_hash, mm_per_pixel=settings.mm_per_pixel)
    if selection_points is not None:
        source.algorithm_version = 'vector-beta-5-proximal-anchor'
    doc = RootDocument(source=source, config=settings, base_marks=marks,
                       analysis_snapshot=snapshot, selection_points=selections,
                       diagnostics=[*notes, *features.diagnostics, *diagnostics])
    if document_id is not None:
        doc.document_id = document_id
    report(65, '判断完整主根，再识别侧根及父子关系')
    doc.junction_decisions = propose_decisions(snapshot, settings)
    if selection_points is not None:
        doc.base_marks, notes = infer_selected_axes(snapshot, doc.junction_decisions, check_cancel, settings)
        doc.diagnostics.extend(notes)
    doc = rebuild_document(doc, check_cancel=check_cancel)
    report(100, '矢量根系解析完成')
    return VectorAnalysisResult(doc, list(doc.diagnostics), features.mask, features.skeleton)
