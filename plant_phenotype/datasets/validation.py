# -*- coding: utf-8 -*-
"""数据校验：缺失文件、重复配对、尺寸不一致、未知标签、有损掩码与未登记文件。"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from plant_phenotype.datasets import mask_io
from plant_phenotype.datasets.schema import IMAGES_DIR, MASKS_DIR, SPLITS, split_label
from plant_phenotype.datasets.store import Dataset, ProgressFn

ISSUE_LABELS = {
    "missing_image": "原图缺失",
    "missing_mask": "掩码缺失",
    "duplicate_image": "重复配对（原图）",
    "duplicate_mask": "重复配对（掩码）",
    "size_mismatch": "尺寸不一致",
    "unknown_label": "未知标签",
    "lossy_mask": "有损掩码格式",
    "unreadable": "无法读取",
    "status_mismatch": "状态与掩码不符",
    "split_mismatch": "划分与目录不符",
    "orphan_file": "未登记文件",
}


def _split_of_rel(rel: str) -> str | None:
    parts = rel.split("/")
    if len(parts) > 2 and parts[0] in SPLITS and parts[1] in (IMAGES_DIR, MASKS_DIR):
        return parts[0]
    return None


@dataclass
class Issue:
    severity: str  # error | warning
    kind: str
    message: str
    sample_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def validate_dataset(ds: Dataset, check_pixels: bool = True, progress: ProgressFn | None = None,
                     cancel: Callable[[], bool] | None = None) -> list[Issue]:
    issues: list[Issue] = []
    known = ds.meta.class_ids
    by_image: dict[str, list[str]] = defaultdict(list)
    by_mask: dict[str, list[str]] = defaultdict(list)
    referenced: set[str] = set()

    for i, s in enumerate(ds.samples):
        if cancel and cancel():
            raise RuntimeError("cancelled")
        if progress:
            progress(i, len(ds.samples), s.image_name)
        by_image[os.path.normcase(s.image)].append(s.id)
        referenced.add(os.path.normcase(s.image))
        img_path = ds.abs(s.image)
        img_size = None
        if not img_path.is_file():
            issues.append(Issue("error", "missing_image", f"原图不存在: {s.image}", s.id))
        else:
            try:
                img_size = mask_io.read_image_size(img_path)
            except Exception as e:
                issues.append(Issue("error", "unreadable", f"原图无法读取: {e}", s.id))

        mask_paths = ([s.mask] if s.mask else []) + [r.mask for r in s.instances]
        wrong = [rel for rel in [s.image, *mask_paths] if _split_of_rel(rel) != s.split]
        if wrong:
            issues.append(Issue(
                "warning", "split_mismatch",
                f"样本划分为「{split_label(s.split)}」，但文件位于其它目录: {', '.join(wrong[:3])}", s.id,
            ))
        if s.status in ("annotated", "predicted") and not mask_paths:
            issues.append(Issue("warning", "status_mismatch", f"状态为 {s.status} 但没有掩码", s.id))
        if s.status == "unlabeled" and mask_paths:
            issues.append(Issue("warning", "status_mismatch", "有掩码但状态为未标注", s.id))
        for r in s.instances:
            if r.class_id not in known:
                issues.append(Issue("error", "unknown_label", f"实例 {r.id} 的类别 {r.class_id} 未定义", s.id))
        for rel in mask_paths:
            by_mask[os.path.normcase(rel)].append(s.id)
            referenced.add(os.path.normcase(rel))
            p = ds.abs(rel)
            if p.suffix.lower() not in mask_io.LOSSLESS_MASK_EXTENSIONS:
                issues.append(Issue("warning", "lossy_mask", f"掩码不是无损格式: {rel}", s.id))
            if not p.is_file():
                issues.append(Issue("error", "missing_mask", f"掩码不存在: {rel}", s.id))
                continue
            try:
                mw, mh = mask_io.read_image_size(p)
            except Exception as e:
                issues.append(Issue("error", "unreadable", f"掩码无法读取: {e}", s.id))
                continue
            if img_size is not None and (mw, mh) != img_size:
                issues.append(Issue(
                    "error", "size_mismatch",
                    f"尺寸不一致: 原图 {img_size[0]}×{img_size[1]}，掩码 {mw}×{mh}（{rel}）", s.id,
                ))
            if check_pixels and rel == s.mask:
                try:
                    vals = set(np.unique(mask_io.read_label_mask(p)).tolist())
                except Exception as e:
                    issues.append(Issue("error", "unreadable", f"掩码不是单通道编号图: {e}", s.id))
                    continue
                unknown = sorted(vals - known)
                if unknown:
                    issues.append(Issue("error", "unknown_label", f"掩码含未定义的标签值 {unknown}", s.id))

    for ids in by_image.values():
        if len(ids) > 1:
            for sid in ids:
                issues.append(Issue("error", "duplicate_image", f"原图被 {len(ids)} 个样本引用: {', '.join(ids)}", sid))
    for rel, ids in by_mask.items():
        if len(ids) > 1:
            for sid in ids:
                issues.append(Issue("error", "duplicate_mask", f"掩码 {rel} 被多个样本引用: {', '.join(ids)}", sid))

    for d in ds.data_dirs():
        if not d.is_dir():
            continue
        for f in d.rglob("*"):
            if f.is_file() and os.path.normcase(ds.rel(f)) not in referenced:
                issues.append(Issue("warning", "orphan_file", f"未被任何样本引用: {ds.rel(f)}"))
    if progress:
        progress(len(ds.samples), len(ds.samples), "校验完成")
    return issues
