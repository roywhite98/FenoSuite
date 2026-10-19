# -*- coding: utf-8 -*-
"""数据集存储：多数据集扫描、创建、重命名、复制、删除，以及样本/标签的读写。

纯 Python，不依赖 Qt；可在后台任务线程中使用。同一数据集根目录的写操作串行化。
"""
from __future__ import annotations

import json
import os
import random
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from plant_phenotype.datasets import mask_io
from plant_phenotype.datasets.schema import (
    BACKGROUND_ID,
    DATASET_FILE,
    FORMAT_VERSION,
    IMAGES_DIR,
    MASKS_DIR,
    REPORTS_DIR,
    SAMPLES_FILE,
    SPLITS,
    TASK_TYPES,
    DatasetMeta,
    InstanceRecord,
    LabelClass,
    Sample,
    default_classes,
    default_color,
    normalize_split,
    split_label,
)

ProgressFn = Callable[[int, int, str], None]

_ROOT_LOCKS: dict[str, threading.RLock] = {}
_ROOT_LOCKS_GUARD = threading.Lock()


def _lock_for(root: Path) -> threading.RLock:
    key = os.path.normcase(str(root.resolve()))
    with _ROOT_LOCKS_GUARD:
        lock = _ROOT_LOCKS.get(key)
        if lock is None:
            lock = _ROOT_LOCKS[key] = threading.RLock()
        return lock


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class DatasetError(Exception):
    """数据集操作失败。"""


class DatasetProtectedError(DatasetError):
    """当前功能只能处理二值语义数据，为避免覆盖多类别/实例标注而拒绝写入。"""


@dataclass(frozen=True)
class DatasetSummary:
    id: str
    name: str
    task_type: str
    n_samples: int
    n_annotated: int
    n_predicted: int
    n_classes: int
    path: Path
    n_splits: dict[str, int] = field(default_factory=dict)

    @property
    def has_splits(self) -> bool:
        return any(self.n_splits.get(s) for s in SPLITS)

    def split_text(self) -> str:
        """如 “训练集 70 · 验证集 20 · 测试集 10 · 未划分 3”；没有划分时为空。"""
        if not self.has_splits:
            return ""
        keys = [s for s in SPLITS if self.n_splits.get(s)] + ([""] if self.n_splits.get("") else [])
        return " · ".join(f"{split_label(k or None)} {self.n_splits[k]}" for k in keys)


def allocate_counts(n: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """最大余数法把 n 个样本按比例分给各划分。"""
    total = sum(ratios.values())
    exact = {k: n * v / total for k, v in ratios.items()}
    out = {k: int(x) for k, x in exact.items()}
    rest = n - sum(out.values())
    for k in sorted(exact, key=lambda k: (-(exact[k] - out[k]), SPLITS.index(k)))[:rest]:
        out[k] += 1
    return out


class Dataset:
    """一个已加载的数据集；修改后调用 :meth:`save` 落盘。"""

    def __init__(self, path: Path, meta: DatasetMeta, samples: list[Sample], next_seq: int) -> None:
        self.path = Path(path)
        self.meta = meta
        self.samples = samples
        self._next_seq = next_seq
        self._used_names: set[str] | None = None
        self._lock = _lock_for(self.path.parent)

    # ---------- 基础 ----------
    @property
    def id(self) -> str:
        return self.meta.id

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def images_dir(self) -> Path:
        """未划分样本的原图目录；已划分样本位于 ``<split>/images``。"""
        return self.path / IMAGES_DIR

    @property
    def masks_dir(self) -> Path:
        return self.path / MASKS_DIR

    @property
    def reports_dir(self) -> Path:
        return self.path / REPORTS_DIR

    @staticmethod
    def _prefix(split: str | None) -> str:
        return f"{split}/" if split else ""

    def images_dir_for(self, split: str | None) -> Path:
        return self.path / self._prefix(split) / IMAGES_DIR

    def masks_dir_for(self, split: str | None) -> Path:
        return self.path / self._prefix(split) / MASKS_DIR

    def data_dirs(self) -> list[Path]:
        """所有可能存放样本文件的 images/masks 目录（含各划分子目录）。"""
        return [d for split in (None, *SPLITS) for d in (self.images_dir_for(split), self.masks_dir_for(split))]

    def split_counts(self) -> dict[str, int]:
        """各划分的样本数；键 ``""`` 为未划分。"""
        counts = {"": 0, **{s: 0 for s in SPLITS}}
        for s in self.samples:
            counts[s.split or ""] += 1
        return counts

    @property
    def has_splits(self) -> bool:
        return any(s.split for s in self.samples)

    def abs(self, rel: str) -> Path:
        return self.path.joinpath(*rel.split("/"))

    def rel(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.path.resolve()).as_posix()

    def sample(self, sample_id: str) -> Sample | None:
        return next((s for s in self.samples if s.id == sample_id), None)

    def summary(self) -> DatasetSummary:
        return DatasetSummary(
            id=self.id,
            name=self.name,
            task_type=self.meta.task_type,
            n_samples=len(self.samples),
            n_annotated=sum(1 for s in self.samples if s.has_annotation and s.status == "annotated"),
            n_predicted=sum(1 for s in self.samples if s.has_annotation and s.status == "predicted"),
            n_classes=len(self.meta.foreground_classes),
            path=self.path,
            n_splits=self.split_counts(),
        )

    def save(self) -> None:
        with self._lock:
            self.meta.updated_at = now_iso()
            _write_json_atomic(self.path / DATASET_FILE, self.meta.to_dict())
            _write_json_atomic(
                self.path / SAMPLES_FILE,
                {
                    "format_version": FORMAT_VERSION,
                    "next_seq": self._next_seq,
                    "samples": [s.to_dict() for s in self.samples],
                },
            )

    def add_history(self, action: str, detail: str = "") -> None:
        self.meta.history.append({"time": now_iso(), "action": action, "detail": detail})
        del self.meta.history[:-200]

    def write_report(self, kind: str, report: dict[str, Any]) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.reports_dir / f"{kind}_{stamp}.json"
        _write_json_atomic(path, report)
        return path

    # ---------- 保护规则 ----------
    @property
    def is_binary_semantic(self) -> bool:
        return self.meta.task_type == "semantic" and len(self.meta.foreground_classes) <= 1

    def protection_reason(self) -> str | None:
        """当前二值化的标注/训练功能不能安全写入或使用本数据集时返回原因。"""
        if self.meta.task_type == "instance":
            return "实例分割数据集：当前标注与训练功能只支持二值语义掩码，为保留实例信息已禁止写入。"
        fg = self.meta.foreground_classes
        if len(fg) > 1:
            names = "、".join(c.name for c in fg)
            return f"多类别语义数据集（{names}）：当前标注与训练功能会把类别合并为二值，已禁止写入。"
        return None

    def binary_foreground_class(self) -> LabelClass:
        reason = self.protection_reason()
        if reason:
            raise DatasetProtectedError(reason)
        fg = self.meta.foreground_classes
        if not fg:
            raise DatasetError("数据集没有前景类别，请先在标签管理中添加类别。")
        return fg[0]

    # ---------- 样本 ----------
    def _new_sample_id(self) -> str:
        sid = f"s{self._next_seq:06d}"
        self._next_seq += 1
        return sid

    def _unique_image_name(self, name: str) -> str:
        # 掩码按主名存放（masks/{stem}.png），且样本可在划分间移动，因此主名在整个数据集内唯一，
        # a.png 与 a.jpg、train/a.png 与 val/a.png 都不能共存
        if self._used_names is None:
            self._used_names = {s.stem.lower() for s in self.samples}
        used = self._used_names
        stem, dot, ext = name.rpartition(".")
        if not dot:
            stem, ext = name, ""
        candidate_stem = stem
        i = 1
        while (candidate_stem.lower() in used
               or any((self.images_dir_for(sp) / (f"{candidate_stem}.{ext}" if ext else candidate_stem)).exists()
                      for sp in (None, *SPLITS))):
            candidate_stem = f"{stem}_{i}"
            i += 1
        used.add(candidate_stem.lower())
        return f"{candidate_stem}.{ext}" if ext else candidate_stem

    def mask_rel_for(self, sample: Sample) -> str:
        return f"{self._prefix(sample.split)}{MASKS_DIR}/{sample.stem}.png"

    def _instance_dir_rel(self, sample: Sample) -> str:
        return f"{self._prefix(sample.split)}{MASKS_DIR}/{sample.stem}"

    def add_image(self, src: str | Path, *, preferred_name: str | None = None, source: str | None = None,
                  split: str | None = None) -> Sample:
        """复制原图到 images/（或 <split>/images/）并新增一条未标注样本（调用方负责 save）。"""
        src = Path(src)
        split = normalize_split(split)
        name = self._unique_image_name(preferred_name or src.name)
        dst = self.images_dir_for(split) / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        try:
            w, h = mask_io.read_image_size(dst)
        except Exception:
            w = h = None
        sample = Sample(
            id=self._new_sample_id(),
            image=f"{self._prefix(split)}{IMAGES_DIR}/{name}",
            status="unlabeled",
            width=w,
            height=h,
            source=source if source is not None else str(src),
            split=split,
        )
        self.samples.append(sample)
        return sample

    def remove_sample(self, sample: Sample, delete_files: bool = True) -> None:
        if delete_files:
            self.clear_annotation(sample)
            img = self.abs(sample.image)
            if img.is_file():
                img.unlink()
        self.samples = [s for s in self.samples if s.id != sample.id]
        self._used_names = None

    def clear_annotation(self, sample: Sample) -> None:
        if sample.mask:
            p = self.abs(sample.mask)
            if p.is_file():
                p.unlink()
        for inst in sample.instances:
            p = self.abs(inst.mask)
            if p.is_file():
                p.unlink()
        inst_dir = self.abs(self._instance_dir_rel(sample))
        if sample.instances and inst_dir.is_dir() and not any(inst_dir.iterdir()):
            inst_dir.rmdir()
        sample.mask = None
        sample.instances = []
        sample.status = "unlabeled"

    def set_status(self, sample: Sample, status: str) -> None:
        if status not in ("unlabeled", "annotated", "predicted"):
            raise DatasetError(f"未知样本状态: {status}")
        if status != "unlabeled" and not sample.has_annotation:
            raise DatasetError(f"{sample.image_name} 没有掩码，不能设为 {status}")
        sample.status = status

    # ---------- 数据划分 ----------
    @staticmethod
    def _rebased(rel: str, split: str | None) -> str:
        """把 ``[<split>/]images|masks/...`` 换到目标划分下；其它位置的文件保持不动。"""
        parts = rel.split("/")
        if len(parts) > 2 and parts[0] in SPLITS and parts[1] in (IMAGES_DIR, MASKS_DIR):
            parts = parts[1:]
        if parts[0] not in (IMAGES_DIR, MASKS_DIR):
            return rel
        return "/".join(([split] if split else []) + parts)

    def set_split(self, sample: Sample, split: str | None) -> bool:
        """把样本的原图与掩码移动到目标划分目录（None 为未划分）。返回是否有变化；调用方负责 save。"""
        split = normalize_split(split)
        rels = [sample.image] + ([sample.mask] if sample.mask else []) + [r.mask for r in sample.instances]
        moves = []
        for rel in rels:
            new = self._rebased(rel, split)
            if new != rel and self.abs(rel).is_file():
                if self.abs(new).exists():
                    raise DatasetError(f"目标位置已存在同名文件，未移动 {sample.image_name}: {new}")
                moves.append((rel, new))
        if sample.split == split and not moves:
            return False
        done: list[tuple[str, str]] = []
        try:
            for old, new in moves:
                dst = self.abs(new)
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(self.abs(old), dst)
                done.append((old, new))
        except OSError as e:
            for old, new in reversed(done):
                os.replace(self.abs(new), self.abs(old))
            raise DatasetError(f"移动 {sample.image_name} 失败: {e}") from e
        old_inst_dir = self.abs(self._instance_dir_rel(sample))
        sample.image = self._rebased(sample.image, split)
        if sample.mask:
            sample.mask = self._rebased(sample.mask, split)
        for rec in sample.instances:
            rec.mask = self._rebased(rec.mask, split)
        sample.split = split
        if sample.instances and old_inst_dir.is_dir() and not any(old_inst_dir.iterdir()):
            old_inst_dir.rmdir()
        return True

    def set_splits(self, samples: Iterable[Sample], split: str | None, progress: ProgressFn | None = None,
                   cancel: Callable[[], bool] | None = None) -> int:
        """批量设置划分并保存，返回实际移动的样本数。"""
        samples = list(samples)
        changed = 0
        try:
            for i, s in enumerate(samples):
                if cancel and cancel():
                    raise RuntimeError("cancelled")
                if progress:
                    progress(i, len(samples), s.image_name)
                changed += int(self.set_split(s, split))
        finally:
            if changed:
                self.add_history("set_split", f"{changed} 个样本 -> {split_label(split)}")
            self.save()
        return changed

    def random_split(self, ratios: Mapping[str, float], seed: int = 42, samples: Iterable[Sample] | None = None,
                     progress: ProgressFn | None = None,
                     cancel: Callable[[], bool] | None = None) -> dict[str, int]:
        """按比例随机划分（同一种子结果可复现）。按样本状态分层，使各子集的已标注比例接近。

        ratios 如 ``{"train": 0.7, "val": 0.2, "test": 0.1}``，自动归一化；返回各划分分到的样本数。
        """
        ratios = {normalize_split(k): float(v) for k, v in ratios.items() if float(v) > 0}
        if not ratios:
            raise DatasetError("划分比例之和必须大于 0")
        targets = sorted(self.samples if samples is None else samples, key=lambda s: s.id)
        rng = random.Random(seed)
        plan: list[tuple[Sample, str]] = []
        for status in ("annotated", "predicted", "unlabeled"):
            group = [s for s in targets if (s.status if s.has_annotation else "unlabeled") == status]
            rng.shuffle(group)
            counts = allocate_counts(len(group), ratios)
            it = iter(group)
            for split in SPLITS:
                plan += [(next(it), split) for _ in range(counts.get(split, 0))]
        # 小数据集分层后可能某个子集为空：从样本最多的子集挪一个过去
        totals = {k: sum(1 for _, sp in plan if sp == k) for k in ratios}
        for k in ratios:
            donor = max(totals, key=totals.get)
            if totals[k] == 0 and totals[donor] > 1:
                idx = next(i for i in range(len(plan) - 1, -1, -1) if plan[i][1] == donor)
                plan[idx] = (plan[idx][0], k)
                totals[k], totals[donor] = 1, totals[donor] - 1
        moved = 0
        try:
            for i, (s, split) in enumerate(plan):
                if cancel and cancel():
                    raise RuntimeError("cancelled")
                if progress:
                    progress(i, len(plan), s.image_name)
                moved += int(self.set_split(s, split))
        finally:
            self.add_history("random_split", f"seed={seed}，比例 {ratios}；样本 {len(plan)}，移动 {moved}")
            self.save()
        if progress:
            progress(len(plan), len(plan), "划分完成")
        return {k: totals.get(k, 0) for k in SPLITS}

    # ---------- 掩码读写 ----------
    def set_semantic_mask(self, sample: Sample, index: np.ndarray, status: str = "annotated") -> None:
        if self.meta.task_type != "semantic":
            raise DatasetProtectedError("实例数据集不能写入语义掩码")
        rel = sample.mask or self.mask_rel_for(sample)
        mask_io.write_label_mask(self.abs(rel), index)
        sample.mask = rel
        sample.status = status

    def set_instances(self, sample: Sample, items: Iterable[tuple[int, np.ndarray]], status: str = "annotated") -> None:
        """写入实例：items 为 (class_id, 二值掩码)。旧实例文件会被清除。"""
        if self.meta.task_type != "instance":
            raise DatasetError("语义数据集不能写入实例掩码")
        self.clear_annotation(sample)
        records: list[InstanceRecord] = []
        for k, (class_id, mask) in enumerate(items, start=1):
            rel = f"{self._instance_dir_rel(sample)}/inst_{k:03d}.png"
            mask_io.write_binary_mask(self.abs(rel), mask)
            records.append(InstanceRecord(id=k, class_id=int(class_id), mask=rel))
        sample.instances = records
        sample.status = status if records else "unlabeled"

    def assign_mask_file(self, sample: Sample, src: str | Path, mapping: Mapping[str, int] | None = None,
                         status: str = "annotated") -> dict[str, int]:
        """把外部掩码按映射转为类别编号图后配对到样本。返回未映射的值（存在时拒绝写入）。"""
        mapping = dict(mapping if mapping is not None else self.meta.value_mapping)
        raw = mask_io.read_raw_mask(src)
        index, unknown = mask_io.apply_mapping(raw, mapping)
        if unknown:
            return unknown
        self.set_semantic_mask(sample, index, status=status)
        return {}

    def load_index_mask(self, sample: Sample) -> np.ndarray | None:
        """语义返回编号图；实例返回按类别合成的编号图（仅用于显示）。"""
        if sample.mask:
            p = self.abs(sample.mask)
            return mask_io.read_label_mask(p) if p.is_file() else None
        if sample.instances:
            out: np.ndarray | None = None
            for rec, m in self.instance_masks(sample):
                if out is None:
                    out = np.zeros(m.shape, dtype=np.int64)
                out[m] = rec.class_id
            return out
        return None

    def instance_masks(self, sample: Sample) -> list[tuple[InstanceRecord, np.ndarray]]:
        out = []
        for rec in sample.instances:
            p = self.abs(rec.mask)
            if p.is_file():
                out.append((rec, mask_io.read_label_mask(p) > 0))
        return out

    def save_binary_annotation(self, sample_id: str, foreground: np.ndarray) -> Sample:
        """二值标注工具的保存入口：只对二值语义数据集写入前景类别编号。"""
        sample = self.sample(sample_id)
        if sample is None:
            raise DatasetError(f"样本不存在: {sample_id}")
        fg_class = self.binary_foreground_class()
        if sample.mask and self.abs(sample.mask).is_file():
            existing = mask_io.read_label_mask(self.abs(sample.mask))
            extra = set(np.unique(existing).tolist()) - {BACKGROUND_ID, fg_class.id}
            if extra:
                raise DatasetProtectedError(
                    f"已有掩码含未定义标签 {sorted(extra)}，保存会覆盖这些标签。请先运行数据校验并修正标签映射。"
                )
        index = np.where(np.asarray(foreground) > 0, fg_class.id, BACKGROUND_ID)
        self.set_semantic_mask(sample, index, status="annotated")
        self.save()
        return sample

    # ---------- 标签管理 ----------
    def add_class(self, name: str, color: tuple[int, int, int] | None = None) -> LabelClass:
        name = name.strip()
        if not name:
            raise DatasetError("类别名称不能为空")
        if any(c.name == name for c in self.meta.classes):
            raise DatasetError(f"类别名称已存在: {name}")
        cid = self.meta.next_class_id()
        cls = LabelClass(cid, name, tuple(color) if color else default_color(cid))
        self.meta.classes.append(cls)
        self.add_history("add_class", f"{cid}={name}")
        return cls

    def rename_class(self, class_id: int, name: str) -> None:
        name = name.strip()
        cls = self.meta.class_by_id(class_id)
        if cls is None:
            raise DatasetError(f"类别不存在: {class_id}")
        if not name:
            raise DatasetError("类别名称不能为空")
        if any(c.name == name and c.id != class_id for c in self.meta.classes):
            raise DatasetError(f"类别名称已存在: {name}")
        self.add_history("rename_class", f"{class_id}: {cls.name} -> {name}")
        cls.name = name

    def set_class_color(self, class_id: int, color: tuple[int, int, int]) -> None:
        cls = self.meta.class_by_id(class_id)
        if cls is None:
            raise DatasetError(f"类别不存在: {class_id}")
        cls.color = tuple(int(c) for c in color[:3])

    def class_usage(self, progress: ProgressFn | None = None,
                    cancel: Callable[[], bool] | None = None) -> dict[int, dict[str, int]]:
        """统计每个标签值出现的样本数与像素数（含未定义的值）。"""
        usage: dict[int, dict[str, int]] = {}
        annotated = [s for s in self.samples if s.has_annotation]
        for i, s in enumerate(annotated):
            if cancel and cancel():
                raise RuntimeError("cancelled")
            if progress:
                progress(i, len(annotated), s.image_name)
            if s.mask:
                p = self.abs(s.mask)
                if not p.is_file():
                    continue
                vals, counts = np.unique(mask_io.read_label_mask(p), return_counts=True)
                for v, c in zip(vals.tolist(), counts.tolist()):
                    u = usage.setdefault(int(v), {"samples": 0, "pixels": 0, "instances": 0})
                    u["samples"] += 1
                    u["pixels"] += int(c)
            else:
                seen: set[int] = set()
                for rec, m in self.instance_masks(s):
                    u = usage.setdefault(rec.class_id, {"samples": 0, "pixels": 0, "instances": 0})
                    u["instances"] += 1
                    u["pixels"] += int(m.sum())
                    if rec.class_id not in seen:
                        u["samples"] += 1
                        seen.add(rec.class_id)
        if progress:
            progress(len(annotated), len(annotated), "完成")
        return usage

    def merge_preview(self, source_ids: Iterable[int], target_id: int,
                      usage: Mapping[int, Mapping[str, int]] | None = None) -> dict[str, Any]:
        src = sorted({int(x) for x in source_ids} - {int(target_id)})
        usage = usage if usage is not None else self.class_usage()
        per = {cid: dict(usage.get(cid, {"samples": 0, "pixels": 0, "instances": 0})) for cid in src}
        affected = set()
        for s in self.samples:
            if s.instances and any(r.class_id in src for r in s.instances):
                affected.add(s.id)
        return {
            "sources": src,
            "target": int(target_id),
            "per_class": per,
            "pixels": sum(p["pixels"] for p in per.values()),
            "instances": sum(p.get("instances", 0) for p in per.values()),
            "samples_upper_bound": sum(p["samples"] for p in per.values()),
            "instance_samples": len(affected),
        }

    def merge_classes(self, source_ids: Iterable[int], target_id: int, progress: ProgressFn | None = None,
                      cancel: Callable[[], bool] | None = None) -> dict[str, int]:
        """把若干类别合并到目标类别并重写掩码；源类别从定义中移除（背景不可作为源）。"""
        src = sorted({int(x) for x in source_ids} - {int(target_id)})
        if not src:
            raise DatasetError("请选择至少一个与目标不同的源类别")
        if BACKGROUND_ID in src:
            raise DatasetError("背景类别不能被合并到其他类别")
        if self.meta.class_by_id(target_id) is None:
            raise DatasetError(f"目标类别不存在: {target_id}")
        changed_samples = changed_pixels = 0
        annotated = [s for s in self.samples if s.has_annotation]
        for i, s in enumerate(annotated):
            if cancel and cancel():
                raise RuntimeError("cancelled")
            if progress:
                progress(i, len(annotated), s.image_name)
            if s.mask:
                p = self.abs(s.mask)
                if not p.is_file():
                    continue
                index = mask_io.read_label_mask(p).astype(np.int64)
                sel = np.isin(index, src)
                n = int(sel.sum())
                if n:
                    index[sel] = target_id
                    mask_io.write_label_mask(p, index)
                    changed_samples += 1
                    changed_pixels += n
            else:
                hit = False
                for rec in s.instances:
                    if rec.class_id in src:
                        rec.class_id = int(target_id)
                        hit = True
                changed_samples += int(hit)
        self.meta.classes = [c for c in self.meta.classes if c.id not in src]
        self.meta.value_mapping = {k: (target_id if v in src else v) for k, v in self.meta.value_mapping.items()}
        self.add_history("merge_classes", f"{src} -> {target_id}; 样本 {changed_samples}，像素 {changed_pixels}")
        self.save()
        if progress:
            progress(len(annotated), len(annotated), "完成")
        return {"samples": changed_samples, "pixels": changed_pixels}

    def delete_class(self, class_id: int, usage: Mapping[int, Mapping[str, int]] | None = None) -> None:
        if class_id == BACKGROUND_ID:
            raise DatasetError("背景类别不能删除")
        usage = usage if usage is not None else self.class_usage()
        if usage.get(class_id, {}).get("pixels", 0) or any(
            r.class_id == class_id for s in self.samples for r in s.instances
        ):
            raise DatasetError("该类别仍被掩码使用，请先将其合并到背景或其他类别")
        cls = self.meta.class_by_id(class_id)
        self.meta.classes = [c for c in self.meta.classes if c.id != class_id]
        self.meta.value_mapping = {k: v for k, v in self.meta.value_mapping.items() if v != class_id}
        self.add_history("delete_class", f"{class_id}={cls.name if cls else ''}")


class DatasetStore:
    """``<root>/<dataset_id>/`` 形式的多数据集仓库。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def _lock(self) -> threading.RLock:
        return _lock_for(self.root)

    def dataset_path(self, dataset_id: str) -> Path:
        if not dataset_id or any(ch in dataset_id for ch in "/\\:") or dataset_id.startswith("."):
            raise DatasetError(f"非法数据集 ID: {dataset_id!r}")
        return self.root / dataset_id

    def exists(self, dataset_id: str) -> bool:
        try:
            return (self.dataset_path(dataset_id) / DATASET_FILE).is_file()
        except DatasetError:
            return False

    def list_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and (p / DATASET_FILE).is_file())

    def list_summaries(self) -> list[DatasetSummary]:
        out = []
        for did in self.list_ids():
            try:
                out.append(self.load(did).summary())
            except Exception:
                continue
        return sorted(out, key=lambda s: s.name.lower())

    def find_legacy_dirs(self) -> list[Path]:
        """数据集根目录下未登记（无 dataset.json）的旧数据目录。"""
        if not self.root.is_dir():
            return []
        return sorted(
            p for p in self.root.iterdir()
            if p.is_dir() and not p.name.startswith((".", "_")) and not (p / DATASET_FILE).is_file()
        )

    def load(self, dataset_id: str) -> Dataset:
        path = self.dataset_path(dataset_id)
        meta_path = path / DATASET_FILE
        if not meta_path.is_file():
            raise DatasetError(f"数据集不存在: {dataset_id}")
        with self._lock:
            meta = DatasetMeta.from_dict(_read_json(meta_path))
            samples_path = path / SAMPLES_FILE
            raw = _read_json(samples_path) if samples_path.is_file() else {}
        samples = [Sample.from_dict(d) for d in raw.get("samples", [])]
        next_seq = int(raw.get("next_seq", len(samples) + 1))
        return Dataset(path, meta, samples, next_seq)

    def _new_id(self) -> str:
        while True:
            did = f"ds-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
            if not (self.root / did).exists():
                return did

    def create(self, name: str, task_type: str = "semantic", classes: list[LabelClass] | None = None,
               description: str = "", value_mapping: Mapping[str, int] | None = None,
               provenance: dict[str, Any] | None = None) -> Dataset:
        name = name.strip()
        if not name:
            raise DatasetError("数据集名称不能为空")
        if task_type not in TASK_TYPES:
            raise DatasetError(f"未知分割类型: {task_type}")
        classes = [LabelClass(c.id, c.name, tuple(c.color)) for c in (classes or default_classes())]
        if not any(c.id == BACKGROUND_ID for c in classes):
            classes.insert(0, LabelClass(BACKGROUND_ID, "背景", default_color(0)))
        with self._lock:
            did = self._new_id()
            path = self.root / did
            (path / IMAGES_DIR).mkdir(parents=True)
            (path / MASKS_DIR).mkdir()
            ts = now_iso()
            if value_mapping is None:
                value_mapping = {"0": 0, **{str(c.id): c.id for c in classes if c.id != BACKGROUND_ID}}
                fg = [c for c in classes if c.id != BACKGROUND_ID]
                if len(fg) == 1:
                    value_mapping["255"] = fg[0].id
            meta = DatasetMeta(
                id=did, name=name, task_type=task_type, classes=classes, description=description,
                created_at=ts, updated_at=ts, value_mapping=dict(value_mapping), provenance=provenance,
            )
            ds = Dataset(path, meta, [], 1)
            ds.add_history("create", name)
            ds.save()
            return ds

    def rename(self, dataset_id: str, name: str) -> Dataset:
        name = name.strip()
        if not name:
            raise DatasetError("数据集名称不能为空")
        with self._lock:
            ds = self.load(dataset_id)
            ds.add_history("rename", f"{ds.meta.name} -> {name}")
            ds.meta.name = name
            ds.save()
            return ds

    def copy(self, dataset_id: str, name: str | None = None) -> Dataset:
        with self._lock:
            src = self.load(dataset_id)
            did = self._new_id()
            dst_path = self.root / did
            shutil.copytree(src.path, dst_path)
            ds = self.load_from_path(dst_path)
            ds.meta.id = did
            ds.meta.name = (name or f"{src.name} 副本").strip()
            ds.meta.created_at = now_iso()
            ds.meta.provenance = {
                "kind": "copy", "source_dataset_id": src.id, "source_name": src.name, "time": now_iso(),
            }
            ds.add_history("copy", f"from {src.id}")
            ds.save()
            return ds

    def load_from_path(self, path: Path) -> Dataset:
        meta = DatasetMeta.from_dict(_read_json(path / DATASET_FILE))
        raw = _read_json(path / SAMPLES_FILE) if (path / SAMPLES_FILE).is_file() else {}
        samples = [Sample.from_dict(d) for d in raw.get("samples", [])]
        return Dataset(path, meta, samples, int(raw.get("next_seq", len(samples) + 1)))

    def delete(self, dataset_id: str) -> None:
        path = self.dataset_path(dataset_id)
        if not (path / DATASET_FILE).is_file():
            raise DatasetError(f"数据集不存在: {dataset_id}")
        with self._lock:
            shutil.rmtree(path)
