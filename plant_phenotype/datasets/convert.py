# -*- coding: utf-8 -*-
"""格式转换：默认生成新数据集，保留源数据并写入转换记录。

- 实例 → 语义：同一像素被多个实例覆盖时后者优先，个体区分会丢失；
- 语义 → 实例：只能按连通域生成候选结果（状态为“预测/候选”），需人工复核；
- 导出：写出同目录 ``*_mask.png`` 或 ``images/masks`` 布局，掩码可为类别编号、彩色或 0/255 二值；
  已划分的样本默认写到 ``train/``、``val/``、``test/`` 子目录下。

转换生成的新数据集沿用源样本的划分。
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from plant_phenotype.datasets import mask_io
from plant_phenotype.datasets.schema import BACKGROUND_ID, LabelClass
from plant_phenotype.datasets.store import Dataset, DatasetError, DatasetStore, ProgressFn, now_iso

CancelFn = Callable[[], bool]

CONVERSIONS = {
    "instance_to_semantic": "实例 → 语义（个体区分会丢失）",
    "semantic_to_instance": "语义 → 实例候选（按连通域）",
}
EXPORT_ENCODINGS = {
    "index": "类别编号（0、1、2…）",
    "color": "彩色（按类别颜色）",
    "binary": "二值 0/255（所有前景合并）",
}
EXPORT_LAYOUTS = {
    "split_dir": "images / masks 分目录",
    "same_dir": "同目录（原图 + *_mask.png）",
}


def _check_cancel(cancel: CancelFn | None) -> None:
    if cancel and cancel():
        raise RuntimeError("cancelled")


def _derive(store: DatasetStore, src: Dataset, name: str, task_type: str, kind: str,
            params: dict[str, Any]) -> Dataset:
    return store.create(
        name, task_type=task_type,
        classes=[LabelClass(c.id, c.name, c.color) for c in src.meta.classes],
        description=src.meta.description,
        value_mapping=src.meta.value_mapping,
        provenance={
            "kind": kind, "source_dataset_id": src.id, "source_name": src.name,
            "params": params, "time": now_iso(),
        },
    )


def _finish(dst: Dataset, kind: str, report: dict[str, Any]) -> dict[str, Any]:
    dst.add_history(kind, f"from {report['source_dataset_id']}；样本 {report['samples']}，失败 {len(report['failed'])}")
    dst.save()
    report["report_path"] = str(dst.write_report(kind, report))
    return report


def instance_to_semantic(store: DatasetStore, src_id: str, name: str, progress: ProgressFn | None = None,
                         cancel: CancelFn | None = None) -> tuple[Dataset, dict[str, Any]]:
    src = store.load(src_id)
    if src.meta.task_type != "instance":
        raise DatasetError("源数据集不是实例分割类型")
    dst = _derive(store, src, name, "semantic", "instance_to_semantic", {"overlap": "later_wins"})
    report: dict[str, Any] = {
        "source_dataset_id": src.id, "target_dataset_id": dst.id, "samples": 0,
        "overlap_pixels": 0, "failed": [],
        "note": "实例转语义会丢失个体区分；重叠像素取编号较大的实例的类别。",
    }
    for i, s in enumerate(src.samples):
        _check_cancel(cancel)
        if progress:
            progress(i, len(src.samples), s.image_name)
        try:
            d = dst.add_image(src.abs(s.image), preferred_name=s.image_name, source=s.source, split=s.split)
            inst = src.instance_masks(s)
            if inst:
                index = np.zeros(inst[0][1].shape, dtype=np.int64)
                cover = np.zeros(index.shape, dtype=np.int32)
                for rec, m in inst:
                    index[m] = rec.class_id
                    cover += m
                report["overlap_pixels"] += int((cover > 1).sum())
                dst.set_semantic_mask(d, index, status=s.status if s.status != "unlabeled" else "annotated")
            report["samples"] += 1
        except Exception as e:
            report["failed"].append({"sample": s.id, "error": str(e)})
    return dst, _finish(dst, "instance_to_semantic", report)


def semantic_to_instance(store: DatasetStore, src_id: str, name: str, min_area: int = 20, connectivity: int = 8,
                         progress: ProgressFn | None = None,
                         cancel: CancelFn | None = None) -> tuple[Dataset, dict[str, Any]]:
    src = store.load(src_id)
    if src.meta.task_type != "semantic":
        raise DatasetError("源数据集不是语义分割类型")
    if connectivity not in (4, 8):
        raise ValueError("连通性只能为 4 或 8")
    dst = _derive(store, src, name, "instance", "semantic_to_instance",
                  {"min_area": int(min_area), "connectivity": int(connectivity)})
    report: dict[str, Any] = {
        "source_dataset_id": src.id, "target_dataset_id": dst.id, "samples": 0, "instances": 0,
        "dropped_small": 0, "failed": [],
        "note": "语义转实例不能可靠还原对象：结果为连通域候选，状态标记为“预测/候选”，需人工复核。",
    }
    for i, s in enumerate(src.samples):
        _check_cancel(cancel)
        if progress:
            progress(i, len(src.samples), s.image_name)
        try:
            d = dst.add_image(src.abs(s.image), preferred_name=s.image_name, source=s.source, split=s.split)
            index = src.load_index_mask(s)
            if index is not None:
                items = []
                for cid in sorted(set(np.unique(index).tolist()) - {BACKGROUND_ID}):
                    n, labels, stats, _ = cv2.connectedComponentsWithStats(
                        (index == cid).astype(np.uint8), connectivity=connectivity
                    )
                    for k in range(1, n):
                        if stats[k, cv2.CC_STAT_AREA] < min_area:
                            report["dropped_small"] += 1
                            continue
                        items.append((int(cid), labels == k))
                dst.set_instances(d, items, status="predicted")
                report["instances"] += len(items)
            report["samples"] += 1
        except Exception as e:
            report["failed"].append({"sample": s.id, "error": str(e)})
    return dst, _finish(dst, "semantic_to_instance", report)


def export_dataset(ds: Dataset, out_dir: str | Path, layout: str = "split_dir", encoding: str = "index",
                   mask_suffix: str = "_mask", only_annotated: bool = False, keep_splits: bool = True,
                   progress: ProgressFn | None = None, cancel: CancelFn | None = None) -> dict[str, Any]:
    """导出为外部目录格式（实例数据集按类别合成编号图后导出）。

    keep_splits 时已划分样本写到 ``<out>/<split>/``（其下再按 layout 组织），未划分样本仍在 ``<out>/`` 下。
    """
    if layout not in EXPORT_LAYOUTS:
        raise ValueError(f"未知导出布局: {layout}")
    if encoding not in EXPORT_ENCODINGS:
        raise ValueError(f"未知掩码编码: {encoding}")
    if layout == "same_dir" and not mask_suffix:
        raise ValueError("同目录布局需要掩码后缀")
    out = Path(out_dir)

    def dirs_for(split: str | None) -> tuple[Path, Path]:
        base = out / split if (keep_splits and split) else out
        if layout == "split_dir":
            return base / "images", base / "masks"
        return base, base

    samples = [s for s in ds.samples if s.has_annotation or not only_annotated]
    report: dict[str, Any] = {
        "dataset_id": ds.id, "out_dir": str(out), "layout": layout, "encoding": encoding,
        "keep_splits": keep_splits, "images": 0, "masks": 0, "failed": [],
    }
    for i, s in enumerate(samples):
        _check_cancel(cancel)
        if progress:
            progress(i, len(samples), s.image_name)
        img_dir, mask_dir = dirs_for(s.split)
        try:
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ds.abs(s.image), img_dir / s.image_name)
            report["images"] += 1
            index = ds.load_index_mask(s)
            if index is None:
                continue
            mask_path = mask_dir / f"{s.stem}{mask_suffix}.png"
            if encoding == "index":
                mask_io.write_label_mask(mask_path, index)
            elif encoding == "color":
                mask_io.write_color_mask(mask_path, index, ds.meta.classes)
            else:
                mask_io.write_binary_mask(mask_path, index != BACKGROUND_ID)
            report["masks"] += 1
        except Exception as e:
            report["failed"].append({"sample": s.id, "error": str(e)})
    if progress:
        progress(len(samples), len(samples), "导出完成")
    return report
