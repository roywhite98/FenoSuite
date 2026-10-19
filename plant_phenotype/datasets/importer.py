# -*- coding: utf-8 -*-
"""导入：把外部原图/掩码复制到软件数据目录，并按映射统一为类别编号图。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from plant_phenotype.datasets import mask_io
from plant_phenotype.datasets.pairing import PairingRules, discover_pairs, index_masks
from plant_phenotype.datasets.schema import BACKGROUND_ID, LabelClass, default_color
from plant_phenotype.datasets.store import Dataset, DatasetStore, ProgressFn, now_iso

CancelFn = Callable[[], bool]


def _check_cancel(cancel: CancelFn | None) -> None:
    if cancel and cancel():
        raise RuntimeError("cancelled")


@dataclass
class MaskScan:
    """掩码中出现的像素值/颜色统计。"""

    keys: dict[str, dict[str, int]] = field(default_factory=dict)
    is_color: bool = False
    errors: list[tuple[str, str]] = field(default_factory=list)
    n_files: int = 0

    @property
    def kind(self) -> str:
        if self.is_color:
            return "color"
        vals = {int(k) for k in self.keys}
        if vals <= {0, 255} or vals <= {0, 1}:
            return "binary"
        return "index"


def scan_mask_values(masks: Iterable[Path], progress: ProgressFn | None = None,
                     cancel: CancelFn | None = None) -> MaskScan:
    masks = list(masks)
    scan = MaskScan(n_files=len(masks))
    for i, path in enumerate(masks):
        _check_cancel(cancel)
        if progress:
            progress(i, len(masks), path.name)
        try:
            raw = mask_io.read_raw_mask(path)
            keys = mask_io.unique_keys(raw)
        except Exception as e:
            scan.errors.append((str(path), str(e)))
            continue
        if raw.ndim == 3:
            scan.is_color = True
        for k, n in keys.items():
            entry = scan.keys.setdefault(k, {"pixels": 0, "files": 0})
            entry["pixels"] += n
            entry["files"] += 1
    if scan.is_color:
        # 混合灰度与彩色掩码时，把灰度值统一表示为灰色键
        scan.keys = {
            (k if mask_io.is_color_key(k) else mask_io.color_key((int(k),) * 3)): v for k, v in scan.keys.items()
        }
    if progress:
        progress(len(masks), len(masks), "扫描完成")
    return scan


def suggest_mapping(scan: MaskScan) -> tuple[dict[str, int], list[LabelClass]]:
    """根据扫描结果给出默认映射：二值 → 背景/根；编号 → 原值为类别 ID；彩色 → 黑色为背景，其余按像素数排序。"""
    classes = [LabelClass(BACKGROUND_ID, "背景", default_color(BACKGROUND_ID))]
    mapping: dict[str, int] = {}
    if not scan.keys:
        classes.append(LabelClass(1, "根", default_color(1)))
        return {"0": 0, "255": 1, "1": 1}, classes
    if scan.is_color:
        next_id = 1
        for key, _ in sorted(scan.keys.items(), key=lambda kv: -kv[1]["pixels"]):
            if key == "#000000":
                mapping[key] = BACKGROUND_ID
                continue
            mapping[key] = next_id
            classes.append(LabelClass(next_id, f"类别{next_id}", mask_io.parse_color_key(key)))
            next_id += 1
        return mapping, classes
    if scan.kind == "binary":
        classes.append(LabelClass(1, "根", default_color(1)))
        for key in scan.keys:
            mapping[key] = 0 if int(key) == 0 else 1
        return mapping, classes
    for key in sorted(scan.keys, key=int):
        v = int(key)
        mapping[key] = v
        if v != BACKGROUND_ID:
            classes.append(LabelClass(v, f"类别{v}", default_color(v)))
    return mapping, classes


@dataclass
class ImportOptions:
    name: str
    rules: PairingRules = field(default_factory=PairingRules)
    mapping: dict[str, int] = field(default_factory=dict)
    classes: list[LabelClass] = field(default_factory=list)
    task_type: str = "semantic"
    mask_status: str = "annotated"
    include_unpaired: bool = True
    description: str = ""
    keep_splits: bool = True


def _validate_mapping(mapping: Mapping[str, int], classes: list[LabelClass]) -> None:
    ids = {c.id for c in classes}
    bad = sorted({v for v in mapping.values() if v not in ids})
    if bad:
        raise ValueError(f"映射指向未定义的类别 ID: {bad}")


def import_dataset(store: DatasetStore, source: str | Path, options: ImportOptions,
                   progress: ProgressFn | None = None, cancel: CancelFn | None = None,
                   log: Callable[[str], None] | None = None) -> tuple[Dataset, dict[str, Any]]:
    """扫描源目录、建立新数据集并复制文件。存在未映射像素值的掩码不写入，记入失败报告。"""
    source = Path(source)
    if options.task_type != "semantic":
        raise ValueError("目录导入目前只生成语义分割数据集；实例数据可由语义数据集转换得到")
    classes = options.classes or suggest_mapping(MaskScan())[1]
    _validate_mapping(options.mapping, classes)
    result = discover_pairs(source, options.rules)
    if log:
        log(f"布局: {result.layout}；原图 {len(result.pairs)} 张，已配对 {result.n_paired} 张，"
            f"未匹配掩码 {len(result.orphan_masks)} 个")
        if result.split_counts:
            log(f"源目录划分: {result.split_counts}" + ("" if options.keep_splits else "（不保留）"))
    pairs = result.pairs if options.include_unpaired else [p for p in result.pairs if p.mask is not None]
    if not pairs:
        raise ValueError(f"源目录中未找到可导入的图像: {source}")
    ds = store.create(
        options.name, task_type="semantic", classes=classes, description=options.description,
        value_mapping=options.mapping,
        provenance={
            "kind": "import", "source": str(source), "time": now_iso(),
            "rules": options.rules.to_dict(), "layout": result.layout,
            "source_splits": result.split_counts, "keep_splits": options.keep_splits,
        },
    )
    report: dict[str, Any] = {
        "source": str(source), "layout": result.layout, "dataset_id": ds.id,
        "images": 0, "masks": 0, "unpaired_images": [], "orphan_masks": [str(p) for p in result.orphan_masks],
        "conflicts": result.conflicts, "failed": [], "size_mismatch": [],
    }
    name_counts = Counter(p.image.name.lower() for p in pairs)
    try:
        for i, pair in enumerate(pairs):
            _check_cancel(cancel)
            if progress:
                progress(i, len(pairs), pair.image.name)
            preferred = pair.image.name
            if pair.group and name_counts[pair.image.name.lower()] > 1:
                preferred = f"{pair.group}_{pair.image.name}"
            try:
                sample = ds.add_image(pair.image, preferred_name=preferred,
                                      split=pair.split if options.keep_splits else None)
            except Exception as e:
                report["failed"].append({"file": str(pair.image), "error": f"复制原图失败: {e}"})
                continue
            report["images"] += 1
            if pair.mask is None:
                report["unpaired_images"].append(str(pair.image))
                continue
            try:
                unknown = ds.assign_mask_file(sample, pair.mask, options.mapping, status=options.mask_status)
            except Exception as e:
                report["failed"].append({"file": str(pair.mask), "error": f"读取掩码失败: {e}"})
                continue
            if unknown:
                report["failed"].append({"file": str(pair.mask), "error": f"存在未映射的像素值: {sorted(unknown)}"})
                continue
            report["masks"] += 1
            if sample.width is not None:
                try:
                    mw, mh = mask_io.read_image_size(pair.mask)
                    if (mw, mh) != (sample.width, sample.height):
                        report["size_mismatch"].append(
                            {"sample": sample.id, "image": [sample.width, sample.height], "mask": [mw, mh]}
                        )
                except Exception:
                    pass
            if i % 50 == 49:
                ds.save()
    finally:
        ds.add_history("import", f"{source}；原图 {report['images']}，掩码 {report['masks']}，失败 {len(report['failed'])}")
        ds.save()
    if ds.has_splits:
        report["splits"] = ds.split_counts()
    report_path = ds.write_report("import", report)
    report["report_path"] = str(report_path)
    if progress:
        progress(len(pairs), len(pairs), "导入完成")
    return ds, report


def add_images(ds: Dataset, files: Iterable[Path], progress: ProgressFn | None = None,
               cancel: CancelFn | None = None, splits: Mapping[str, str | None] | None = None) -> list[str]:
    """splits 以原文件路径（str）为键指定样本划分，未列出的为未划分。"""
    files = list(files)
    splits = splits or {}
    added = []
    for i, f in enumerate(files):
        _check_cancel(cancel)
        if progress:
            progress(i, len(files), Path(f).name)
        added.append(ds.add_image(Path(f), split=splits.get(str(f))).id)
    ds.add_history("add_images", f"{len(added)} 张")
    ds.save()
    return added


def pair_masks_from_dir(ds: Dataset, mask_dir: str | Path, mask_suffix: str = "_mask", recursive: bool = True,
                        mapping: Mapping[str, int] | None = None, overwrite: bool = False,
                        only_preview: bool = False, progress: ProgressFn | None = None,
                        cancel: CancelFn | None = None, status: str = "annotated",
                        protect_annotated: bool = False, history_kind: str = "pair_masks") -> dict[str, Any]:
    """按规则把外部掩码目录批量配对到已有样本（按样本图像名或来源文件名匹配）。

    overwrite 允许覆盖已有掩码；protect_annotated 时人工标注（状态为已标注）的样本始终跳过。
    """
    conflicts: list[str] = []
    index = index_masks(Path(mask_dir), mask_suffix, recursive, conflicts)
    matches: list[tuple[str, Path]] = []
    skipped = 0
    unmatched = 0
    for s in ds.samples:
        keys = [s.stem.lower()]
        if s.source:
            keys.append(Path(s.source).stem.lower())
        m = next((index[k] for k in keys if k in index), None)
        if m is None:
            unmatched += 1
            continue
        manual = s.has_annotation and s.status == "annotated"
        if (s.has_annotation and not overwrite) or (protect_annotated and manual):
            skipped += 1
            continue
        matches.append((s.id, m))
    report: dict[str, Any] = {
        "mask_dir": str(mask_dir), "matched": len(matches), "conflicts": conflicts,
        "skipped_annotated": skipped, "unmatched_samples": unmatched,
        "failed": [], "written": 0,
    }
    if only_preview:
        report["preview"] = [{"sample": sid, "mask": str(p)} for sid, p in matches[:500]]
        return report
    for i, (sid, path) in enumerate(matches):
        _check_cancel(cancel)
        if progress:
            progress(i, len(matches), path.name)
        sample = ds.sample(sid)
        try:
            unknown = ds.assign_mask_file(sample, path, mapping, status=status)
        except Exception as e:
            report["failed"].append({"file": str(path), "error": str(e)})
            continue
        if unknown:
            report["failed"].append({"file": str(path), "error": f"存在未映射的像素值: {sorted(unknown)}"})
            continue
        report["written"] += 1
    ds.add_history(history_kind, f"{mask_dir}；写入 {report['written']}，失败 {len(report['failed'])}")
    ds.save()
    report["report_path"] = str(ds.write_report("pair" if history_kind == "pair_masks" else history_kind, report))
    return report


def _prediction_mapping(ds: Dataset) -> dict[str, int]:
    fg = ds.binary_foreground_class()
    return {"0": BACKGROUND_ID, "1": fg.id, "255": fg.id}


def write_back_predictions(ds: Dataset, mask_dir: str | Path, mask_suffix: str = "_mask",
                           info: str = "", progress: ProgressFn | None = None,
                           cancel: CancelFn | None = None) -> dict[str, Any]:
    """把二值分割结果写回数据集：只填充未标注/预测样本（状态=预测），人工标注不覆盖。"""
    mapping = _prediction_mapping(ds)  # 非二值语义数据集在此抛 DatasetProtectedError
    report = pair_masks_from_dir(
        ds, mask_dir, mask_suffix, recursive=True, mapping=mapping, overwrite=True,
        progress=progress, cancel=cancel, status="predicted", protect_annotated=True,
        history_kind="predict",
    )
    if info:
        report["info"] = info
    return report


def predictions_to_dataset(store: DatasetStore, images: Iterable[Path], mask_dir: str | Path, name: str,
                           mask_suffix: str = "_mask", provenance: dict[str, Any] | None = None,
                           progress: ProgressFn | None = None, cancel: CancelFn | None = None,
                           splits: Mapping[str, str | None] | None = None) -> tuple[Dataset, dict[str, Any]]:
    """用原图与二值分割结果新建数据集（背景/根，状态=预测），便于复核后再用于训练。

    splits 为原图路径 → 划分，用于沿用源数据集的 train/val/test。
    """
    images = list(images)
    prov = {"kind": "predict", "time": now_iso()}
    prov.update(provenance or {})
    ds = store.create(name, task_type="semantic", provenance=prov)
    n = len(images)

    def step(offset: int):
        if progress is None:
            return None
        return lambda i, total, msg: progress(offset + i, 2 * n, msg)

    add_images(ds, images, progress=step(0), cancel=cancel, splits=splits)
    report = pair_masks_from_dir(
        ds, mask_dir, mask_suffix, recursive=True, mapping=_prediction_mapping(ds), overwrite=True,
        progress=step(n), cancel=cancel, status="predicted", history_kind="predict",
    )
    return ds, report
