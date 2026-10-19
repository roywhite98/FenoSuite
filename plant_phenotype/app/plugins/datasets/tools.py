# -*- coding: utf-8 -*-
"""数据集后台任务：导入、校验、转换、导出、合并类别、批量配对、添加图像、分割结果入库。"""
from __future__ import annotations

from pathlib import Path

from plant_phenotype.app.data_context import DataContext
from plant_phenotype.app.task_models import CancelToken, ToolResult, ToolSpec
from plant_phenotype.datasets import convert, importer, validation
from plant_phenotype.datasets.pairing import PairingRules
from plant_phenotype.datasets.schema import LabelClass, normalize_split, split_label
from plant_phenotype.datasets.store import DatasetStore


def _store(inputs: dict) -> DatasetStore:
    root = str(inputs.get("datasets_root", "")).strip()
    if not root:
        raise ValueError("缺少 datasets_root")
    return DatasetStore(root)


def _progress(reporter):
    return lambda cur, total, msg: reporter.progress(cur, total, msg)


def _spec(tool_id: str, name: str, description: str, required: list[str]) -> ToolSpec:
    return ToolSpec(
        id=tool_id,
        name=name,
        category="datasets",
        description=description,
        input_schema={"type": "object", "required": ["datasets_root", *required]},
        output_schema={"type": "object"},
        long_running=True,
        result_viewer="log",
    )


def _failed_summary(report: dict) -> str:
    n = len(report.get("failed", []))
    return f"，失败 {n} 项（详见报告）" if n else ""


class ImportDatasetTool:
    spec = _spec("datasets.import", "导入数据集", "扫描外部目录、自动配对并复制到软件数据目录", ["source_dir", "name"])

    def validate(self, inputs: dict, context: DataContext) -> None:
        _store(inputs)
        if not Path(str(inputs.get("source_dir", ""))).is_dir():
            raise ValueError(f"源目录不存在: {inputs.get('source_dir')}")
        if not str(inputs.get("name", "")).strip():
            raise ValueError("数据集名称不能为空")

    def run(self, inputs: dict, context: DataContext, reporter, cancel: CancelToken) -> ToolResult:
        options = importer.ImportOptions(
            name=str(inputs["name"]),
            rules=PairingRules.from_dict(inputs.get("rules")),
            mapping={str(k): int(v) for k, v in (inputs.get("mapping") or {}).items()},
            classes=[LabelClass.from_dict(c) for c in inputs.get("classes", [])],
            mask_status=str(inputs.get("mask_status", "annotated")),
            include_unpaired=bool(inputs.get("include_unpaired", True)),
            keep_splits=bool(inputs.get("keep_splits", True)),
        )
        ds, report = importer.import_dataset(
            _store(inputs), inputs["source_dir"], options,
            progress=_progress(reporter), cancel=cancel.is_cancelled, log=reporter.log,
        )
        split_text = ds.summary().split_text()
        return ToolResult(
            status="success",
            message=f"已导入「{ds.name}」：原图 {report['images']} 张，掩码 {report['masks']} 个"
                    f"{f'（{split_text}）' if split_text else ''}{_failed_summary(report)}",
            metrics={"dataset_id": ds.id, "report": report},
        )


class ValidateDatasetTool:
    spec = _spec("datasets.validate", "数据校验", "检查缺失文件、重复配对、尺寸不一致和未知标签", ["dataset_id"])

    def validate(self, inputs: dict, context: DataContext) -> None:
        _store(inputs).load(str(inputs.get("dataset_id", "")))

    def run(self, inputs: dict, context: DataContext, reporter, cancel: CancelToken) -> ToolResult:
        ds = _store(inputs).load(inputs["dataset_id"])
        issues = validation.validate_dataset(
            ds, check_pixels=bool(inputs.get("check_pixels", True)),
            progress=_progress(reporter), cancel=cancel.is_cancelled,
        )
        report = {"dataset_id": ds.id, "issues": [i.to_dict() for i in issues]}
        path = ds.write_report("validate", report)
        n_err = sum(1 for i in issues if i.severity == "error")
        return ToolResult(
            status="success",
            message=f"校验完成：错误 {n_err} 项，警告 {len(issues) - n_err} 项",
            metrics={"dataset_id": ds.id, "issues": report["issues"], "report_path": str(path)},
        )


class ConvertDatasetTool:
    spec = _spec("datasets.convert", "格式转换", "转换为新数据集并保留源数据与转换记录", ["dataset_id", "kind", "name"])

    def validate(self, inputs: dict, context: DataContext) -> None:
        _store(inputs).load(str(inputs.get("dataset_id", "")))
        if inputs.get("kind") not in convert.CONVERSIONS:
            raise ValueError(f"未知转换类型: {inputs.get('kind')}")

    def run(self, inputs: dict, context: DataContext, reporter, cancel: CancelToken) -> ToolResult:
        store = _store(inputs)
        kind = inputs["kind"]
        params = inputs.get("params") or {}
        if kind == "instance_to_semantic":
            ds, report = convert.instance_to_semantic(
                store, inputs["dataset_id"], inputs["name"], progress=_progress(reporter), cancel=cancel.is_cancelled
            )
        else:
            ds, report = convert.semantic_to_instance(
                store, inputs["dataset_id"], inputs["name"],
                min_area=int(params.get("min_area", 20)), connectivity=int(params.get("connectivity", 8)),
                progress=_progress(reporter), cancel=cancel.is_cancelled,
            )
        return ToolResult(
            status="success",
            message=f"已生成新数据集「{ds.name}」：样本 {report['samples']}{_failed_summary(report)}",
            metrics={"dataset_id": ds.id, "report": report},
        )


class ExportDatasetTool:
    spec = _spec("datasets.export", "导出数据集", "导出为同目录 *_mask 或 images/masks 外部格式", ["dataset_id", "out_dir"])

    def validate(self, inputs: dict, context: DataContext) -> None:
        _store(inputs).load(str(inputs.get("dataset_id", "")))
        if not str(inputs.get("out_dir", "")).strip():
            raise ValueError("缺少导出目录")

    def run(self, inputs: dict, context: DataContext, reporter, cancel: CancelToken) -> ToolResult:
        ds = _store(inputs).load(inputs["dataset_id"])
        report = convert.export_dataset(
            ds, inputs["out_dir"], layout=inputs.get("layout", "split_dir"),
            encoding=inputs.get("encoding", "index"), mask_suffix=inputs.get("mask_suffix", "_mask"),
            only_annotated=bool(inputs.get("only_annotated", False)),
            keep_splits=bool(inputs.get("keep_splits", True)),
            progress=_progress(reporter), cancel=cancel.is_cancelled,
        )
        return ToolResult(
            status="success",
            message=f"已导出到 {report['out_dir']}：原图 {report['images']}，掩码 {report['masks']}{_failed_summary(report)}",
            metrics={"dataset_id": ds.id, "report": report},
        )


class MergeClassesTool:
    spec = _spec("datasets.merge_classes", "合并类别", "把若干类别合并到目标类别并重写掩码", ["dataset_id", "sources", "target"])

    def validate(self, inputs: dict, context: DataContext) -> None:
        _store(inputs).load(str(inputs.get("dataset_id", "")))

    def run(self, inputs: dict, context: DataContext, reporter, cancel: CancelToken) -> ToolResult:
        ds = _store(inputs).load(inputs["dataset_id"])
        res = ds.merge_classes(
            [int(x) for x in inputs["sources"]], int(inputs["target"]),
            progress=_progress(reporter), cancel=cancel.is_cancelled,
        )
        return ToolResult(
            status="success",
            message=f"合并完成：修改样本 {res['samples']} 个，像素 {res['pixels']}",
            metrics={"dataset_id": ds.id, **res},
        )


class PairMasksTool:
    spec = _spec("datasets.pair_masks", "批量配对", "按命名规则把外部掩码批量配对到已有样本", ["dataset_id", "mask_dir"])

    def validate(self, inputs: dict, context: DataContext) -> None:
        _store(inputs).load(str(inputs.get("dataset_id", "")))
        if not Path(str(inputs.get("mask_dir", ""))).is_dir():
            raise ValueError(f"掩码目录不存在: {inputs.get('mask_dir')}")

    def run(self, inputs: dict, context: DataContext, reporter, cancel: CancelToken) -> ToolResult:
        ds = _store(inputs).load(inputs["dataset_id"])
        mapping = inputs.get("mapping")
        report = importer.pair_masks_from_dir(
            ds, inputs["mask_dir"], mask_suffix=str(inputs.get("mask_suffix", "_mask")),
            recursive=bool(inputs.get("recursive", True)),
            mapping={str(k): int(v) for k, v in mapping.items()} if mapping else None,
            overwrite=bool(inputs.get("overwrite", False)),
            progress=_progress(reporter), cancel=cancel.is_cancelled,
        )
        return ToolResult(
            status="success",
            message=f"批量配对完成：写入 {report['written']} 个{_failed_summary(report)}",
            metrics={"dataset_id": ds.id, "report": report},
        )


class AddImagesTool:
    spec = _spec("datasets.add_images", "添加图像", "复制图像到数据集并新增未标注样本", ["dataset_id", "files"])

    def validate(self, inputs: dict, context: DataContext) -> None:
        _store(inputs).load(str(inputs.get("dataset_id", "")))
        missing = [f for f in inputs.get("files", []) if not Path(f).is_file()]
        if missing:
            raise ValueError(f"文件不存在: {missing[:3]}")

    def run(self, inputs: dict, context: DataContext, reporter, cancel: CancelToken) -> ToolResult:
        ds = _store(inputs).load(inputs["dataset_id"])
        added = importer.add_images(ds, [Path(f) for f in inputs["files"]], _progress(reporter), cancel.is_cancelled)
        return ToolResult(status="success", message=f"已添加 {len(added)} 张图像", metrics={"dataset_id": ds.id})


class ImportPredictionsTool:
    spec = _spec(
        "datasets.import_predictions", "分割结果入库",
        "把二值分割结果写回当前数据集（不覆盖人工标注）或另存为新数据集",
        ["mask_dir", "target"],
    )

    def validate(self, inputs: dict, context: DataContext) -> None:
        store = _store(inputs)
        if not Path(str(inputs.get("mask_dir", ""))).is_dir():
            raise ValueError(f"分割结果目录不存在: {inputs.get('mask_dir')}")
        target = inputs.get("target")
        if target == "current":
            store.load(str(inputs.get("dataset_id", ""))).binary_foreground_class()
        elif target == "new":
            if not str(inputs.get("name", "")).strip():
                raise ValueError("新数据集名称不能为空")
        else:
            raise ValueError(f"未知目标: {target}")

    def run(self, inputs: dict, context: DataContext, reporter, cancel: CancelToken) -> ToolResult:
        store = _store(inputs)
        suffix = str(inputs.get("mask_suffix", "_mask"))
        info = str(inputs.get("info", ""))
        if inputs["target"] == "current":
            ds = store.load(inputs["dataset_id"])
            report = importer.write_back_predictions(
                ds, inputs["mask_dir"], suffix, info=info,
                progress=_progress(reporter), cancel=cancel.is_cancelled,
            )
            message = (f"已写回「{ds.name}」：预测 {report['written']} 张，"
                       f"跳过人工标注 {report['skipped_annotated']} 张{_failed_summary(report)}")
        else:
            ds, report = importer.predictions_to_dataset(
                store, [Path(p) for p in inputs.get("images", [])], inputs["mask_dir"], str(inputs["name"]),
                suffix, provenance={"params": {"info": info}, **(inputs.get("provenance") or {})},
                progress=_progress(reporter), cancel=cancel.is_cancelled, splits=inputs.get("splits"),
            )
            message = f"已生成新数据集「{ds.name}」：预测 {report['written']} 张{_failed_summary(report)}"
        return ToolResult(status="success", message=message, metrics={"dataset_id": ds.id, "report": report})


class SplitDatasetTool:
    spec = _spec(
        "datasets.split", "数据划分",
        "按比例随机划分 train / val / test，或把指定样本移到某个划分（文件移动到对应子目录）",
        ["dataset_id", "mode"],
    )

    def validate(self, inputs: dict, context: DataContext) -> None:
        _store(inputs).load(str(inputs.get("dataset_id", "")))
        mode = inputs.get("mode")
        if mode == "random":
            ratios = inputs.get("ratios") or {}
            if sum(float(v) for v in ratios.values()) <= 0:
                raise ValueError("划分比例之和必须大于 0")
        elif mode == "assign":
            normalize_split(inputs.get("split"))
        else:
            raise ValueError(f"未知划分方式: {mode}")

    def run(self, inputs: dict, context: DataContext, reporter, cancel: CancelToken) -> ToolResult:
        ds = _store(inputs).load(inputs["dataset_id"])
        ids = inputs.get("sample_ids")
        samples = None if ids is None else [s for s in (ds.sample(i) for i in ids) if s is not None]
        if inputs["mode"] == "random":
            ds.random_split(
                {str(k): float(v) for k, v in inputs["ratios"].items()}, seed=int(inputs.get("seed", 42)),
                samples=samples, progress=_progress(reporter), cancel=cancel.is_cancelled,
            )
            message = f"随机划分完成：{ds.summary().split_text()}"
        else:
            split = normalize_split(inputs.get("split"))
            moved = ds.set_splits(ds.samples if samples is None else samples, split,
                                  progress=_progress(reporter), cancel=cancel.is_cancelled)
            message = f"已把 {moved} 个样本设为{split_label(split)}" + (
                f"；当前 {ds.summary().split_text()}" if ds.has_splits else "")
        return ToolResult(status="success", message=message,
                          metrics={"dataset_id": ds.id, "splits": ds.split_counts()})


ALL_TOOLS = (
    ImportDatasetTool,
    ValidateDatasetTool,
    ConvertDatasetTool,
    ExportDatasetTool,
    MergeClassesTool,
    PairMasksTool,
    AddImagesTool,
    ImportPredictionsTool,
    SplitDatasetTool,
)
