# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any

from plant_phenotype.app.data_context import DataContext
from plant_phenotype.app.task_models import Artifact, CancelToken, ToolResult, ToolSpec
from plant_phenotype.root_phenotype.facade import get_facade


class RootAnalyzeSelectedRootsTool:
    spec = ToolSpec(
        id="root.analyze_selected_roots",
        name="根系分析（已选根）",
        category="root_phenotype",
        description="对已预处理掩码按种子点执行 run_one（与 Tab 内选根分析一致）",
        input_schema={
            "type": "object",
            "required": ["mask_path", "selected_seeds"],
            "properties": {
                "mask_path": {"type": "string"},
                "selected_seeds": {"type": "array"},
                "original_image_path": {"type": "string"},
                "config_overrides": {"type": "object"},
            },
        },
        output_schema={"type": "object"},
        long_running=True,
    )

    def validate(self, inputs: dict, context: DataContext) -> None:
        if "mask_path" not in inputs or not str(inputs["mask_path"]).strip():
            raise ValueError("缺少 mask_path")
        if not Path(str(inputs["mask_path"])).is_file():
            raise ValueError("mask_path 不是有效文件")
        seeds = inputs.get("selected_seeds")
        if not isinstance(seeds, (list, tuple)) or len(seeds) < 1:
            raise ValueError("selected_seeds 必须为非空列表，每项为 [row, col]")

    def run(
        self,
        inputs: dict,
        context: DataContext,
        reporter,
        cancel: CancelToken,
    ) -> ToolResult:
        mask_path = str(inputs["mask_path"])
        seeds_raw = inputs["selected_seeds"]
        seeds = [(int(p[0]), int(p[1])) for p in seeds_raw]
        orig = inputs.get("original_image_path")
        orig = str(orig) if orig else None
        overrides: dict[str, Any] | None = inputs.get("config_overrides")
        if overrides is not None and not isinstance(overrides, dict):
            raise ValueError("config_overrides 须为 dict")

        cancel.raise_if_cancelled()
        reporter.log("执行根系分析…")
        res = get_facade().analyze(
            mask_path,
            seeds=seeds or None,
            mode="interactive",
            original_image_path=orig,
            config_overrides=overrides,
        )
        context.set_last_mask_path(mask_path)
        context.set_pipeline_result(res)
        cancel.raise_if_cancelled()
        if res.error:
            return ToolResult(status="failed", error=str(res.error), message=str(res.error))
        n_roots = len(getattr(res, "roots", None) or [])
        sdf = getattr(res, "summary_df", None)
        metrics: dict[str, Any] = {"roots_components": n_roots}
        if sdf is not None and not getattr(sdf, "empty", True) and "main_root_length" in sdf.columns:
            try:
                metrics["main_root_length"] = float(sdf["main_root_length"].iloc[0])
            except (TypeError, ValueError, IndexError):
                pass
        reporter.log("分析完成")
        return ToolResult(status="success", message="分析完成", metrics=metrics)


class RootAnalyzePipelineTool:
    """薄封装 `run_one`，与计划中的 `root.analyze` 工具 id 对齐。"""

    spec = ToolSpec(
        id="root.analyze",
        name="根系分析（run_one）",
        category="root_phenotype",
        description="调用 run_one：支持 mask_path、可选种子、batch_mode、config_overrides 等",
        input_schema={
            "type": "object",
            "required": ["mask_path"],
            "properties": {
                "mask_path": {"type": "string"},
                "original_image_path": {"type": "string"},
                "selected_seeds": {"type": "array"},
                "batch_mode": {"type": "boolean", "default": False},
                "preprocessed_mask": {"type": "object"},
                "config_overrides": {"type": "object"},
            },
        },
        output_schema={"type": "object"},
        long_running=True,
    )

    def validate(self, inputs: dict, context: DataContext) -> None:
        if "mask_path" not in inputs or not str(inputs["mask_path"]).strip():
            raise ValueError("缺少 mask_path")
        if not Path(str(inputs["mask_path"])).is_file():
            raise ValueError("mask_path 不是有效文件")
        co = inputs.get("config_overrides")
        if co is not None and not isinstance(co, dict):
            raise ValueError("config_overrides 须为 dict")

    def run(
        self,
        inputs: dict,
        context: DataContext,
        reporter,
        cancel: CancelToken,
    ) -> ToolResult:
        mask_path = str(inputs["mask_path"])
        orig = inputs.get("original_image_path")
        orig = str(orig) if orig else None
        seeds_raw = inputs.get("selected_seeds")
        seeds = None
        if seeds_raw is not None:
            if not isinstance(seeds_raw, (list, tuple)):
                raise ValueError("selected_seeds 须为列表")
            seeds = [(int(p[0]), int(p[1])) for p in seeds_raw]
        batch_mode = bool(inputs.get("batch_mode", False))
        overrides: dict[str, Any] | None = inputs.get("config_overrides")

        cancel.raise_if_cancelled()
        reporter.log("执行 run_one…")
        res = get_facade().analyze(
            mask_path,
            seeds=seeds,
            mode="batch" if batch_mode else "interactive",
            original_image_path=orig,
            preprocessed_mask=inputs.get("preprocessed_mask"),
            config_overrides=overrides,
        )
        context.set_last_mask_path(mask_path)
        context.set_pipeline_result(res)
        cancel.raise_if_cancelled()
        if res.error:
            return ToolResult(
                status="failed",
                error=str(res.error),
                message=str(res.error),
                artifacts=[Artifact(id="pipeline_result", kind="object", data=res)],
            )
        reporter.log("run_one 完成")
        return ToolResult(
            status="success",
            message="run_one 完成",
            artifacts=[Artifact(id="pipeline_result", kind="object", data=res)],
        )
