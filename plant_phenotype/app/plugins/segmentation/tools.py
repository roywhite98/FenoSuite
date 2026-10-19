# -*- coding: utf-8 -*-
from pathlib import Path

from plant_phenotype.app.data_context import DataContext
from plant_phenotype.app.task_models import Artifact, CancelToken, ToolResult, ToolSpec
from plant_phenotype.segmentation.facade import get_facade


class BatchPredictTool:
    spec = ToolSpec(
        id="segmentation.batch_predict",
        name="批量分割",
        category="segmentation",
        description="对目录内图像批量预测，结果写入 output_dir/masks/",
        input_schema={
            "type": "object",
            "required": ["config_path", "checkpoint_path", "image_dir", "output_dir"],
            "properties": {
                "config_path": {"type": "string"},
                "checkpoint_path": {"type": "string"},
                "image_dir": {"type": "string"},
                "image_files": {"type": "array", "items": {"type": "string"},
                                "description": "可选：只预测这些文件（如数据集各划分下的原图）"},
                "output_dir": {"type": "string"},
                "no_cuda": {"type": "boolean", "default": False},
            },
        },
        output_schema={"type": "object"},
        long_running=True,
        result_viewer="directory",
    )

    def validate(self, inputs: dict, context: DataContext) -> None:
        for k in ("config_path", "checkpoint_path", "image_dir", "output_dir"):
            if k not in inputs or not str(inputs[k]).strip():
                raise ValueError(f"缺少或为空: {k}")
        if not Path(str(inputs["checkpoint_path"])).is_file():
            raise ValueError("checkpoint_path 不是有效文件")
        files = inputs.get("image_files")
        if files is not None:
            missing = [f for f in files if not Path(str(f)).is_file()]
            if missing:
                raise ValueError(f"图像文件不存在: {missing[:3]}")
        elif not Path(str(inputs["image_dir"])).is_dir():
            raise ValueError("image_dir 不是有效目录")
        cfg_p = Path(str(inputs["config_path"]))
        if not cfg_p.is_file():
            raise ValueError(f"配置文件不存在: {cfg_p}")

    def run(
        self,
        inputs: dict,
        context: DataContext,
        reporter,
        cancel: CancelToken,
    ) -> ToolResult:
        cfg = str(inputs["config_path"])
        ckpt = str(inputs["checkpoint_path"])
        img_dir = str(inputs["image_dir"])
        out_dir = str(inputs["output_dir"])
        no_cuda = bool(inputs.get("no_cuda", False))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        cancel.raise_if_cancelled()

        def line_cb(line: str) -> None:
            reporter.log(line)
            cancel.raise_if_cancelled()

        try:
            success, failed = get_facade().batch_predict(
                config_path=cfg,
                checkpoint_path=ckpt,
                image_dir=img_dir,
                output_dir=out_dir,
                no_cuda=no_cuda,
                line_callback=line_cb,
                image_files=[str(f) for f in inputs["image_files"]] if inputs.get("image_files") is not None else None,
            )
        except RuntimeError as e:
            if cancel.is_cancelled() or "cancelled" in str(e).lower():
                return ToolResult(status="cancelled", message=str(e))
            raise
        masks_dir = Path(out_dir) / "masks"
        context.set_last_mask_dir(masks_dir)
        return ToolResult(
            status="success",
            message=f"成功 {success} 张，失败 {len(failed)} 张",
            metrics={"success": success, "failed_count": len(failed), "failed": failed[:50]},
            artifacts=[Artifact(id="masks_dir", kind="directory", path=masks_dir.resolve() if masks_dir.is_dir() else Path(out_dir).resolve())],
        )


class PreviewTool:
    spec = ToolSpec(
        id="segmentation.preview",
        name="单张预览",
        category="segmentation",
        description="对单张图像运行分割预览",
        input_schema={
            "type": "object",
            "required": ["config_path", "checkpoint_path", "image_path"],
            "properties": {
                "config_path": {"type": "string"},
                "checkpoint_path": {"type": "string"},
                "image_path": {"type": "string"},
                "no_cuda": {"type": "boolean", "default": False},
            },
        },
        output_schema={"type": "object"},
        long_running=True,
    )

    def validate(self, inputs: dict, context: DataContext) -> None:
        for k in ("config_path", "checkpoint_path", "image_path"):
            if k not in inputs or not str(inputs[k]).strip():
                raise ValueError(f"缺少或为空: {k}")
        if not Path(str(inputs["checkpoint_path"])).is_file():
            raise ValueError("checkpoint_path 不是有效文件")
        if not Path(str(inputs["image_path"])).is_file():
            raise ValueError("image_path 不是有效文件")

    def run(
        self,
        inputs: dict,
        context: DataContext,
        reporter,
        cancel: CancelToken,
    ) -> ToolResult:
        cancel.raise_if_cancelled()
        reporter.log("运行单张预览…")
        img_rgb, mask = get_facade().preview(
            str(inputs["config_path"]),
            str(inputs["checkpoint_path"]),
            str(inputs["image_path"]),
            bool(inputs.get("no_cuda", False)),
        )
        return ToolResult(
            status="success",
            message="预览完成",
            metrics={"height": int(img_rgb.shape[0]), "width": int(img_rgb.shape[1])},
            artifacts=[
                Artifact(id="preview_rgb", kind="image", data=img_rgb),
                Artifact(id="preview_mask", kind="mask", data=mask),
            ],
        )
