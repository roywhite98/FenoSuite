# -*- coding: utf-8 -*-
from pathlib import Path

from plant_phenotype.annotation.facade import get_facade
from plant_phenotype.app.data_context import DataContext
from plant_phenotype.app.task_models import Artifact, CancelToken, ToolResult, ToolSpec

_facade = get_facade()
_TILE = _facade.default_tile_params()
DEFAULT_TILE_SIZE = _TILE["tile_size"]
DEFAULT_MIN_FOREGROUND_RATIO = _TILE["min_foreground_ratio"]


class TileDirectoryTool:
    spec = ToolSpec(
        id="annotation.tile_directory",
        name="目录切块",
        category="annotation",
        description="对目录内原图执行切块与 meta 写入（run_preprocessing_on_dir）",
        input_schema={
            "type": "object",
            "required": ["annotated_dir", "tiles_base_dir"],
            "properties": {
                "annotated_dir": {"type": "string"},
                "tiles_base_dir": {"type": "string"},
                "tile_size": {"type": "integer", "default": DEFAULT_TILE_SIZE},
                "min_foreground_ratio": {"type": "number", "default": DEFAULT_MIN_FOREGROUND_RATIO},
            },
        },
        output_schema={"type": "object"},
        long_running=True,
        result_viewer="directory",
    )

    def validate(self, inputs: dict, context: DataContext) -> None:
        for key in ("annotated_dir", "tiles_base_dir"):
            if key not in inputs or not str(inputs[key]).strip():
                raise ValueError(f"缺少或为空: {key}")
        ad = Path(str(inputs["annotated_dir"]))
        if not ad.is_dir():
            raise ValueError(f"annotated_dir 不是目录: {ad}")

    def run(
        self,
        inputs: dict,
        context: DataContext,
        reporter,
        cancel: CancelToken,
    ) -> ToolResult:
        annotated_dir = Path(str(inputs["annotated_dir"]))
        tiles_base_dir = Path(str(inputs["tiles_base_dir"]))
        tile_size = int(inputs.get("tile_size", DEFAULT_TILE_SIZE))
        min_ratio = float(inputs.get("min_foreground_ratio", DEFAULT_MIN_FOREGROUND_RATIO))
        tiles_base_dir.mkdir(parents=True, exist_ok=True)
        cancel.raise_if_cancelled()
        reporter.log(f"开始切块: 源={annotated_dir} 输出根={tiles_base_dir} tile={tile_size}")
        n = _facade.tile_directory(
            annotated_dir, tiles_base_dir, tile_size=tile_size, min_foreground_ratio=min_ratio
        )
        reporter.log(f"完成: 处理 {n} 张图")
        return ToolResult(
            status="success",
            message=f"已对 {n} 张图执行切块",
            metrics={"image_count": n},
            artifacts=[
                Artifact(id="tiles_root", kind="directory", path=tiles_base_dir.resolve()),
            ],
        )


class ReassembleTilesTool:
    spec = ToolSpec(
        id="annotation.reassemble_tiles",
        name="还原切块",
        category="annotation",
        description="从含 meta.json 的切块目录拼接还原整图与 mask",
        input_schema={
            "type": "object",
            "required": ["tile_dir", "output_image_path"],
            "properties": {
                "tile_dir": {"type": "string"},
                "output_image_path": {"type": "string"},
                "output_mask_path": {"type": "string"},
                "use_predicted_mask": {"type": "boolean", "default": False},
            },
        },
        output_schema={"type": "object"},
        long_running=True,
        result_viewer="directory",
    )

    def validate(self, inputs: dict, context: DataContext) -> None:
        tile_dir = Path(str(inputs.get("tile_dir", "")).strip())
        if not tile_dir.is_dir():
            raise ValueError(f"tile_dir 不是目录: {tile_dir}")
        if not (tile_dir / "meta.json").is_file():
            raise ValueError("tile_dir 下无 meta.json")
        if not str(inputs.get("output_image_path", "")).strip():
            raise ValueError("缺少 output_image_path")

    def run(
        self,
        inputs: dict,
        context: DataContext,
        reporter,
        cancel: CancelToken,
    ) -> ToolResult:
        tile_dir = Path(str(inputs["tile_dir"]))
        out_img = Path(str(inputs["output_image_path"]))
        out_mask = inputs.get("output_mask_path")
        out_mask_path = Path(str(out_mask)) if out_mask else None
        cancel.raise_if_cancelled()
        reporter.log(f"还原切块: {tile_dir} → {out_img}")
        _facade.reassemble_tiles(
            tile_dir,
            output_image_path=out_img,
            output_mask_path=out_mask_path,
            use_predicted_mask=bool(inputs.get("use_predicted_mask", False)),
        )
        arts = [Artifact(id="image", kind="image", path=out_img.resolve())]
        if out_mask_path is not None:
            arts.append(Artifact(id="mask", kind="mask", path=out_mask_path.resolve()))
        return ToolResult(status="success", message="还原完成", artifacts=arts)
