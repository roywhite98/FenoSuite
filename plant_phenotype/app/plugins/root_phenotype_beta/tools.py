"""Background wrappers; results are committed only by the UI controller."""
from pathlib import Path

from plant_phenotype.app.task_models import Artifact, ToolResult, ToolSpec
from plant_phenotype.root_phenotype.facade import get_facade


def _result(inputs, document, message):
    return ToolResult(
        status="success", message=message,
        artifacts=[Artifact(id="vector_document", kind="object", data=document)],
        metrics={"request_id": inputs.get("request_id"),
                 "document_id": inputs.get("document_id"),
                 "revision": inputs.get("revision"),
                 "source_fingerprint": inputs.get("source_fingerprint")},
    )


class RootVectorAnalyzeTool:
    spec = ToolSpec(
        id="root.vector.analyze", name="根系矢量解析", category="root_phenotype_beta",
        description="选中根内任意位置，修复断口并识别主根、侧根",
        input_schema={"type": "object", "required": ["mask_path"], "properties": {
            "mask_path": {"type": "string"}, "base_marks": {"type": "array"},
            "selection_points": {"type": "array"},
            "original_image_path": {"type": "string"}, "config": {"type": "object"}}},
        output_schema={"type": "object"}, long_running=True,
    )

    def validate(self, inputs, context):
        if not Path(str(inputs.get("mask_path", ""))).is_file():
            raise ValueError("请选择有效的掩码文件")
        if not isinstance(inputs.get("base_marks", []), (list, tuple)):
            raise ValueError("基部标记必须是列表")
        if inputs.get("config") is not None and not isinstance(inputs["config"], dict):
            raise ValueError("分析参数必须是字典")
        if inputs.get("selection_points") is not None and not isinstance(inputs['selection_points'], (list, tuple)):
            raise ValueError('根系选点必须是列表')

    def run(self, inputs, context, reporter, cancel):
        cancel.raise_if_cancelled()
        reporter.log("正在提取骨架、方向与矢量根段…")
        result = get_facade().analyze_vectors(
            inputs["mask_path"], base_marks=inputs.get("base_marks", []),
            original_image_path=inputs.get("original_image_path"),
            config=inputs.get("config"), document_id=inputs.get("document_id"),
            selection_points=inputs.get("selection_points"),
            progress=reporter.progress, check_cancel=cancel.raise_if_cancelled,
        )
        cancel.raise_if_cancelled()
        return _result(inputs, result.document, "矢量解析完成")


class RootVectorResolveTool:
    spec = ToolSpec(
        id="root.vector.resolve", name="更新根系接续", category="root_phenotype_beta",
        description="按用户选择重组交汇处的矢量根系",
        input_schema={"type": "object", "required": ["document", "decisions"]},
        output_schema={"type": "object"}, long_running=True,
    )

    def validate(self, inputs, context):
        if inputs.get("document") is None or not isinstance(inputs.get("decisions"), dict):
            raise ValueError("缺少矢量文档或交汇选择")

    def run(self, inputs, context, reporter, cancel):
        cancel.raise_if_cancelled()
        reporter.log("正在更新交汇接续和父子关系…")
        result = get_facade().resolve_vectors(
            inputs["document"], inputs["decisions"], check_cancel=cancel.raise_if_cancelled,
        )
        cancel.raise_if_cancelled()
        return _result(inputs, result.document, "接续已更新")
