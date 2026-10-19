# -*- coding: utf-8 -*-
from pathlib import Path

from plant_phenotype.app.data_context import DataContext
from plant_phenotype.app.task_models import CancelToken, ToolResult, ToolSpec
from plant_phenotype.datasets.training_source import binary_training_pairs, is_dataset_dir
from plant_phenotype.training.facade import get_facade


class LocalTrainTool:
    spec = ToolSpec(
        id="training.local_train",
        name="本地训练",
        category="training",
        description="进程内执行模型训练",
        input_schema={
            "type": "object",
            "required": ["config_path"],
            "properties": {
                "config_path": {"type": "string"},
                "data_root": {"type": "string"},
                "save_dir": {"type": "string"},
                "use_cpu": {"type": "boolean", "default": False},
                "resume_from": {"type": "string"},
            },
        },
        output_schema={"type": "object"},
        long_running=True,
        result_viewer="log",
    )

    def validate(self, inputs: dict, context: DataContext) -> None:
        cfg = str(inputs.get("config_path", "")).strip()
        if not cfg:
            raise ValueError("缺少 config_path")
        if not Path(cfg).is_file():
            raise ValueError(f"配置文件不存在: {cfg}")
        data_root = inputs.get("data_root")
        if data_root and not Path(str(data_root)).is_dir():
            raise ValueError(f"数据目录不存在: {data_root}")
        if data_root and is_dataset_dir(data_root):
            binary_training_pairs(data_root)
        resume = inputs.get("resume_from")
        if resume and not Path(str(resume)).is_file():
            raise ValueError(f"继续训练权重不存在: {resume}")

    def run(
        self,
        inputs: dict,
        context: DataContext,
        reporter,
        cancel: CancelToken,
    ) -> ToolResult:
        cancel.raise_if_cancelled()

        def line_cb(line: str) -> None:
            reporter.log(line)
            cancel.raise_if_cancelled()

        get_facade().run_local(
            config_path=str(inputs["config_path"]),
            data_root=str(inputs["data_root"]) if inputs.get("data_root") else None,
            save_dir=str(inputs["save_dir"]) if inputs.get("save_dir") else None,
            use_cpu=bool(inputs.get("use_cpu", False)),
            resume_from=str(inputs["resume_from"]) if inputs.get("resume_from") else None,
            line_callback=line_cb,
        )
        import yaml
        config = yaml.safe_load(Path(str(inputs["config_path"])).read_text(encoding="utf-8")) or {}
        save_dir = inputs.get("save_dir") or config.get("train", {}).get("save_dir", "./workspace/runs/training")
        checkpoint = Path(save_dir) / "best_model.pth"
        if checkpoint.is_file():
            context.set_last_checkpoint_path(checkpoint)
        return ToolResult(status="success", message="训练完成")


class RemoteTrainTool:
    spec = ToolSpec(
        id="training.remote_train",
        name="远程训练",
        category="training",
        description="打包数据并提交至远程训练服务执行模型训练",
        input_schema={
            "type": "object",
            "required": ["base_url", "config_path", "data_root"],
            "properties": {
                "base_url": {"type": "string"},
                "config_path": {"type": "string"},
                "data_root": {"type": "string"},
                "poll_interval": {"type": "number", "default": 2.0},
            },
        },
        output_schema={"type": "object"},
        long_running=True,
        result_viewer="log",
    )

    def validate(self, inputs: dict, context: DataContext) -> None:
        base_url = str(inputs.get("base_url", "")).strip()
        if not base_url:
            raise ValueError("缺少远程服务地址 base_url")
        cfg = str(inputs.get("config_path", "")).strip()
        if not cfg:
            raise ValueError("缺少 config_path")
        if not Path(cfg).is_file():
            raise ValueError(f"配置文件不存在: {cfg}")
        data_root = str(inputs.get("data_root", "")).strip()
        if not data_root:
            raise ValueError("远程训练必须指定 data_root（将打包上传）")
        if not Path(data_root).is_dir():
            raise ValueError(f"数据目录不存在: {data_root}")
        if is_dataset_dir(data_root):
            binary_training_pairs(data_root)

    def run(
        self,
        inputs: dict,
        context: DataContext,
        reporter,
        cancel: CancelToken,
    ) -> ToolResult:
        cancel.raise_if_cancelled()

        def line_cb(line: str) -> None:
            reporter.log(line)
            cancel.raise_if_cancelled()

        job_id = get_facade().run_remote(
            base_url=str(inputs["base_url"]),
            config_path=str(inputs["config_path"]),
            data_root=str(inputs["data_root"]),
            poll_interval=float(inputs.get("poll_interval", 2.0)),
            line_callback=line_cb,
            should_cancel=cancel.is_cancelled,
        )
        return ToolResult(
            status="success",
            message=f"远程训练完成 (job_id={job_id})",
            metrics={"job_id": job_id},
        )


_SSH_PROPERTIES = {
    "ssh_address": {"type": "string", "description": "如 ssh -p 2222 root@host、root@host:2222 或仅 host（端口 22、用户 root）"},
    "password": {"type": "string"},
    "key_path": {"type": "string"},
    "python": {"type": "string", "default": "python"},
    "remote_dir": {"type": "string", "default": "~/fenosuite_train"},
    "auto_install": {"type": "boolean", "default": False},
}


def _ssh_target(inputs: dict):
    from plant_phenotype.training.ssh_client import DEFAULT_REMOTE_DIR, DEFAULT_REMOTE_PYTHON, SSHTarget, parse_ssh_address

    address = str(inputs.get("ssh_address", "")).strip()
    if not address:
        raise ValueError("缺少 SSH 连接地址")
    host, port, user, key = parse_ssh_address(address)
    key_path = str(inputs.get("key_path") or "").strip() or key
    if key_path and not Path(key_path).expanduser().is_file():
        raise ValueError(f"私钥文件不存在: {key_path}")
    return SSHTarget(
        host=host,
        port=port,
        username=user or "root",
        password=str(inputs.get("password") or "") or None,
        key_path=str(Path(key_path).expanduser()) if key_path else None,
        python=str(inputs.get("python") or "").strip() or DEFAULT_REMOTE_PYTHON,
        remote_dir=str(inputs.get("remote_dir") or "").strip() or DEFAULT_REMOTE_DIR,
    )


class SSHCheckTool:
    spec = ToolSpec(
        id="training.ssh_check",
        name="测试 SSH 训练环境",
        category="training",
        description="连接 SSH 服务器并检查远端 Python / PyTorch / GPU 与训练依赖",
        input_schema={"type": "object", "required": ["ssh_address"], "properties": dict(_SSH_PROPERTIES)},
        output_schema={"type": "object"},
        long_running=True,
        result_viewer="log",
    )

    def validate(self, inputs: dict, context: DataContext) -> None:
        _ssh_target(inputs)

    def run(self, inputs: dict, context: DataContext, reporter, cancel: CancelToken) -> ToolResult:
        env = get_facade().check_ssh(
            _ssh_target(inputs),
            auto_install=bool(inputs.get("auto_install", False)),
            line_callback=reporter.log,
        )
        return ToolResult(status="success", message="SSH 训练环境可用", metrics={"cuda": env.cuda, "gpus": env.gpus})


class SSHTrainTool:
    spec = ToolSpec(
        id="training.ssh_train",
        name="SSH 远程训练",
        category="training",
        description="经 SSH 上传最小训练代码与数据，在远端训练并下载产物",
        input_schema={
            "type": "object",
            "required": ["ssh_address", "config_path", "data_root"],
            "properties": {
                **_SSH_PROPERTIES,
                "config_path": {"type": "string"},
                "data_root": {"type": "string"},
                "save_dir": {"type": "string"},
                "resume_from": {"type": "string"},
            },
        },
        output_schema={"type": "object"},
        long_running=True,
        result_viewer="log",
    )

    def validate(self, inputs: dict, context: DataContext) -> None:
        _ssh_target(inputs)
        cfg = str(inputs.get("config_path", "")).strip()
        if not cfg or not Path(cfg).is_file():
            raise ValueError(f"配置文件不存在: {cfg}")
        data_root = str(inputs.get("data_root", "")).strip()
        if not data_root or not Path(data_root).is_dir():
            raise ValueError(f"数据目录不存在: {data_root}")
        if is_dataset_dir(data_root):
            binary_training_pairs(data_root)
        resume = inputs.get("resume_from")
        if resume and not Path(str(resume)).is_file():
            raise ValueError(f"继续训练权重不存在: {resume}")

    def run(self, inputs: dict, context: DataContext, reporter, cancel: CancelToken) -> ToolResult:
        cancel.raise_if_cancelled()

        def line_cb(line: str) -> None:
            reporter.log(line)
            cancel.raise_if_cancelled()

        import yaml
        config = yaml.safe_load(Path(str(inputs["config_path"])).read_text(encoding="utf-8")) or {}
        save_dir = inputs.get("save_dir") or config.get("train", {}).get("save_dir", "./workspace/runs/training")
        result = get_facade().run_ssh(
            _ssh_target(inputs),
            config_path=str(inputs["config_path"]),
            data_root=str(inputs["data_root"]),
            save_dir=str(save_dir),
            resume_from=str(inputs["resume_from"]) if inputs.get("resume_from") else None,
            auto_install=bool(inputs.get("auto_install", False)),
            line_callback=line_cb,
            should_cancel=cancel.is_cancelled,
        )
        checkpoint = Path(result["save_dir"]) / "best_model.pth"
        if checkpoint.is_file():
            context.set_last_checkpoint_path(checkpoint)
        return ToolResult(
            status="success",
            message=f"SSH 远程训练完成，产物已下载到 {result['save_dir']}",
            metrics=result,
        )
