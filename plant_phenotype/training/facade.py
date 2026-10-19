# -*- coding: utf-8 -*-
"""训练对外门面（本地训练 + 远程训练客户端）。"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from plant_phenotype.training.remote_client import (
    DEFAULT_REMOTE_URL,
    poll_remote_job,
    run_remote_training,
    submit_remote_job,
)


class TrainingFacade:
    def run_local(
        self,
        config_path: str,
        data_root: Optional[str] = None,
        save_dir: Optional[str] = None,
        *,
        use_cpu: bool = False,
        resume_from: Optional[str] = None,
        line_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        from plant_phenotype.training.train import run_training as _run_training

        _run_training(
            config_path=config_path,
            data_root=data_root,
            save_dir=save_dir,
            line_callback=line_callback,
            use_cpu=use_cpu,
            resume_from=resume_from,
        )

    def run_remote(
        self,
        base_url: str,
        config_path: str,
        data_root: str,
        *,
        poll_interval: float = 2.0,
        line_callback: Optional[Callable[[str], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> str:
        return run_remote_training(
            base_url=base_url,
            config_path=config_path,
            data_root=data_root,
            poll_interval=poll_interval,
            line_callback=line_callback,
            should_cancel=should_cancel,
        )

    def run_ssh(
        self,
        target,
        config_path: str,
        data_root: str,
        save_dir: str,
        *,
        resume_from: Optional[str] = None,
        auto_install: bool = False,
        line_callback: Optional[Callable[[str], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> dict:
        from plant_phenotype.training.ssh_client import run_ssh_training

        return run_ssh_training(
            target,
            config_path=config_path,
            data_root=data_root,
            save_dir=save_dir,
            resume_from=resume_from,
            auto_install=auto_install,
            line_callback=line_callback,
            should_cancel=should_cancel,
        )

    def check_ssh(
        self,
        target,
        *,
        auto_install: bool = False,
        line_callback: Optional[Callable[[str], None]] = None,
    ):
        """连接并检查（可选安装）远端训练环境，返回 `RemoteEnv`。"""
        from plant_phenotype.training.ssh_client import connect, prepare_remote_env

        client = connect(target)
        try:
            if line_callback:
                line_callback(f"已连接 {target.display}")
            return prepare_remote_env(client, target, auto_install=auto_install, line_callback=line_callback)
        finally:
            client.close()

    def default_config_path(self) -> Path:
        """包内默认 UNet 训练配置路径。"""
        return Path(__file__).resolve().parents[1] / "resources" / "configs" / "unet.yaml"

    def default_remote_url(self) -> str:
        """默认远程训练服务地址。"""
        return DEFAULT_REMOTE_URL


_default_facade = TrainingFacade()


def get_facade() -> TrainingFacade:
    return _default_facade


__all__ = [
    "DEFAULT_REMOTE_URL",
    "TrainingFacade",
    "get_facade",
    "poll_remote_job",
    "run_remote_training",
    "submit_remote_job",
]
