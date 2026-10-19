# -*- coding: utf-8 -*-
"""远程训练服务 HTTP 客户端：打包数据目录、提交任务、轮询状态、拉取产物。"""
from __future__ import annotations

import io
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

DEFAULT_REMOTE_URL = "http://localhost:8765"


def pack_directory_to_zip(data_root: str | Path) -> Path:
    """将数据目录打包为临时 zip 文件并返回路径（调用方负责删除）。"""
    root = Path(data_root)
    fd, data_zip_path = tempfile.mkstemp(suffix=".zip", prefix="training_data_")
    os.close(fd)
    with zipfile.ZipFile(data_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in root.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(root))
    return Path(data_zip_path)


def submit_remote_job(
    base_url: str,
    config_yaml: str,
    data_root: Optional[str | Path] = None,
    *,
    line_callback: Optional[Callable[[str], None]] = None,
    timeout: float = 180.0,
) -> str:
    """打包数据（可选）并提交到远程 `/job`，返回 `job_id`。失败时抛出异常。"""
    url = base_url.rstrip("/")
    data_zip_path: Optional[Path] = None
    try:
        if data_root and Path(data_root).is_dir():
            if line_callback:
                line_callback("正在打包数据目录…")
            data_zip_path = pack_directory_to_zip(data_root)
            if line_callback:
                line_callback("打包完成，提交任务…")

        boundary = "----WebKitFormBoundary" + "x" * 16
        body = io.BytesIO()
        body.write(
            f'--{boundary}\r\nContent-Disposition: form-data; name="config"; filename="config.yaml"\r\n'
            f"Content-Type: text/yaml; charset=utf-8\r\n\r\n".encode("utf-8")
        )
        body.write(config_yaml.encode("utf-8"))
        if data_zip_path and data_zip_path.is_file():
            body.write(f"\r\n--{boundary}\r\n".encode("utf-8"))
            zip_data = data_zip_path.read_bytes()
            body.write(
                'Content-Disposition: form-data; name="data_zip"; filename="data.zip"\r\n'
                "Content-Type: application/zip\r\n\r\n".encode("utf-8")
            )
            body.write(zip_data)
        body.write(f"\r\n--{boundary}--\r\n".encode("utf-8"))
        body_bytes = body.getvalue()

        req = urllib.request.Request(
            f"{url}/job",
            data=body_bytes,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body_bytes)),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = str(e)
            raise RuntimeError(f"HTTP {e.code}: {err_body}") from e

        if "error" in out and out["error"]:
            raise RuntimeError(str(out["error"]))
        job_id = out.get("job_id")
        if not job_id:
            raise RuntimeError("服务器未返回 job_id")
        return str(job_id)
    finally:
        if data_zip_path is not None:
            try:
                data_zip_path.unlink(missing_ok=True)
            except Exception:
                pass


def poll_remote_job(
    base_url: str,
    job_id: str,
    *,
    timeout: float = 10.0,
) -> Tuple[str, List[str], Optional[str]]:
    """请求 `GET /job/<job_id>`，返回 `(status, log_tail, error)`。"""
    url = f"{base_url.rstrip('/')}/job/{job_id}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read().decode("utf-8"))
            return (
                str(out.get("status", "unknown")),
                [str(x) for x in out.get("log_tail", [])],
                out.get("error"),
            )
    except Exception as e:
        return ("error", [], str(e))


def run_remote_training(
    base_url: str,
    config_path: str | Path,
    data_root: str | Path,
    *,
    poll_interval: float = 2.0,
    line_callback: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> str:
    """
    提交远程训练任务并轮询直至结束，实时通过 `line_callback` 推送增量日志。
    返回 `job_id`。若任务失败则抛出 `RuntimeError`。
    """
    config_yaml = Path(config_path).read_text(encoding="utf-8")
    job_id = submit_remote_job(
        base_url,
        config_yaml,
        data_root,
        line_callback=line_callback,
    )
    if line_callback:
        line_callback(f"任务已提交: {job_id}")

    last_log_tail: List[str] = []
    while True:
        if should_cancel and should_cancel():
            raise RuntimeError("cancelled")
        status, log_tail, err = poll_remote_job(base_url, job_id)
        if log_tail and line_callback:
            # 计算新增行（兼容 log_tail 未截断与滑动窗口截断两种情况）
            if len(log_tail) >= len(last_log_tail) and log_tail[: len(last_log_tail)] == last_log_tail:
                new_lines = log_tail[len(last_log_tail) :]
            else:
                # 发生窗口滑动时，找到最大重叠后缀
                overlap = 0
                max_k = min(len(last_log_tail), len(log_tail))
                for k in range(max_k, 0, -1):
                    if last_log_tail[-k:] == log_tail[:k]:
                        overlap = k
                        break
                new_lines = log_tail[overlap:]
            for line in new_lines:
                line_callback(line)
            last_log_tail = list(log_tail)

        if status == "success":
            return job_id
        if status == "failed":
            raise RuntimeError(err or "远程训练任务失败")

        # 睡眠期间分段检查取消
        slept = 0.0
        step = min(0.2, poll_interval)
        while slept < poll_interval:
            if should_cancel and should_cancel():
                raise RuntimeError("cancelled")
            time.sleep(step)
            slept += step
