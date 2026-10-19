# -*- coding: utf-8 -*-
"""SSH 远程训练：上传最小训练代码与数据，在远端 Python 环境中后台训练，回传日志与产物。

远端只需一个装好 PyTorch 的 Python 环境，无需部署项目或启动服务。
"""
from __future__ import annotations

import codecs
import json
import posixpath
import re
import shlex
import stat
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from plant_phenotype.training.bundle import build_bundle_zip
from plant_phenotype.training.remote_client import pack_directory_to_zip

DEFAULT_REMOTE_DIR = "~/fenosuite_train"
DEFAULT_REMOTE_PYTHON = "python"

# import 名 -> pip 包名；torch 需与服务器 CUDA 匹配，不自动安装
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "cv2": "opencv-python-headless",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "albumentations": "albumentations",
    "segmentation_models_pytorch": "segmentation-models-pytorch",
    "timm": "timm",
    "tqdm": "tqdm",
}
OPTIONAL_PACKAGES = {"matplotlib": "matplotlib", "seaborn": "seaborn"}

_SSH_FLAGS_WITH_ARG = set("bcDEeFIiJLlmOopQRSWw")
_PROGRESS_RE = re.compile(r"\|\s*\d+/\d+\s*\[|\d+(\.\d+)?(it/s|s/it)")
_ENV_MARKER = "@@FENOSUITE_ENV@@"

LineCallback = Optional[Callable[[str], None]]


@dataclass
class SSHTarget:
    host: str
    port: int = 22
    username: str = "root"
    password: Optional[str] = None
    key_path: Optional[str] = None
    python: str = DEFAULT_REMOTE_PYTHON
    remote_dir: str = DEFAULT_REMOTE_DIR

    @property
    def display(self) -> str:
        return f"{self.username}@{self.host}:{self.port}"


@dataclass
class RemoteEnv:
    python: str = ""
    torch: Optional[str] = None
    cuda: bool = False
    gpus: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def missing_required(self) -> list[str]:
        return [m for m in self.missing if m == "torch" or m in REQUIRED_PACKAGES]

    @property
    def missing_optional(self) -> list[str]:
        return [m for m in self.missing if m in OPTIONAL_PACKAGES]

    def describe(self) -> str:
        gpu = ", ".join(self.gpus) if self.gpus else ("CUDA 可用" if self.cuda else "无可用 GPU（将使用 CPU）")
        return f"Python {self.python}，PyTorch {self.torch or '未安装'}，{gpu}"


def parse_ssh_address(text: str) -> tuple[str, int, Optional[str], Optional[str]]:
    """解析 `ssh -p 2222 root@host`、`root@host:2222`、`ssh://root@host:2222`、仅 `host` 等写法。

    返回 `(host, port, username, key_path)`，未指定的项为 None（端口默认 22）。
    """
    tokens = [t.strip("\"'") for t in shlex.split(text.strip(), posix=False)]
    if tokens and Path(tokens[0]).name.lower() in ("ssh", "ssh.exe"):
        tokens = tokens[1:]
    port: Optional[int] = None
    user: Optional[str] = None
    key: Optional[str] = None
    dest: Optional[str] = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-") and len(tok) >= 2 and tok[1] in _SSH_FLAGS_WITH_ARG:
            value = tok[2:]
            if not value:
                if i + 1 >= len(tokens):
                    raise ValueError(f"SSH 参数 {tok} 缺少取值")
                value = tokens[i + 1]
                i += 1
            if tok[1] == "p":
                port = _parse_port(value)
            elif tok[1] == "l":
                user = value
            elif tok[1] == "i":
                key = value
        elif not tok.startswith("-"):
            dest = tok
        i += 1
    if not dest:
        raise ValueError("未找到主机地址，示例：ssh -p 2222 root@192.168.1.100，或只填服务器 IP")
    if dest.lower().startswith("ssh://"):
        dest = dest[6:].rstrip("/")
    if "@" in dest:
        user, dest = dest.rsplit("@", 1)
    if dest.count(":") == 1:
        dest, port_text = dest.split(":")
        port = _parse_port(port_text)
    if not dest:
        raise ValueError("主机地址为空")
    return dest, port or 22, user or None, key


def _parse_port(text: str) -> int:
    try:
        port = int(text)
    except ValueError:
        raise ValueError(f"端口无效: {text}") from None
    if not 0 < port < 65536:
        raise ValueError(f"端口无效: {text}")
    return port


def connect(target: SSHTarget, timeout: float = 20.0):
    """建立 SSH 连接，返回 `paramiko.SSHClient`。"""
    try:
        import paramiko
    except ImportError:
        raise RuntimeError("SSH 远程训练需要 paramiko，请执行: pip install paramiko") from None
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=target.host,
            port=target.port,
            username=target.username,
            password=target.password or None,
            key_filename=target.key_path or None,
            look_for_keys=not (target.password or target.key_path),
            timeout=timeout,
            banner_timeout=timeout,
            auth_timeout=timeout,
        )
    except paramiko.AuthenticationException:
        raise RuntimeError(f"SSH 认证失败（{target.display}）：请检查用户名、密码或私钥") from None
    except (paramiko.SSHException, OSError) as e:
        raise RuntimeError(f"无法连接 {target.display}: {e}") from e
    return client


def _run(client, command: str, *, stdin_text: Optional[str] = None,
         line_callback: LineCallback = None, timeout: Optional[float] = None) -> tuple[int, str]:
    """执行远端命令（stderr 合并到 stdout），返回 `(退出码, 输出)`；给出 line_callback 时逐行推送。"""
    chan = client.get_transport().open_session(timeout=timeout)
    chan.set_combine_stderr(True)
    if timeout:
        chan.settimeout(timeout)
    chan.exec_command(command)
    if stdin_text is not None:
        chan.sendall(stdin_text.encode("utf-8"))
        chan.shutdown_write()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    chunks: list[str] = []
    pending = ""
    while True:
        data = chan.recv(65536)
        if not data:
            break
        text = decoder.decode(data)
        chunks.append(text)
        if line_callback:
            pending += text
            *lines, pending = re.split(r"\r\n|\n|\r", pending)
            for line in lines:
                if line.strip() and not _PROGRESS_RE.search(line):
                    line_callback(line.rstrip())
    tail = decoder.decode(b"", final=True)
    chunks.append(tail)
    if line_callback and (pending + tail).strip():
        line_callback((pending + tail).rstrip())
    return chan.recv_exit_status(), "".join(chunks)


def _env_script() -> str:
    modules = ["torch", *REQUIRED_PACKAGES, *OPTIONAL_PACKAGES]
    return f"""
import importlib, json, sys
res = {{"python": sys.version.split()[0], "torch": None, "cuda": False, "gpus": [], "missing": []}}
for m in {modules!r}:
    try:
        importlib.import_module(m)
    except Exception:
        res["missing"].append(m)
try:
    import torch
    res["torch"] = torch.__version__
    res["cuda"] = bool(torch.cuda.is_available())
    if res["cuda"]:
        res["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except Exception:
    pass
print({_ENV_MARKER!r} + json.dumps(res))
"""


def check_remote_env(client, python: str) -> RemoteEnv:
    """检查远端 Python 与训练依赖。"""
    code, out = _run(client, f"{python} -", stdin_text=_env_script(), timeout=120)
    for line in out.splitlines():
        if line.startswith(_ENV_MARKER):
            return RemoteEnv(**json.loads(line[len(_ENV_MARKER):]))
    raise RuntimeError(f"远端 Python 无法运行（命令: {python}，退出码 {code}）:\n{out.strip()[-1000:]}")


def install_packages(client, python: str, modules: list[str], line_callback: LineCallback = None) -> None:
    names = [REQUIRED_PACKAGES.get(m) or OPTIONAL_PACKAGES[m] for m in modules]
    if line_callback:
        line_callback(f"正在远端安装: {' '.join(names)}")
    code, out = _run(client, f"{python} -m pip install {' '.join(map(shlex.quote, names))}",
                     line_callback=line_callback)
    if code != 0:
        raise RuntimeError(f"远端安装依赖失败（退出码 {code}）")


def prepare_remote_env(client, target: SSHTarget, *, auto_install: bool = False,
                       line_callback: LineCallback = None) -> RemoteEnv:
    """检查环境，按需安装缺失依赖；缺少 PyTorch 或必需依赖时抛出异常。"""
    log = line_callback or (lambda _m: None)
    env = check_remote_env(client, target.python)
    log(f"远端环境: {env.describe()}")
    if env.torch is None:
        raise RuntimeError(
            f"远端 Python（{target.python}）未安装 PyTorch。请在服务器上安装与 CUDA 匹配的 torch，"
            "或在「远端 Python」中填写已装好 PyTorch 的解释器路径（如 /root/miniconda3/envs/xxx/bin/python）"
        )
    to_install = [m for m in env.missing if m != "torch"]
    if to_install and auto_install:
        install_packages(client, target.python, to_install, log)
        env = check_remote_env(client, target.python)
    if env.missing_required:
        pkgs = " ".join(REQUIRED_PACKAGES[m] for m in env.missing_required if m in REQUIRED_PACKAGES)
        raise RuntimeError(
            f"远端缺少训练依赖: {', '.join(env.missing_required)}。"
            f"可勾选「自动安装缺少的依赖」，或在服务器上执行: {target.python} -m pip install {pkgs}"
        )
    if env.missing_optional:
        log(f"提示: 远端缺少 {', '.join(env.missing_optional)}，训练曲线图将无法生成")
    return env


def _resolve_remote_dir(sftp, remote_dir: str) -> str:
    home = sftp.normalize(".")
    if remote_dir in ("", "~"):
        return home
    if remote_dir.startswith("~/"):
        return posixpath.join(home, remote_dir[2:])
    return remote_dir if remote_dir.startswith("/") else posixpath.join(home, remote_dir)


def _upload(sftp, local: Path, remote: str, label: str, log: Callable[[str], None]) -> None:
    total = local.stat().st_size
    log(f"上传{label}（{total / 1024 / 1024:.1f} MB）…")
    marks = {"next": 25}

    def progress(sent: int, size: int) -> None:
        pct = int(sent * 100 / size) if size else 100
        if pct >= marks["next"] and pct < 100:
            log(f"  {label}已上传 {pct}%")
            marks["next"] = (pct // 25 + 1) * 25

    sftp.put(str(local), remote, callback=progress)


class _LogTail:
    """增量读取远端日志文件，按行（含 `\\r` 分隔的进度刷新）推送，过滤 tqdm 进度条。"""

    def __init__(self, sftp, path: str, line_callback: LineCallback) -> None:
        self._sftp = sftp
        self._path = path
        self._cb = line_callback
        self._offset = 0
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending = ""
        self.recent: list[str] = []

    def poll(self, final: bool = False) -> None:
        try:
            with self._sftp.open(self._path, "rb") as f:
                f.seek(self._offset)
                data = f.read()
        except FileNotFoundError:
            data = b""
        self._offset += len(data)
        self._pending += self._decoder.decode(data, final=final)
        *lines, self._pending = re.split(r"\r\n|\n|\r", self._pending)
        if final:
            lines.append(self._pending)
            self._pending = ""
        for line in lines:
            line = line.rstrip()
            if not line.strip() or _PROGRESS_RE.search(line):
                continue
            self.recent = (self.recent + [line])[-20:]
            if self._cb:
                self._cb(line)


def _remote_exists(sftp, path: str) -> bool:
    try:
        sftp.stat(path)
        return True
    except FileNotFoundError:
        return False


def _kill_remote(client, job_dir: str) -> None:
    pid_file = shlex.quote(posixpath.join(job_dir, "pid"))
    _run(client, f"pid=$(cat {pid_file} 2>/dev/null) && "
                 "{ kill -TERM -- -$pid 2>/dev/null || { pkill -TERM -P $pid; kill -TERM $pid; } 2>/dev/null; }; true",
         timeout=30)


def run_ssh_training(
    target: SSHTarget,
    config_path: str | Path,
    data_root: str | Path,
    save_dir: str | Path,
    *,
    resume_from: Optional[str | Path] = None,
    use_cpu: bool = False,
    auto_install: bool = False,
    poll_interval: float = 2.0,
    line_callback: LineCallback = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """通过 SSH 在远端训练，完成后把产物下载到 `save_dir`。

    返回 `{"job_id", "remote_dir", "save_dir", "files"}`；失败抛出 `RuntimeError`，取消时抛出 `RuntimeError("cancelled")`。
    """
    log = line_callback or (lambda _m: None)
    cancelled = should_cancel or (lambda: False)

    def check_cancel() -> None:
        if cancelled():
            raise RuntimeError("cancelled")

    client = connect(target)
    try:
        log(f"已连接 {target.display}")
        prepare_remote_env(client, target, auto_install=auto_install, line_callback=log)
        check_cancel()

        sftp = client.open_sftp()
        root = _resolve_remote_dir(sftp, target.remote_dir)
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        job_dir = posixpath.join(root, "jobs", job_id)
        q = shlex.quote
        code, out = _run(client, f"mkdir -p {q(job_dir)}", timeout=30)
        if code != 0:
            raise RuntimeError(f"无法创建远端目录 {job_dir}: {out.strip()}")
        log(f"远端任务目录: {job_dir}")

        with tempfile.TemporaryDirectory(prefix="fenosuite_ssh_") as td:
            code_zip = build_bundle_zip(Path(td) / "code.zip")
            _upload(sftp, code_zip, posixpath.join(job_dir, "code.zip"), "训练代码", log)
            check_cancel()
            log("正在打包数据目录…")
            data_zip = pack_directory_to_zip(data_root)
            try:
                _upload(sftp, data_zip, posixpath.join(job_dir, "data.zip"), "数据", log)
            finally:
                data_zip.unlink(missing_ok=True)
        check_cancel()
        sftp.put(str(config_path), posixpath.join(job_dir, "config.yaml"))
        args = ["--config", "../config.yaml", "--data-root", "../data", "--save-dir", "../output"]
        if resume_from:
            _upload(sftp, Path(resume_from), posixpath.join(job_dir, "resume.pth"), "继续训练权重", log)
            args += ["--resume-from", "../resume.pth"]
        if use_cpu:
            args.append("--cpu")

        py = target.python
        code, out = _run(
            client,
            f"cd {q(job_dir)} && {py} -m zipfile -e code.zip code && {py} -m zipfile -e data.zip data"
            " && rm -f code.zip data.zip",
            timeout=1800,
        )
        if code != 0:
            raise RuntimeError(f"远端解压失败: {out.strip()[-1000:]}")
        check_cancel()

        train_cmd = f"{py} -u -m plant_phenotype.training.train {' '.join(map(q, args))}"
        inner = f"export PYTHONIOENCODING=utf-8; {train_cmd} > ../train.log 2>&1; echo $? > ../exit_code"
        launch = (
            f"cd {q(posixpath.join(job_dir, 'code'))} && "
            f"nohup setsid sh -c {q(inner)} > /dev/null 2>&1 < /dev/null & echo $! > {q(posixpath.join(job_dir, 'pid'))}"
        )
        code, out = _run(client, launch, timeout=30)
        if code != 0:
            raise RuntimeError(f"远端启动训练失败: {out.strip()}")
        log("远端训练已启动（断开连接不会中断训练）")

        import paramiko

        tail = _LogTail(sftp, posixpath.join(job_dir, "train.log"), line_callback)
        exit_file = posixpath.join(job_dir, "exit_code")
        finished = False
        try:
            polls = 0
            while True:
                tail.poll()
                if _remote_exists(sftp, exit_file):
                    tail.poll(final=True)
                    finished = True
                    break
                polls += 1
                if polls % 15 == 0:
                    alive, _ = _run(client, f"kill -0 $(cat {q(posixpath.join(job_dir, 'pid'))})", timeout=30)
                    if alive != 0 and not _remote_exists(sftp, exit_file):
                        tail.poll(final=True)
                        finished = True
                        raise RuntimeError(f"远端训练进程意外退出，日志: {job_dir}/train.log")
                slept = 0.0
                while slept < poll_interval:
                    check_cancel()
                    time.sleep(0.2)
                    slept += 0.2
        except (OSError, EOFError, paramiko.SSHException) as e:
            raise RuntimeError(f"SSH 连接中断（{e}），远端训练仍在后台运行，产物目录: {job_dir}/output") from e
        except BaseException:
            if not finished:
                try:
                    _kill_remote(client, job_dir)
                    log("已终止远端训练进程")
                except Exception:
                    log(f"未能终止远端训练进程，可登录服务器检查: {job_dir}")
            raise

        with sftp.open(exit_file, "r") as f:
            exit_code = f.read().decode("utf-8", errors="replace").strip()
        _run(client, f"rm -rf {q(posixpath.join(job_dir, 'data'))}", timeout=300)
        if exit_code != "0":
            last = "\n".join(tail.recent[-5:])
            raise RuntimeError(f"远端训练失败（退出码 {exit_code}），日志: {job_dir}/train.log\n{last}")

        out_dir = posixpath.join(job_dir, "output")
        local_dir = Path(save_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(a.filename for a in sftp.listdir_attr(out_dir) if not stat.S_ISDIR(a.st_mode or 0))
        log(f"下载训练产物（{len(files)} 个文件）到 {local_dir}")
        for name in files:
            sftp.get(posixpath.join(out_dir, name), str(local_dir / name))
        sftp.close()
        return {"job_id": job_id, "remote_dir": job_dir, "save_dir": str(local_dir), "files": files}
    finally:
        client.close()
