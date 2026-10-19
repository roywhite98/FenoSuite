# -*- coding: utf-8 -*-
"""SSH 远程训练：地址解析、最小代码包、日志增量读取与工具接线。"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plant_phenotype.app.plugin_api import PlatformApp  # noqa: E402
from plant_phenotype.app.plugins.training.tools import SSHTrainTool, _ssh_target  # noqa: E402
from plant_phenotype.app.task_models import CancelToken  # noqa: E402
from plant_phenotype.training.bundle import build_bundle_zip, collect_module_files  # noqa: E402
from plant_phenotype.training.ssh_client import _LogTail, parse_ssh_address  # noqa: E402


class _Reporter:
    def __init__(self) -> None:
        self.logs: list[str] = []

    def log(self, message: str) -> None:
        self.logs.append(message)

    def progress(self, current, total=None, message="") -> None:
        pass

    def artifact(self, artifact) -> None:
        pass


class TestParseSSHAddress(unittest.TestCase):
    def test_ssh_command(self) -> None:
        self.assertEqual(
            parse_ssh_address("ssh -p 2222 root@gpu.example.com"),
            ("gpu.example.com", 2222, "root", None),
        )

    def test_ip_only(self) -> None:
        self.assertEqual(parse_ssh_address("192.168.1.100"), ("192.168.1.100", 22, None, None))
        t = _ssh_target({"ssh_address": "192.168.1.100", "password": "pw"})
        self.assertEqual((t.host, t.port, t.username), ("192.168.1.100", 22, "root"))

    def test_other_forms(self) -> None:
        self.assertEqual(parse_ssh_address("root@host:2222"), ("host", 2222, "root", None))
        self.assertEqual(parse_ssh_address("ssh://ubuntu@host:2200/"), ("host", 2200, "ubuntu", None))
        self.assertEqual(parse_ssh_address("10.0.0.5"), ("10.0.0.5", 22, None, None))
        self.assertEqual(
            parse_ssh_address('ssh -o StrictHostKeyChecking=no -i "C:\\keys\\id rsa" -p2201 -l me host'),
            ("host", 2201, "me", "C:\\keys\\id rsa"),
        )

    def test_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_ssh_address("ssh -p 2222")
        with self.assertRaises(ValueError):
            parse_ssh_address("root@host:abc")

    def test_target_defaults(self) -> None:
        t = _ssh_target({"ssh_address": "ssh -p 2222 gpu.example.com", "password": "pw"})
        self.assertEqual((t.username, t.port, t.password, t.python), ("root", 2222, "pw", "python"))


class TestBundle(unittest.TestCase):
    def test_closure_is_minimal(self) -> None:
        rel = {f.relative_to(_ROOT).as_posix() for f in collect_module_files()}
        self.assertIn("plant_phenotype/__init__.py", rel)
        self.assertIn("plant_phenotype/training/train.py", rel)
        self.assertIn("plant_phenotype/models/build_model.py", rel)
        for prefix in ("plant_phenotype/gui/", "plant_phenotype/app/", "plant_phenotype/annotation/",
                       "plant_phenotype/root_phenotype/", "plant_phenotype/server/"):
            self.assertFalse(any(r.startswith(prefix) for r in rel), prefix)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("albumentations"),
                         "需要训练依赖")
    def test_bundle_runs_standalone(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            z = build_bundle_zip(Path(td) / "code.zip")
            code_dir = Path(td) / "code"
            with zipfile.ZipFile(z) as zf:
                zf.extractall(code_dir)
            proc = subprocess.run(
                [sys.executable, "-c",
                 "import plant_phenotype.training.train as t, plant_phenotype.training.visualize; print(t.__file__)"],
                cwd=code_dir, capture_output=True, text=True, timeout=300,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(Path(proc.stdout.strip()).resolve().is_relative_to(code_dir.resolve()))


class _FakeFile:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass

    def seek(self, pos: int) -> None:
        self._pos = pos

    def read(self) -> bytes:
        return self._data[self._pos:]


class _FakeSFTP:
    def __init__(self) -> None:
        self.data = b""

    def open(self, path, mode="r"):
        return _FakeFile(self.data)


class TestLogTail(unittest.TestCase):
    def test_incremental_and_progress_filter(self) -> None:
        sftp = _FakeSFTP()
        lines: list[str] = []
        tail = _LogTail(sftp, "/x/train.log", lines.append)
        text = "训练样本 8 个\nTrain 1/2:  50%|#####     | 1/2 [00:01<00:01, 1.2it/s]\r"
        sftp.data = text.encode("utf-8")[:-5]
        tail.poll()
        sftp.data = (text + "Train 1/2: 100%|##########| 2/2 [00:02<00:00, 1.1it/s]\nEpoch [1/2] val_dice=0.5").encode("utf-8")
        tail.poll()
        self.assertEqual(lines, ["训练样本 8 个"])
        tail.poll(final=True)
        self.assertEqual(lines, ["训练样本 8 个", "Epoch [1/2] val_dice=0.5"])


class TestSSHTrainTool(unittest.TestCase):
    def test_run_mocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            cfg = base / "config.yaml"
            cfg.write_text("train:\n  epochs: 1\n", encoding="utf-8")
            data = base / "data"
            data.mkdir()
            out = base / "out"
            out.mkdir()
            (out / "best_model.pth").write_bytes(b"x")
            inputs = {
                "ssh_address": "ssh -p 2222 root@192.168.1.100",
                "password": "pw",
                "config_path": str(cfg),
                "data_root": str(data),
                "save_dir": str(out),
            }
            app = PlatformApp()
            tool = SSHTrainTool()
            tool.validate(inputs, app.context)
            fake = {"job_id": "j1", "remote_dir": "/root/x", "save_dir": str(out), "files": ["best_model.pth"]}
            with patch("plant_phenotype.training.ssh_client.run_ssh_training", return_value=fake) as mock_run:
                res = tool.run(inputs, app.context, _Reporter(), CancelToken())
            self.assertEqual(res.status, "success")
            target = mock_run.call_args.args[0]
            self.assertEqual((target.host, target.port, target.username), ("192.168.1.100", 2222, "root"))
            self.assertEqual(mock_run.call_args.kwargs["save_dir"], str(out))
            self.assertEqual(str(app.context.get_last_checkpoint_path()), str(out / "best_model.pth"))


if __name__ == "__main__":
    unittest.main()
