# -*- coding: utf-8 -*-
"""平台层：工具注册、切块工具同步执行、（可选）QtTaskRunner 烟测。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plant_phenotype.app.plugin_api import PlatformApp  # noqa: E402
from plant_phenotype.app.plugin_loader import load_builtin_plugins  # noqa: E402
from plant_phenotype.app.tab_manager import TabContribution, TabManager  # noqa: E402
from plant_phenotype.app.task_models import CancelToken  # noqa: E402
from plant_phenotype.app.plugins.annotation.tools import TileDirectoryTool  # noqa: E402
from plant_phenotype.app.plugins.root_phenotype.tools import RootAnalyzePipelineTool  # noqa: E402
from plant_phenotype.app.plugins.training.tools import RemoteTrainTool  # noqa: E402

try:
    import PyQt6.QtWidgets  # noqa: F401
except ImportError:
    _HAS_PYQT6 = False
else:
    _HAS_PYQT6 = True


class _DummyReporter:
    def log(self, message: str) -> None:
        pass

    def progress(self, current: int, total: int | None = None, message: str = "") -> None:
        pass

    def artifact(self, artifact) -> None:
        pass


class TestPlatform(unittest.TestCase):
    def test_tile_directory_tool_sync(self) -> None:
        tool = TileDirectoryTool()
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            annotated = base / "ann"
            tiles = base / "tiles"
            annotated.mkdir()
            tiles.mkdir()
            cancel = CancelToken()
            r = tool.run(
                {"annotated_dir": str(annotated), "tiles_base_dir": str(tiles)},
                PlatformApp().context,
                _DummyReporter(),
                cancel,
            )
            self.assertEqual(r.status, "success")
            self.assertIn("0", r.message)

    def test_tab_manager_order(self) -> None:
        tm = TabManager()
        stub = MagicMock()
        tm.register(
            TabContribution(
                id="z",
                title="Z",
                factory=lambda _app, _parent=None: stub,
                order=10,
            )
        )
        tm.register(
            TabContribution(
                id="a",
                title="A",
                factory=lambda _app, _parent=None: stub,
                order=0,
            )
        )
        self.assertEqual([c.id for c in tm.contributions()], ["a", "z"])

    def test_builtin_plugins_register_tools(self) -> None:
        app = PlatformApp()
        load_builtin_plugins(app)
        tool_ids = set(app.tools.ids())
        self.assertIn("training.local_train", tool_ids)
        self.assertIn("training.remote_train", tool_ids)
        self.assertIn("training.ssh_train", tool_ids)
        self.assertIn("training.ssh_check", tool_ids)
        self.assertIn("segmentation.batch_predict", tool_ids)
        self.assertIn("segmentation.preview", tool_ids)
        self.assertIn("annotation.tile_directory", tool_ids)
        self.assertIn("annotation.reassemble_tiles", tool_ids)
        self.assertIn("root.analyze", tool_ids)
        self.assertIn("datasets.import", tool_ids)
        self.assertIn("datasets.import_predictions", tool_ids)
        self.assertIn("builtin.datasets.workspace", {t.id for t in app.tabs.contributions()})

    def test_remote_train_tool_sync_mocked(self) -> None:
        from unittest.mock import patch

        tool = RemoteTrainTool()
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            cfg = base / "config.yaml"
            cfg.write_text("train:\n  epochs: 1\n", encoding="utf-8")
            data_dir = base / "data"
            data_dir.mkdir()
            (data_dir / "sample.txt").write_text("ok", encoding="utf-8")

            logs: list[str] = []

            class _LogReporter(_DummyReporter):
                def log(self, message: str) -> None:
                    logs.append(message)

            poll_SideEffects = [
                ("running", ["epoch 1/2", "loss: 0.5"], None),
                ("success", ["epoch 1/2", "loss: 0.5", "epoch 2/2", "dice: 0.91"], None),
            ]
            with (
                patch("plant_phenotype.training.remote_client.submit_remote_job", return_value="job-123") as mock_sub,
                patch("plant_phenotype.training.remote_client.poll_remote_job", side_effect=poll_SideEffects),
            ):
                res = tool.run(
                    {
                        "base_url": "http://localhost:8765",
                        "config_path": str(cfg),
                        "data_root": str(data_dir),
                        "poll_interval": 0.01,
                    },
                    PlatformApp().context,
                    _LogReporter(),
                    CancelToken(),
                )
            self.assertEqual(res.status, "success")
            self.assertEqual(res.metrics.get("job_id"), "job-123")
            mock_sub.assert_called_once()
            self.assertIn("任务已提交: job-123", logs)
            self.assertIn("dice: 0.91", logs)

    def test_root_analyze_tool_sync(self) -> None:
        mask_dir = _ROOT / "tests" / "fixtures" / "root_masks"
        if not mask_dir.is_dir():
            self.skipTest("缺少 root_masks fixtures")
        mask_path = mask_dir / "straight_horizontal.png"
        if not mask_path.is_file():
            self.skipTest("缺少 straight_horizontal.png")
        tool = RootAnalyzePipelineTool()
        app = PlatformApp()
        r = tool.run(
            {"mask_path": str(mask_path), "batch_mode": True},
            app.context,
            _DummyReporter(),
            CancelToken(),
        )
        self.assertEqual(r.status, "success")
        self.assertIsNotNone(app.context.get_pipeline_result())
        self.assertEqual(app.context.get_last_mask_path(), str(mask_path))

    @unittest.skipUnless(_HAS_PYQT6, "需要安装 PyQt6")
    def test_qt_task_runner_root_analyze_tool(self) -> None:
        from PyQt6.QtCore import QEventLoop, QTimer
        from PyQt6.QtWidgets import QApplication

        from plant_phenotype.app.task_runner import QtTaskRunner

        mask_dir = _ROOT / "tests" / "fixtures" / "root_masks"
        if not mask_dir.is_dir():
            self.skipTest("缺少 root_masks fixtures")
        mask_path = mask_dir / "straight_horizontal.png"
        if not mask_path.is_file():
            self.skipTest("缺少 straight_horizontal.png")

        qapp = QApplication.instance() or QApplication(sys.argv)
        _ = qapp

        shell = PlatformApp()
        shell.register_tool(RootAnalyzePipelineTool())
        runner = QtTaskRunner(shell.tools, shell.context)
        states: list = []
        loop = QEventLoop()

        def on_state(s) -> None:
            states.append(s.status)
            if s.status in ("success", "failed", "cancelled"):
                loop.quit()

        runner.task_state_changed.connect(on_state)
        runner.run_tool(
            "root.analyze",
            {"mask_path": str(mask_path), "batch_mode": True},
        )
        QTimer.singleShot(30_000, loop.quit)
        loop.exec()
        self.assertIn("success", states)
        self.assertIsNotNone(shell.context.get_pipeline_result())

    @unittest.skipUnless(_HAS_PYQT6, "需要安装 PyQt6")
    def test_qt_task_runner_tile_tool(self) -> None:
        from PyQt6.QtCore import QEventLoop, QTimer
        from PyQt6.QtWidgets import QApplication

        from plant_phenotype.app.task_runner import QtTaskRunner

        qapp = QApplication.instance() or QApplication(sys.argv)
        _ = qapp

        shell = PlatformApp()
        shell.register_tool(TileDirectoryTool())
        runner = QtTaskRunner(shell.tools, shell.context)

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            annotated = base / "ann"
            tiles = base / "tiles"
            annotated.mkdir()
            tiles.mkdir()

            states: list = []

            def on_state(s) -> None:
                states.append(s.status)

            runner.task_state_changed.connect(on_state)

            loop = QEventLoop()

            def maybe_quit(s) -> None:
                if s.status in ("success", "failed", "cancelled"):
                    loop.quit()

            runner.task_state_changed.connect(maybe_quit)
            runner.run_tool(
                "annotation.tile_directory",
                {
                    "annotated_dir": str(annotated),
                    "tiles_base_dir": str(tiles),
                },
            )
            QTimer.singleShot(20_000, loop.quit)
            loop.exec()
            self.assertIn("success", states)


if __name__ == "__main__":
    unittest.main()
