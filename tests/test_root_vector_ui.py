"""Offscreen UI checks for transactions, stale jobs, picking, and workflow."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from copy import deepcopy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QImage
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QGraphicsPathItem, QGraphicsSimpleTextItem

from plant_phenotype.app.plugin_api import PlatformApp
from plant_phenotype.app.task_models import Artifact, TaskState, ToolResult
from plant_phenotype.app.task_runner import QtTaskRunner
from plant_phenotype.app.plugins.root_phenotype_beta.controller import RootVectorController
from plant_phenotype.app.plugins.root_phenotype_beta.plugin import get_plugin
from plant_phenotype.app.plugins.root_phenotype_beta.tab import RootPhenotypeBetaTab
from plant_phenotype.gui.root_vector_canvas import RootVectorCanvas
from plant_phenotype.root_phenotype.facade import BaseMark, RootDocument, RootNode, VectorRoot, get_facade


class FakeRunner(QObject):
    task_state_changed = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.calls = []
        self.cancelled = []

    def run_tool(self, tool_id, inputs):
        task_id = f"task-{len(self.calls)}"
        self.calls.append((task_id, tool_id, deepcopy(inputs)))
        return task_id

    def cancel(self, task_id):
        self.cancelled.append(task_id)

    def finish(self, index, document):
        task_id, tool_id, inputs = self.calls[index]
        document.document_id = inputs["document_id"]
        result = ToolResult(status="success", artifacts=[Artifact("vector_document", "object", data=document)],
            metrics={k: inputs[k] for k in ("request_id", "document_id", "revision", "source_fingerprint")})
        self.task_state_changed.emit(TaskState(task_id, tool_id, "test", "success", result=result))


def make_document():
    doc = RootDocument()
    doc.source.width = 120
    doc.source.height = 100
    doc.roots = [VectorRoot("root-a", [RootNode("n0", 20, 10, 4), RootNode("n1", 20, 50, 4), RootNode("n2", 20, 90, 2)],
        base_mark_id="base-a", direction="base_to_tip", review_state="auto")]
    doc.base_marks = [BaseMark("base-a", 20, 10, "root-a")]
    return doc


class VectorUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.platform = PlatformApp()
        self.runner = FakeRunner()
        self.platform.set_task_runner(self.runner)

    def test_pending_primary_uses_dashes_without_root_order_labels(self):
        canvas = RootVectorCanvas()
        document = make_document()
        document.roots[0].review_state = 'pending'
        canvas.set_document(document)
        paths = [item for item in canvas.scene().items() if isinstance(item, QGraphicsPathItem)]
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].pen().style(), Qt.PenStyle.DashLine)
        labels = [item.text() for item in canvas.scene().items() if isinstance(item, QGraphicsSimpleTextItem)]
        self.assertFalse(any('侧根' in label or '主根' in label for label in labels))
        canvas.close()

    def test_plugin_registers_independent_tab_and_tools(self):
        get_plugin().register(self.platform)
        self.assertIn("root.vector.analyze", self.platform.tools.ids())
        self.assertIn("root.vector.resolve", self.platform.tools.ids())
        self.assertNotIn("root.analyze", self.platform.tools.ids())

    def test_superseded_job_and_cancel_cannot_publish(self):
        c = RootVectorController(self.platform)
        c.analyze("first.png", [])
        c.analyze("second.png", [])
        self.runner.finish(0, make_document())
        self.assertIsNone(c.document)
        self.runner.finish(1, make_document())
        self.assertIs(c.document, self.platform.context.get_vector_document())
        self.assertEqual(c.document.document_id, self.runner.calls[1][2]["document_id"])
        current = c.document
        c.analyze("second.png", [])
        c.cancel()
        self.runner.finish(2, make_document())
        self.assertIs(c.document, current)

    def test_file_change_discards_old_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"mask.png"
            path.write_bytes(b"first")
            c = RootVectorController(self.platform)
            c.analyze(str(path), [])
            path.write_bytes(b"different-size")
            self.runner.finish(0, make_document())
            self.assertIsNone(c.document)

    def test_edit_invalidates_jobs_and_history_revisions_are_monotonic(self):
        c = RootVectorController(self.platform)
        c.load_document(make_document())
        r0 = c.document.revision
        c.analyze("mask.png", [])
        self.assertTrue(c.apply_edit({"type": "move_node", "root_id": "root-a", "node_id": "n1", "x": 24, "y": 51}))
        self.runner.finish(0, make_document())
        self.assertEqual(c.document.roots[0].nodes[1].x, 24)
        r1 = c.document.revision
        c.undo()
        self.assertEqual(c.document.roots[0].nodes[1].x, 20)
        r2 = c.document.revision
        c.redo()
        self.assertEqual(c.document.roots[0].nodes[1].x, 24)
        self.assertLess(r0, r1)
        self.assertLess(r1, r2)
        self.assertLess(r2, c.document.revision)
        self.assertIs(c.document, self.platform.context.get_vector_document())
        c.reset()
        self.assertIsNone(self.platform.context.get_vector_document())

    def test_canvas_coordinate_picking_and_full_size_export(self):
        canvas = RootVectorCanvas()
        canvas.resize(480, 400)
        canvas.show()
        doc = make_document()
        doc.roots.append(VectorRoot("root-b", [RootNode("b0", 0, 50, 2), RootNode("b1", 100, 50, 2)]))
        canvas.set_document(doc)
        canvas.fit_image()
        self.qt.processEvents()
        self.assertEqual(set(canvas.pick_roots(QPointF(20, 50))), {"root-a", "root-b"})
        canvas.scale(2, 2)
        self.assertEqual(canvas.pick_roots(QPointF(20, 12)), ["root-a"])
        positions = []
        canvas.baseClicked.connect(lambda x, y: positions.append((x, y)))
        canvas.set_mode("base")
        QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=canvas.mapFromScene(20, 50))
        self.assertAlmostEqual(positions[0][0], 20, delta=.5)
        self.assertAlmostEqual(positions[0][1], 50, delta=.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"full.png"
            canvas.export_png(path)
            image = QImage(str(path))
            self.assertEqual((image.width(), image.height()), (120, 100))
        canvas.close()

    def test_source_missing_document_remains_editable(self):
        get_plugin().register(self.platform)
        tab = RootPhenotypeBetaTab(platform_app=self.platform)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"roots.json"
            get_facade().save_vector_document(make_document(), path)
            tab.load_document(path)
            self.assertEqual(tab.tree.topLevelItemCount(), 1)
            self.assertTrue(tab.save_btn.isEnabled())
            self.assertFalse(tab.resolve_btn.isEnabled())
            self.assertTrue(tab.diameter_btn.isEnabled())
            self.assertIn("源掩码不可用", tab.status.text())
        tab.close()

    def test_crossing_ports_are_distinct_and_correction_is_undoable(self):
        get_plugin().register(self.platform)
        tasks = QtTaskRunner(self.platform.tools, self.platform.context)
        self.platform.set_task_runner(tasks)
        tab = RootPhenotypeBetaTab(platform_app=self.platform)
        errors = []
        tab.controller.errorOccurred.disconnect(tab._show_error)
        tab.controller.errorOccurred.connect(errors.append)
        with tempfile.TemporaryDirectory() as directory:
            mask = np.zeros((121, 121), np.uint8)
            cv2.line(mask, (60, 10), (60, 110), 255, 3)
            cv2.line(mask, (10, 60), (110, 60), 255, 3)
            path = Path(directory)/'cross.png'
            cv2.imencode('.png', mask)[1].tofile(str(path))
            tab.load_mask(path)
            doc = tab.facade.analyze_vectors(str(path), [(60, 10)]).document
            tab.controller.load_document(doc)
            decision = doc.junction_decisions[0]
            self.assertEqual(decision.state, 'pending')
            positions = [tab.canvas.port_display_position(p) for p in doc.analysis_snapshot.ports]
            self.assertEqual(len({(round(p.x(), 2), round(p.y(), 2)) for p in positions}), 4)
            crossing = min((a for a in decision.alternatives if a.kind == 'crossing'), key=lambda a:a.cost)
            tab.controller.resolve(decision.id, crossing.id)
            deadline = time.monotonic()+15
            while tab.controller.busy and time.monotonic() < deadline:
                QTest.qWait(10)
            tasks.wait_all(1000)
            self.qt.processEvents()
            self.assertFalse(errors, errors)
            self.assertEqual(len(tab.controller.document.roots), 2)
            self.assertTrue(all(r.parent_attachment is None for r in tab.controller.document.roots))
            tab.controller.undo()
            self.assertEqual(tab.controller.document.junction_decisions[0].state, 'pending')
            tab.controller.redo()
            self.assertEqual(tab.controller.document.junction_decisions[0].chosen, crossing.id)
        tab.close()

    def test_real_background_analysis_and_json_round_trip(self):
        get_plugin().register(self.platform)
        tasks = QtTaskRunner(self.platform.tools, self.platform.context)
        self.platform.set_task_runner(tasks)
        tab = RootPhenotypeBetaTab(platform_app=self.platform)
        errors = []
        tab.controller.errorOccurred.disconnect(tab._show_error)
        tab.controller.errorOccurred.connect(errors.append)
        with tempfile.TemporaryDirectory() as directory:
            mask = np.zeros((120, 120), np.uint8)
            cv2.line(mask, (50, 10), (50, 110), 255, 5)
            cv2.line(mask, (50, 55), (95, 85), 255, 3)
            cv2.line(mask, (110, 10), (110, 110), 255, 3)
            mask_path = Path(directory)/"mask.png"
            cv2.imencode(".png", mask)[1].tofile(str(mask_path))
            tab.load_mask(mask_path)
            tab._add_base(0, 0)
            self.assertEqual(tab._base_marks, [])
            tab._add_base(95, 85)  # A side-root tip selects the whole connected plant.
            self.assertTrue(tab.controller.busy, "clicking a root should start analysis")
            deadline = time.monotonic()+15
            while tab.controller.busy and time.monotonic() < deadline:
                QTest.qWait(10)
            tasks.wait_all(1000)
            self.qt.processEvents()
            self.assertFalse(tab.controller.busy, "background analysis did not complete")
            self.assertFalse(errors, errors)
            self.assertIsNotNone(tab.controller.document)
            self.assertGreaterEqual(len(tab.controller.document.roots), 1)
            doc = tab.controller.document
            self.assertEqual(sorted(doc.root_order(r.id) for r in doc.roots), [0, 1])
            self.assertTrue(all(n.x < 105 for r in doc.roots for n in r.nodes))
            self.assertIn('主根 1 条', tab.overview.text())
            self.assertIn('侧根 1 条', tab.overview.text())
            self.assertIs(tab.controller.document, self.platform.context.get_vector_document())
            json_path = Path(directory)/"roots.json"
            tab.facade.save_vector_document(tab.controller.document, json_path)
            tab.load_document(json_path)
            self.assertEqual(len(tab.controller.document.base_marks), 1)
            self.assertEqual(tab._base_marks[0]['x'], 95)
            self.assertEqual(tab._base_marks[0]['y'], 85)
            self.assertTrue(tab._source_valid)
            tab.diameter.setValue(6)
            tab._set_diameter()
            self.assertEqual(tab.controller.document.root_by_id(tab._selected_root_id).nodes[0].diameter_px, 6)
            tab.controller.undo()
            self.assertNotEqual(tab.controller.document.root_by_id(tab._selected_root_id).nodes[0].diameter_px, 6)
        tab.close()

    def test_click_after_analysis_adds_next_root(self):
        get_plugin().register(self.platform)
        tasks = QtTaskRunner(self.platform.tools, self.platform.context)
        self.platform.set_task_runner(tasks)
        tab = RootPhenotypeBetaTab(platform_app=self.platform)
        errors = []
        tab.controller.errorOccurred.disconnect(tab._show_error)
        tab.controller.errorOccurred.connect(errors.append)

        def wait():
            deadline = time.monotonic()+15
            while tab.controller.busy and time.monotonic() < deadline:
                QTest.qWait(10)
            tasks.wait_all(1000)
            self.qt.processEvents()
            self.assertFalse(tab.controller.busy)
            self.assertFalse(errors, errors)

        with tempfile.TemporaryDirectory() as directory:
            mask = np.zeros((120, 160), np.uint8)
            cv2.line(mask, (30, 10), (30, 110), 255, 5)
            cv2.line(mask, (130, 10), (130, 110), 255, 5)
            mask_path = Path(directory)/"two.png"
            cv2.imencode(".png", mask)[1].tofile(str(mask_path))
            tab.load_mask(mask_path)
            tab._add_base(30, 60)
            wait()
            first = tab.controller.document
            self.assertTrue(all(n.x < 80 for r in first.roots for n in r.nodes))
            self.assertEqual(tab.mode.currentData(), "base")
            tab._add_base(130, 60)
            self.assertTrue(tab.controller.busy)
            wait()
            doc = tab.controller.document
            self.assertEqual(len(tab._base_marks), 2)
            self.assertTrue(any(n.x > 80 for r in doc.roots for n in r.nodes))
            self.assertTrue(any(n.x < 80 for r in doc.roots for n in r.nodes))
            tab._remove_base()
            wait()
            self.assertEqual(len(tab._base_marks), 1)
            self.assertTrue(all(n.x < 80 for r in tab.controller.document.roots for n in r.nodes))
            tab._remove_base()
            self.assertIsNone(tab.controller.document)
            self.assertEqual(tab._base_marks, [])
        tab.close()

    def test_dataset_sample_opens_mask_and_original(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            platform = PlatformApp(base / "datasets")
            platform.set_task_runner(self.runner)
            tab = RootPhenotypeBetaTab(platform_app=platform)
            image = np.full((40, 50, 3), 80, np.uint8)
            mask = np.zeros((40, 50), np.uint8)
            mask[10:30, 20:24] = 255
            image_path = base / "plant.png"
            cv2.imencode(".png", image)[1].tofile(str(image_path))
            ds = platform.datasets.store.create("根系样本")
            sample = ds.add_image(image_path)
            mask_path = ds.masks_dir / "plant.png"
            cv2.imencode(".png", mask)[1].tofile(str(mask_path))
            sample.mask = "masks/plant.png"
            sample.status = "annotated"
            ds.save()
            platform.datasets.set_current(ds.id)
            QTest.qWait(0)
            self.assertTrue(tab._dataset_panel.select_sample(sample.id))
            self.assertEqual(tab._current_sample, (ds.id, sample.id))
            self.assertEqual(Path(tab._mask_path).resolve(), mask_path.resolve())
            self.assertEqual(Path(tab._original_path).resolve(), ds.abs(sample.image).resolve())
            self.assertEqual(tab._dataset_panel.display_mode(), "mask")
            self.assertEqual(Path(tab._view_path).resolve(), mask_path.resolve())
            tab._dataset_panel.set_display_mode("image")
            self.assertEqual(Path(tab._view_path).resolve(), ds.abs(sample.image).resolve())
            self.assertIn("plant.png", tab.mask_name.text())
            tab._leave_dataset_mode()
            self.assertIsNone(tab._current_sample)
        tab.close()


if __name__ == "__main__":
    unittest.main()
