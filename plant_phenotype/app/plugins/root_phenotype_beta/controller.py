"""UI-thread document ownership, revision guards, and undo/redo snapshots."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from PyQt6.QtCore import QObject, pyqtSignal

from plant_phenotype.root_phenotype.facade import get_facade


class RootVectorController(QObject):
    documentChanged = pyqtSignal(object)
    busyChanged = pyqtSignal(bool)
    statusChanged = pyqtSignal(str)
    errorOccurred = pyqtSignal(str)
    historyChanged = pyqtSignal(bool, bool)

    def __init__(self, app, parent=None, facade=None):
        super().__init__(parent)
        self.app = app
        self.facade = facade or get_facade()
        self.document = None
        self._undo = []
        self._redo = []
        self._clock = 0
        self._request = None
        self._task_id = None
        self._source_input = None
        if app.tasks is not None:
            app.tasks.task_state_changed.connect(self._task_changed)

    @property
    def busy(self):
        return self._request is not None

    def cancel(self):
        task = self._task_id
        self._task_id = None
        self._request = None
        self._source_input = None
        if task is not None and self.app.tasks is not None:
            self.app.tasks.cancel(task)
        self.busyChanged.emit(False)

    def reset(self):
        self.cancel()
        self.document = None
        self._undo.clear()
        self._redo.clear()
        self._clock += 1
        self._publish()
        self.historyChanged.emit(False, False)

    def load_document(self, document):
        self.cancel()
        self._undo.clear()
        self._redo.clear()
        self._commit(deepcopy(document), remember=False)

    def _commit(self, document, remember=True):
        if remember and self.document is not None:
            self._undo.append(deepcopy(self.document))
            self._undo = self._undo[-50:]
        self._redo.clear()
        self._clock = max(self._clock, document.revision) + 1
        document.revision = self._clock
        self.document = document
        self._publish()
        self.historyChanged.emit(bool(self._undo), bool(self._redo))

    def apply_edit(self, command):
        if self.document is None:
            return False
        try:
            # Run the domain transaction on a snapshot before invalidating work.
            result = self.facade.apply_vector_edit(deepcopy(self.document), command)
            self.cancel()
            self._commit(result.document)
            return True
        except Exception as exc:
            self.errorOccurred.emit(str(exc))
            return False

    def undo(self):
        self._restore(self._undo, self._redo)

    def redo(self):
        self._restore(self._redo, self._undo)

    def _restore(self, source, target):
        if not source or self.document is None:
            return
        self.cancel()
        target.append(deepcopy(self.document))
        restored = source.pop()
        self._clock += 1
        restored.revision = self._clock
        self.document = restored
        self._publish()
        self.historyChanged.emit(bool(self._undo), bool(self._redo))

    def _publish(self):
        self.app.context.set_vector_document(self.document)
        self.documentChanged.emit(self.document)

    @staticmethod
    def _source_signature(source_input):
        paths, config = source_input
        sources = []
        for path in paths:
            if path:
                p = Path(path)
                try:
                    st = p.stat()
                    sources.append((str(p.resolve()), st.st_size, st.st_mtime_ns))
                except OSError:
                    sources.append((str(p), None, None))
            else:
                sources.append(None)
        return hashlib.sha256(json.dumps([sources, config], sort_keys=True).encode()).hexdigest()

    def analyze(self, mask_path, base_marks, original_image_path=None, config=None, *, selection_points=None):
        document_id = self.document.document_id if self.document is not None else uuid4().hex
        return self._start("root.vector.analyze", {
            "mask_path": str(mask_path), "base_marks": deepcopy(base_marks),
            "original_image_path": original_image_path, "config": config or {},
            "selection_points": deepcopy(selection_points),
        }, document_id)

    def resolve(self, junction_id, alternative_id):
        if self.document is None:
            return
        if not self.facade.vector_source_matches(self.document):
            self.errorOccurred.emit("源掩码缺失或内容已改变，不能复用旧接续快照；请重新分析。现有矢量仍可编辑和导出。")
            return
        return self._start("root.vector.resolve", {
            "document": deepcopy(self.document), "decisions": {junction_id: alternative_id},
        }, self.document.document_id)

    def _start(self, tool_id, inputs, document_id):
        if self.app.tasks is None:
            self.errorOccurred.emit("任务运行器尚未就绪")
            return None
        self.cancel()
        source_doc = inputs.get("document")
        if source_doc is not None:
            source_input = ([source_doc.source.mask_path, source_doc.source.image_path], asdict(source_doc.config))
        else:
            source_input = ([inputs.get("mask_path"), inputs.get("original_image_path")], inputs.get("config", {}))
        request = {
            "request_id": uuid4().hex, "document_id": document_id,
            "revision": self.document.revision if self.document is not None else self._clock,
            "source_fingerprint": self._source_signature(source_input),
        }
        self._source_input = deepcopy(source_input)
        self._request = request
        inputs.update(request)
        self.busyChanged.emit(True)
        self.statusChanged.emit("正在解析…")
        try:
            self._task_id = self.app.tasks.run_tool(tool_id, inputs)
        except Exception as exc:
            self.cancel()
            self.errorOccurred.emit(str(exc))
        return self._task_id

    def _task_changed(self, state):
        if state.id != self._task_id or self._request is None:
            return
        self.statusChanged.emit(state.message)
        if state.status not in ("success", "failed", "cancelled"):
            return
        request = self._request
        source_input = self._source_input
        self._task_id = None
        self._request = None
        self._source_input = None
        self.busyChanged.emit(False)
        if state.status == "failed":
            self.errorOccurred.emit(state.error or state.message or "解析失败")
            return
        if state.status != "success" or state.result is None:
            return
        if any(state.result.metrics.get(k) != v for k, v in request.items()):
            self.statusChanged.emit("已忽略过期的分析结果")
            return
        if self._source_signature(source_input) != request["source_fingerprint"]:
            self.statusChanged.emit("源文件已变化，已忽略旧分析结果；请重新分析")
            return
        if self.document is not None and (
            self.document.document_id != request["document_id"] or
            self.document.revision != request["revision"]
        ):
            self.statusChanged.emit("文档已有更新，已忽略旧分析结果")
            return
        doc = next((a.data for a in state.result.artifacts if a.id == "vector_document"), None)
        if doc is None or doc.document_id != request["document_id"]:
            self.errorOccurred.emit("任务未返回匹配的矢量文档")
            return
        self._commit(doc)
