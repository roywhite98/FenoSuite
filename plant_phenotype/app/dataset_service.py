# -*- coding: utf-8 -*-
"""当前数据集服务：各功能模块共享同一个选中的数据集，并在变化时收到通知。

监听回调在调用 ``set_current`` / ``notify_changed`` 的线程中执行，GUI 中只应从主线程调用；
后台 Tool 只读写 ``DatasetStore``，完成后由界面在主线程通知。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from plant_phenotype.datasets.paths import default_datasets_root
from plant_phenotype.datasets.store import Dataset, DatasetStore

_LOG = logging.getLogger(__name__)
_STATE_FILE = ".state.json"

Listener = Callable[[str | None], None]


class DatasetService:
    def __init__(self, root: str | Path | None = None) -> None:
        self.store = DatasetStore(root or default_datasets_root())
        self._current_id: str | None = None
        self._state_loaded = False
        self._current_listeners: list[Listener] = []
        self._changed_listeners: list[Listener] = []

    @property
    def root(self) -> Path:
        return self.store.root

    def _state_path(self) -> Path:
        return self.root / _STATE_FILE

    def _ensure_state(self) -> None:
        if self._state_loaded:
            return
        self._state_loaded = True
        try:
            data = json.loads(self._state_path().read_text(encoding="utf-8"))
            did = data.get("current")
        except (OSError, ValueError):
            did = None
        self._current_id = did if did and self.store.exists(did) else None

    def _persist(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._state_path().write_text(json.dumps({"current": self._current_id}), encoding="utf-8")
        except OSError:
            _LOG.warning("无法保存当前数据集状态", exc_info=True)

    @property
    def current_id(self) -> str | None:
        self._ensure_state()
        if self._current_id and not self.store.exists(self._current_id):
            self._current_id = None
        return self._current_id

    def current(self) -> Dataset | None:
        did = self.current_id
        return self.store.load(did) if did else None

    def set_current(self, dataset_id: str | None) -> None:
        self._ensure_state()
        if dataset_id is not None and not self.store.exists(dataset_id):
            raise ValueError(f"数据集不存在: {dataset_id}")
        if dataset_id == self._current_id:
            return
        self._current_id = dataset_id
        self._persist()
        for cb in list(self._current_listeners):
            cb(dataset_id)

    def notify_changed(self, dataset_id: str | None = None) -> None:
        """数据集内容（样本、标签、名称）或数据集列表发生变化。"""
        if self._current_id and not self.store.exists(self._current_id):
            self.set_current(None)
        for cb in list(self._changed_listeners):
            cb(dataset_id)

    def on_current_changed(self, cb: Listener) -> None:
        self._current_listeners.append(cb)

    def on_dataset_changed(self, cb: Listener) -> None:
        self._changed_listeners.append(cb)

    def remove_listener(self, cb: Listener) -> None:
        # Qt can emit destroyed while cyclic GC has already cleared this service.
        # Detaching from a disposed service must remain an idempotent operation.
        for lst in (getattr(self, '_current_listeners', ()), getattr(self, '_changed_listeners', ())):
            if cb in lst:
                lst.remove(cb)
