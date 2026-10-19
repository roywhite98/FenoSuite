# -*- coding: utf-8 -*-
"""骨架化 / 拓扑图结果 LRU 缓存。"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

_CacheValue = Tuple[np.ndarray, np.ndarray]


class _LRUCache:
    def __init__(self, maxsize: int = 64) -> None:
        self._maxsize = max(1, int(maxsize))
        self._data: OrderedDict[tuple, Any] = OrderedDict()

    def get(self, key: tuple) -> Optional[Any]:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: tuple, value: Any) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


# 兼容旧名
SkeletonCache = _LRUCache

_skeleton_cache = _LRUCache(maxsize=64)
_graph_cache = _LRUCache(maxsize=32)


def get_skeleton_cache() -> _LRUCache:
    return _skeleton_cache


def get_graph_cache() -> _LRUCache:
    return _graph_cache


def clear_skeleton_cache() -> None:
    _skeleton_cache.clear()
    _graph_cache.clear()


def mask_mtime_ns(mask_path: str) -> int:
    try:
        return Path(mask_path).stat().st_mtime_ns
    except OSError:
        return 0
