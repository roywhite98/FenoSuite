# -*- coding: utf-8 -*-
"""run_one 可选分段计时（环境变量 PROFILE=1）。"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

_log = logging.getLogger("plant_phenotype.root_phenotype.core.pipeline")


def profiling_enabled() -> bool:
    return os.environ.get("PROFILE", "").strip() in ("1", "true", "yes")


class StageTimer:
    def __init__(self) -> None:
        self._times: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not profiling_enabled():
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._times[name] = self._times.get(name, 0.0) + (time.perf_counter() - t0)

    def log_summary(self, prefix: str = "") -> None:
        if not self._times:
            return
        total = sum(self._times.values())
        parts = ", ".join(f"{k}={v * 1000:.1f}ms" for k, v in sorted(self._times.items()))
        _log.info("%sPROFILE total=%.1fms: %s", prefix, total * 1000, parts)


_timer: Optional[StageTimer] = None


def get_timer() -> StageTimer:
    global _timer
    if _timer is None:
        _timer = StageTimer()
    return _timer


def reset_timer() -> None:
    global _timer
    _timer = StageTimer()
