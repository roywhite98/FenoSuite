# -*- coding: utf-8 -*-
"""工作区 Tab 注册与填充（不在模块顶层依赖 PyQt6，便于无 GUI 环境测试）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal


@dataclass(frozen=True)
class TabContribution:
    id: str
    title: str
    factory: Callable[[Any, Any], Any]
    area: Literal["workspace", "side", "result"] = "workspace"
    order: int = 100


class TabManager:
    def __init__(self) -> None:
        self._tabs: list[TabContribution] = []

    def register(self, tab: TabContribution) -> None:
        if any(t.id == tab.id for t in self._tabs):
            raise ValueError(f"Tab id 已注册: {tab.id}")
        self._tabs.append(tab)

    def contributions(self) -> list[TabContribution]:
        return sorted(self._tabs, key=lambda t: (t.order, t.id))

    def populate_workspace(
        self,
        tab_widget: Any,
        app: Any,
        parent: Any,
    ) -> dict[str, Any]:
        """按 order 将 workspace 区 Tab 加入 QTabWidget，返回 id -> widget。"""
        out: dict[str, Any] = {}
        for c in self.contributions():
            if c.area != "workspace":
                continue
            w = c.factory(app, parent)
            w.setObjectName(c.id)
            tab_widget.addTab(w, c.title)
            out[c.id] = w
        return out
