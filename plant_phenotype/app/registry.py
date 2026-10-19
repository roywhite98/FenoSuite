# -*- coding: utf-8 -*-
"""Tool 注册表。"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, List

if TYPE_CHECKING:
    from plant_phenotype.app.tool_api import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        tid = tool.spec.id
        if tid in self._tools:
            raise ValueError(f"Tool id 已注册: {tid}")
        self._tools[tid] = tool

    def get(self, tool_id: str) -> Tool:
        if tool_id not in self._tools:
            raise KeyError(f"未知 Tool: {tool_id}")
        return self._tools[tool_id]

    def list(self, category: str | None = None) -> List[Tool]:
        tools = list(self._tools.values())
        if category is None:
            return sorted(tools, key=lambda t: t.spec.id)
        return sorted(
            [t for t in tools if t.spec.category == category],
            key=lambda t: t.spec.id,
        )

    def ids(self) -> Iterator[str]:
        return iter(sorted(self._tools.keys()))
