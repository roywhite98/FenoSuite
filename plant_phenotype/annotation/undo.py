"""标注层操作的撤销/重做栈。"""
from __future__ import annotations

from typing import Optional
from .layer import AnnotationLayer


class UndoStack:
    def __init__(self, max_size: int = 50):
        self._undo_stack: list = []
        self._redo_stack: list = []
        self._max_size = max_size

    def push(self, state: AnnotationLayer) -> None:
        self._redo_stack.clear()
        self._undo_stack.append(state.copy())
        if len(self._undo_stack) > self._max_size:
            self._undo_stack.pop(0)

    def undo(self, current: AnnotationLayer) -> Optional[AnnotationLayer]:
        if not self._undo_stack:
            return None
        self._redo_stack.append(current.copy())
        return self._undo_stack.pop()

    def redo(self, current: AnnotationLayer) -> Optional[AnnotationLayer]:
        if not self._redo_stack:
            return None
        self._undo_stack.append(current.copy())
        return self._redo_stack.pop()

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
