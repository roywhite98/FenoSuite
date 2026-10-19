# -*- coding: utf-8 -*-
"""右侧工作目录树：与 dataset-tool/rootmask 一致的 QTreeView + QFileSystemModel，支持右键菜单。"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

from PyQt6.QtCore import Qt, pyqtSignal, QDir, QSettings, QEvent
from PyQt6.QtGui import QAction, QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QTreeView,
    QHeaderView,
    QAbstractItemView,
    QFileDialog,
    QMessageBox,
    QMenu,
    QInputDialog,
    QSizePolicy,
)

try:
    from PyQt6.QtWidgets import QFileSystemModel
except ImportError:
    try:
        from PyQt6.QtGui import QFileSystemModel
    except ImportError:
        QFileSystemModel = None


class WorkdirTreePanel(QWidget):
    """
    工作目录树面板：标题 +「选择工作目录」+ QTreeView（QFileSystemModel 或回退）。
    选中文件时发射 fileSelected(Path)；支持右键菜单：删除、新建文件夹、在文件管理器打开。
    与 dataset-tool / rootmask 的右侧边栏逻辑一致。
    """
    fileSelected = pyqtSignal(object)  # Path

    def __init__(
        self,
        parent=None,
        title: str = "工作目录",
        file_extensions: Optional[Tuple[str, ...]] = None,
        settings_key: Optional[str] = None,
    ):
        super().__init__(parent)
        self._root_path: Optional[Path] = None
        self._file_extensions = file_extensions  # 若不为 None，仅在这些扩展名时发射 fileSelected
        self._settings_key_str = settings_key if settings_key else "last_work_directory"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        # 外层已有标题（如 QDockWidget）时传 title=""，避免重复「工作目录」等字样
        if title and str(title).strip():
            layout.addWidget(QLabel(title))
        btn = QPushButton("选择工作目录...")
        btn.clicked.connect(self._select_root)
        layout.addWidget(btn)
        self._tree = QTreeView()
        self._tree.setHeaderHidden(True)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._tree.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        if QFileSystemModel is not None:
            self._fs_model = QFileSystemModel()
            self._fs_model.setFilter(QDir.Filter.AllDirs | QDir.Filter.Files | QDir.Filter.NoDotAndDotDot)
            self._fs_model.setRootPath("")
            self._tree.setModel(self._fs_model)
            self._tree.setRootIndex(self._fs_model.index(""))
            self._tree.hideColumn(1)
            self._tree.hideColumn(2)
            self._tree.hideColumn(3)
            self._std_model = None
        else:
            self._fs_model = None
            self._std_model = QStandardItemModel()
            self._tree.setModel(self._std_model)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setMinimumSectionSize(40)
        self._tree.expanded.connect(self._on_tree_expanded_or_collapsed)
        self._tree.collapsed.connect(self._on_tree_expanded_or_collapsed)
        sm = self._tree.selectionModel()
        if sm is not None:
            sm.selectionChanged.connect(self._on_selection_changed)
        self._tree.installEventFilter(self)
        layout.addWidget(self._tree, 1)
        self.setMaximumWidth(280)
        self.setSizePolicy(self.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Expanding)
        self._restore_last_workdir()

    def _settings_key(self) -> str:
        return self._settings_key_str

    def _restore_last_workdir(self) -> None:
        """启动时恢复上次选择的工作目录。"""
        saved = QSettings().value(self._settings_key(), None)
        if saved is not None:
            saved = str(saved).strip()
        if saved and Path(saved).is_dir():
            self.set_root_path(Path(saved))

    def _select_root(self) -> None:
        saved = QSettings().value(self._settings_key(), None)
        if saved is not None:
            saved = str(saved).strip()
        start = str(Path.home())
        if saved and Path(saved).is_dir():
            start = saved
        elif self._root_path and self._root_path.is_dir():
            start = str(self._root_path)
        d = QFileDialog.getExistingDirectory(self, "选择工作目录（原图/掩码所在根目录）", start)
        if not d:
            return
        self.set_root_path(Path(d))

    def _on_tree_expanded_or_collapsed(self) -> None:
        """展开或折叠节点后重新计算列宽，使滚动条随显示内容适应。"""
        self._tree.resizeColumnToContents(0)

    def set_root_path(self, path: Path) -> None:
        self._root_path = Path(path)
        if self._root_path.is_dir():
            QSettings().setValue(self._settings_key(), str(self._root_path.resolve()))
        if self._fs_model is not None:
            self._fs_model.setRootPath(str(self._root_path))
            self._tree.setRootIndex(self._fs_model.index(str(self._root_path)))
            self._tree.expandToDepth(0)
            self._tree.resizeColumnToContents(0)
        else:
            self._std_model.removeRows(0, self._std_model.rowCount())
            try:
                for p in sorted(self._root_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                    item = QStandardItem(p.name + ("/" if p.is_dir() else ""))
                    item.setData(str(p.resolve()), Qt.ItemDataRole.UserRole)
                    item.setEditable(False)
                    self._std_model.appendRow(item)
            except OSError:
                pass

    def get_root_path(self) -> Optional[Path]:
        return self._root_path

    def _path_from_index(self, index):
        if not index.isValid():
            return None
        if self._fs_model is not None:
            return Path(self._fs_model.filePath(index))
        path = self._std_model.data(index, Qt.ItemDataRole.UserRole)
        return Path(path) if path else None

    def _get_selected_paths(self) -> List[Path]:
        """返回当前选中的全部路径（去重，每行一条）。支持 Ctrl/Shift 多选后的批量操作。"""
        sm = self._tree.selectionModel()
        if sm is None:
            return []
        paths: List[Path] = []
        seen: Set[Path] = set()
        for idx in sm.selectedIndexes():
            p = self._path_from_index(idx)
            if p is not None and p not in seen:
                seen.add(p)
                paths.append(p)
        return paths

    def _on_selection_changed(self) -> None:
        indexes = self._tree.selectionModel().selectedIndexes()
        if not indexes:
            return
        path = self._path_from_index(indexes[0])
        if path is None or not path.is_file():
            return
        if self._file_extensions is not None:
            if path.suffix.lower() not in {e.lower() for e in self._file_extensions}:
                return
        self.fileSelected.emit(path)

    def eventFilter(self, obj, event) -> bool:
        """处理工作目录树快捷键：Delete 删除（带确认）、F2 重命名。"""
        if obj == self._tree and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                paths = self._get_selected_paths()
                if paths:
                    self._workdir_delete(paths)
                return True
            if key == Qt.Key.Key_F2:
                index = self._tree.currentIndex()
                path = self._path_from_index(index)
                if path is not None:
                    self._workdir_rename(path)
                return True
        return super().eventFilter(obj, event)

    def _on_context_menu(self, pos) -> None:
        index = self._tree.indexAt(pos)
        path = self._path_from_index(index) if index.isValid() else None
        if path is None and self._root_path is not None:
            path = self._root_path
        selected_paths = self._get_selected_paths()
        if path is not None and path in selected_paths and len(selected_paths) > 1:
            paths_for_delete = selected_paths
        else:
            paths_for_delete = [path] if path is not None else []
        menu = QMenu(self)
        act_delete = QAction("删除", self)
        act_rename = QAction("重命名", self)
        act_new = QAction("新建文件夹", self)
        act_open = QAction("在文件管理器打开", self)
        act_delete.triggered.connect(lambda: self._workdir_delete(paths_for_delete))
        act_rename.triggered.connect(lambda: self._workdir_rename(path))
        act_new.triggered.connect(lambda: self._workdir_new_folder(path))
        act_open.triggered.connect(lambda: self._workdir_open(path))
        menu.addAction(act_delete)
        menu.addAction(act_rename)
        menu.addAction(act_new)
        menu.addAction(act_open)
        menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _workdir_delete(self, paths: Optional[Path] | List[Path]) -> None:
        """删除一个或多个路径。paths 可为单个 Path 或路径列表。"""
        if paths is None:
            paths = []
        elif isinstance(paths, Path):
            paths = [paths]
        paths = [p for p in paths if p is not None and p.exists()]
        if not paths:
            QMessageBox.information(self, "删除", "请选中要删除的文件或文件夹。")
            return
        root_resolved = self._root_path.resolve() if self._root_path else None
        allowed = []
        for p in paths:
            if root_resolved and p.resolve() == root_resolved:
                QMessageBox.warning(self, "删除", "不能删除工作目录根。")
                return
            allowed.append(p)
        paths = allowed
        if not paths:
            return
        n = len(paths)
        if n == 1:
            msg = f"确定从磁盘删除吗？\n{paths[0]}"
        else:
            preview = "\n".join(p.name for p in paths[:10])
            if n > 10:
                preview += f"\n... 共 {n} 项"
            msg = f"确定从磁盘删除以下 {n} 项吗？\n\n{preview}"
        reply = QMessageBox.question(
            self,
            "删除",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        failed = []
        for p in paths:
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            except OSError as e:
                failed.append((p, e))
        if self._fs_model is not None and self._root_path is not None:
            root_str = str(self._root_path)
            self._fs_model.setRootPath(root_str)
            self._tree.setRootIndex(self._fs_model.index(root_str))
        else:
            self.set_root_path(self._root_path)
        if failed:
            QMessageBox.critical(
                self, "错误", "部分删除失败:\n" + "\n".join(f"{p}: {e}" for p, e in failed)
            )
        elif n == 1:
            QMessageBox.information(self, "删除", f"已删除: {paths[0].name}")
        else:
            QMessageBox.information(self, "删除", f"已删除 {n} 项。")

    def _workdir_rename(self, path: Optional[Path]) -> None:
        if path is None or not path.exists():
            QMessageBox.information(self, "重命名", "请选中要重命名的文件或文件夹。")
            return
        if self._root_path is not None and path.resolve() == self._root_path.resolve():
            QMessageBox.warning(self, "重命名", "不能重命名工作目录根。")
            return
        default_name = path.name
        new_name, ok = QInputDialog.getText(
            self, "重命名", "新名称:", text=default_name
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        if new_name == default_name:
            return
        new_path = path.parent / new_name
        if new_path.exists():
            QMessageBox.warning(self, "重命名", f"已存在同名项: {new_name}")
            return
        try:
            path.rename(new_path)
            QMessageBox.information(self, "重命名", f"已重命名为: {new_name}")
            if self._fs_model is not None and self._root_path is not None:
                root_str = str(self._root_path)
                self._fs_model.setRootPath(root_str)
                self._tree.setRootIndex(self._fs_model.index(root_str))
            else:
                self.set_root_path(self._root_path)
        except OSError as e:
            QMessageBox.critical(self, "错误", f"重命名失败: {e}")

    def _workdir_new_folder(self, path: Optional[Path]) -> None:
        if path is None:
            if self._root_path is None or not self._root_path.is_dir():
                QMessageBox.warning(self, "新建文件夹", "请先选择工作目录，或在树中选中一个文件夹。")
                return
            parent = self._root_path
        else:
            parent = path if path.is_dir() else path.parent
        name, ok = QInputDialog.getText(self, "新建文件夹", "文件夹名称:", text="新建文件夹")
        if not ok or not name.strip():
            return
        name = name.strip()
        new_path = parent / name
        try:
            new_path.mkdir(parents=True, exist_ok=True)
            QMessageBox.information(self, "新建文件夹", f"已创建: {new_path}")
            if self._fs_model is not None and self._root_path is not None:
                root_str = str(self._root_path)
                self._fs_model.setRootPath(root_str)
                self._tree.setRootIndex(self._fs_model.index(root_str))
            else:
                self.set_root_path(self._root_path)
        except OSError as e:
            QMessageBox.critical(self, "错误", f"创建失败: {e}")

    def _workdir_open(self, path: Optional[Path]) -> None:
        if path is None:
            if self._root_path is None or not self._root_path.is_dir():
                QMessageBox.warning(self, "在文件管理器打开", "请先选择工作目录。")
                return
            path = self._root_path
        else:
            path = path.parent if path.is_file() else path
        path = path.resolve()
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开失败: {e}")
