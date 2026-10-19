from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QPlainTextEdit,
    QDialogButtonBox, QMessageBox, QFileDialog,
)


def equalize_widths(widgets: Iterable[QWidget]) -> None:
    """把一组同类按钮的宽度对齐到组内最宽者，避免同排按钮长短参差。

    需在控件已加入布局后调用，否则样式表带来的内边距不会计入 sizeHint。
    """
    items = list(widgets)
    if not items:
        return
    for w in items:
        w.ensurePolished()
    width = max(w.sizeHint().width() for w in items)
    for w in items:
        w.setMinimumWidth(width)


def match_width(widgets: Iterable[QWidget], reference: QWidget) -> None:
    """把一组按钮固定为与 reference 相同的宽度，避免在网格布局中随列宽被拉伸。

    需在控件已加入布局后调用，否则样式表带来的内边距不会计入 sizeHint。
    """
    reference.ensurePolished()
    width = reference.sizeHint().width()
    reference.setFixedWidth(width)
    for w in widgets:
        w.setFixedWidth(width)


def show_yaml_editor_dialog(parent: QWidget, path: str | Path) -> None:
    """
    打开 YAML 编辑对话框，支持保存。
    替换 TrainingTab._preview_yaml 与 SegmentationTab._preview_yaml 中的重复逻辑。

    Args:
        parent: 父窗口控件
        path:   YAML 文件路径
    """
    path = Path(path)
    if not path.is_file():
        QMessageBox.warning(parent, "预览 YAML", f"配置文件不存在:\n{path}")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        QMessageBox.warning(parent, "预览 YAML", f"无法读取文件: {e}")
        return
    dlg = QDialog(parent)
    dlg.setWindowTitle(f"编辑配置 — {path.name}")
    dlg.setMinimumSize(520, 400)
    layout = QVBoxLayout(dlg)
    te = QPlainTextEdit()
    te.setPlainText(text)
    layout.addWidget(te)
    btns = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close
    )
    def do_save():
        try:
            path.write_text(te.toPlainText(), encoding="utf-8")
            QMessageBox.information(dlg, "保存", f"已保存: {path}")
        except Exception as e:
            QMessageBox.warning(dlg, "保存失败", str(e))
    btns.button(QDialogButtonBox.StandardButton.Save).clicked.connect(do_save)
    btns.button(QDialogButtonBox.StandardButton.Close).clicked.connect(dlg.close)
    layout.addWidget(btns)
    dlg.exec()


def browse_directory(
    parent: QWidget,
    title: str,
    settings_key: str,
    fallback: str = "",
) -> Optional[str]:
    """
    弹出目录选择对话框，记忆上次路径到 QSettings。
    返回选中路径字符串，取消则返回 None。

    Args:
        parent:       父窗口控件
        title:        对话框标题
        settings_key: QSettings 键名（格式建议 "last_dir/xxx"）
        fallback:     无历史记录时的起始目录
    """
    saved = QSettings().value(settings_key, None)
    start = str(saved) if saved and Path(str(saved)).is_dir() else (fallback or str(Path.home()))
    path = QFileDialog.getExistingDirectory(parent, title, start)
    if path:
        QSettings().setValue(settings_key, path)
        return path
    return None


def browse_file(
    parent: QWidget,
    title: str,
    settings_key: str,
    file_filter: str = "All (*)",
    fallback: str = "",
) -> Optional[str]:
    """
    弹出文件选择对话框，记忆上次目录到 QSettings。
    返回选中文件路径字符串，取消则返回 None。

    Args:
        parent:       父窗口控件
        title:        对话框标题
        settings_key: QSettings 键名
        file_filter:  文件过滤器字符串
        fallback:     无历史记录时的起始目录
    """
    saved = QSettings().value(settings_key, None)
    start = str(saved) if saved and Path(str(saved)).is_dir() else (fallback or str(Path.home()))
    path, _ = QFileDialog.getOpenFileName(parent, title, start, file_filter)
    if path:
        QSettings().setValue(settings_key, str(Path(path).parent))
        return path
    return None
