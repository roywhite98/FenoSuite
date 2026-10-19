# -*- coding: utf-8 -*-
"""数据集页使用的对话框：新建、导入（配对规则 + 像素值映射）、批量配对、映射编辑、随机划分。"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from plant_phenotype.datasets import importer, mask_io
from plant_phenotype.datasets.pairing import LAYOUT_LABELS, PairingRules, discover_pairs, index_masks
from plant_phenotype.datasets.schema import (
    BACKGROUND_ID,
    SPLITS,
    TASK_TYPE_LABELS,
    LabelClass,
    default_color,
    split_label,
)
from plant_phenotype.datasets.store import allocate_counts
from plant_phenotype.gui.utils import browse_directory


def _progress_dialog(parent: QWidget, title: str) -> tuple[QProgressDialog, callable]:
    dlg = QProgressDialog(title, "取消", 0, 100, parent)
    dlg.setWindowModality(Qt.WindowModality.WindowModal)
    dlg.setMinimumDuration(300)

    def progress(cur: int, total: int, msg: str) -> None:
        dlg.setMaximum(max(1, total))
        dlg.setValue(min(cur, max(1, total)))
        dlg.setLabelText(f"{title}\n{msg}")
        QApplication.processEvents()

    return dlg, progress


def _swatch_item(key: str) -> QTableWidgetItem:
    item = QTableWidgetItem(key)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if mask_io.is_color_key(key):
        r, g, b = mask_io.parse_color_key(key)
        item.setBackground(QColor(r, g, b))
        item.setForeground(QColor(255, 255, 255) if (r * 299 + g * 587 + b * 114) < 128000 else QColor(0, 0, 0))
    return item


class NewDatasetDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("新建数据集")
        form = QFormLayout(self)
        self._name = QLineEdit()
        form.addRow("名称:", self._name)
        self._type = QComboBox()
        for key, label in TASK_TYPE_LABELS.items():
            self._type.addItem(label, key)
        form.addRow("分割类型:", self._type)
        self._classes = QLineEdit("根")
        self._classes.setToolTip("前景类别名称，用逗号分隔；背景（编号 0）自动添加")
        form.addRow("前景类别:", self._classes)
        hint = QLabel("类别按顺序编号为 1、2、3…；编号即语义掩码中保存的像素值。")
        hint.setProperty("mdRole", "caption")
        form.addRow(hint)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _accept(self) -> None:
        if not self._name.text().strip():
            QMessageBox.warning(self, "新建数据集", "请填写名称。")
            return
        self.accept()

    def values(self) -> tuple[str, str, list[LabelClass]]:
        names = [n.strip() for n in self._classes.text().replace("，", ",").split(",") if n.strip()]
        classes = [LabelClass(0, "背景", default_color(0))]
        classes += [LabelClass(i, n, default_color(i)) for i, n in enumerate(dict.fromkeys(names), start=1)]
        return self._name.text().strip(), self._type.currentData(), classes


class MappingTable(QTableWidget):
    """源像素值/颜色 → 类别编号与名称。编号相同即合并为同一类别。"""

    COLS = ("源值 / 颜色", "像素数", "文件数", "类别编号", "类别名称")

    def __init__(self, parent=None) -> None:
        super().__init__(0, len(self.COLS), parent)
        self.setHorizontalHeaderLabels(self.COLS)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)

    def populate(self, keys: dict[str, dict[str, int]], mapping: dict[str, int], classes: list[LabelClass]) -> None:
        names = {c.id: c.name for c in classes}
        self.setRowCount(0)
        order = sorted(keys, key=lambda k: (mask_io.is_color_key(k), -keys[k].get("pixels", 0), k))
        for key in order:
            r = self.rowCount()
            self.insertRow(r)
            self.setItem(r, 0, _swatch_item(key))
            for col, field in ((1, "pixels"), (2, "files")):
                it = QTableWidgetItem(str(keys[key].get(field, "")))
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.setItem(r, col, it)
            spin = QSpinBox()
            spin.setRange(0, 65535)
            spin.setValue(int(mapping.get(key, 0)))
            self.setCellWidget(r, 3, spin)
            self.setItem(r, 4, QTableWidgetItem(names.get(int(mapping.get(key, 0)), "")))

    def result(self) -> tuple[dict[str, int], list[LabelClass]]:
        mapping: dict[str, int] = {}
        names: dict[int, str] = {BACKGROUND_ID: "背景"}
        colors: dict[int, tuple[int, int, int]] = {}
        for r in range(self.rowCount()):
            key = self.item(r, 0).text()
            cid = self.cellWidget(r, 3).value()
            mapping[key] = cid
            name = (self.item(r, 4).text() if self.item(r, 4) else "").strip()
            if cid != BACKGROUND_ID and name and cid not in names:
                names[cid] = name
            if mask_io.is_color_key(key) and cid not in colors and cid != BACKGROUND_ID:
                colors[cid] = mask_io.parse_color_key(key)
        ids = sorted({BACKGROUND_ID, *mapping.values()})
        classes = [
            LabelClass(cid, names.get(cid, f"类别{cid}"), colors.get(cid, default_color(cid))) for cid in ids
        ]
        return mapping, classes


class ImportDialog(QDialog):
    """选择源目录 → 扫描配对与掩码取值 → 确认映射与名称。"""

    def __init__(self, parent=None, source: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("导入数据集")
        self.resize(760, 640)
        self._scanned = False
        layout = QVBoxLayout(self)

        src_grp = QGroupBox("源目录与配对规则")
        form = QFormLayout(src_grp)
        row = QHBoxLayout()
        self._source = QLineEdit(source or "")
        row.addWidget(self._source, 1)
        btn = QPushButton("浏览...")
        btn.clicked.connect(self._browse)
        row.addWidget(btn)
        form.addRow("源目录:", row)
        self._layout = QComboBox()
        for key, label in LAYOUT_LABELS.items():
            self._layout.addItem(label, key)
        form.addRow("目录布局:", self._layout)
        self._suffix = QLineEdit("_mask")
        self._suffix.setToolTip("掩码文件名 = 原图名 + 后缀；分目录布局中后缀可有可无")
        form.addRow("掩码后缀:", self._suffix)
        sub_row = QHBoxLayout()
        self._images_sub = QLineEdit("images")
        self._masks_sub = QLineEdit("masks")
        sub_row.addWidget(QLabel("原图子目录"))
        sub_row.addWidget(self._images_sub)
        sub_row.addWidget(QLabel("掩码子目录"))
        sub_row.addWidget(self._masks_sub)
        form.addRow("分目录名:", sub_row)
        self._recursive = QCheckBox("包含子目录")
        self._recursive.setChecked(True)
        form.addRow(self._recursive)
        scan_btn = QPushButton("扫描")
        scan_btn.setProperty("mdRole", "tonal")
        scan_btn.clicked.connect(self._scan)
        form.addRow(scan_btn)
        layout.addWidget(src_grp)

        self._summary = QPlainTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setMaximumHeight(110)
        self._summary.setPlaceholderText("点击「扫描」查看配对结果")
        layout.addWidget(self._summary)

        map_grp = QGroupBox("像素值 / 颜色 → 类别映射（编号相同即合并；编号 0 为背景）")
        map_ly = QVBoxLayout(map_grp)
        self._kind_label = QLabel("")
        self._kind_label.setProperty("mdRole", "caption")
        map_ly.addWidget(self._kind_label)
        self._table = MappingTable()
        map_ly.addWidget(self._table)
        layout.addWidget(map_grp, 1)

        out = QFormLayout()
        self._name = QLineEdit(Path(source).name if source else "")
        out.addRow("数据集名称:", self._name)
        self._status = QComboBox()
        self._status.addItem("已标注", "annotated")
        self._status.addItem("预测/候选（需复核）", "predicted")
        out.addRow("掩码样本状态:", self._status)
        self._include_unpaired = QCheckBox("同时导入没有掩码的原图（作为未标注样本）")
        self._include_unpaired.setChecked(True)
        out.addRow(self._include_unpaired)
        self._keep_splits = QCheckBox("保留源目录中的 train / val / test 划分")
        self._keep_splits.setChecked(True)
        self._keep_splits.setEnabled(False)
        self._keep_splits.setToolTip("扫描后若在路径中识别到 train / val / test（及 training、valid、validation、"
                                     "testing）目录，导入后样本放入数据集对应的 train/、val/、test/ 子目录")
        out.addRow(self._keep_splits)
        layout.addLayout(out)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("导入")
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _browse(self) -> None:
        path = browse_directory(self, "选择要导入的目录", "last_dir/dataset_import")
        if path:
            self._source.setText(path)
            if not self._name.text().strip():
                self._name.setText(Path(path).name)
            self._scanned = False

    def _rules(self) -> PairingRules:
        return PairingRules(
            layout=self._layout.currentData(),
            mask_suffix=self._suffix.text().strip(),
            images_subdir=self._images_sub.text().strip() or "images",
            masks_subdir=self._masks_sub.text().strip() or "masks",
            recursive=self._recursive.isChecked(),
        )

    def _scan(self) -> None:
        src = Path(self._source.text().strip())
        if not src.is_dir():
            QMessageBox.warning(self, "扫描", "请选择有效的源目录。")
            return
        try:
            result = discover_pairs(src, self._rules())
        except Exception as e:
            QMessageBox.warning(self, "扫描失败", str(e))
            return
        dlg, progress = _progress_dialog(self, "统计掩码取值…")
        try:
            scan = importer.scan_mask_values(result.masks, progress=progress, cancel=dlg.wasCanceled)
        except RuntimeError:
            return
        finally:
            dlg.close()
        mapping, classes = importer.suggest_mapping(scan)
        self._table.populate(scan.keys or {"0": {}, "255": {}}, mapping, classes)
        kind_text = {"binary": "二值掩码", "index": "类别编号掩码", "color": "彩色掩码"}[scan.kind]
        self._kind_label.setText(
            f"识别为：{kind_text}。导入后统一保存为类别编号图（PNG 无损），界面颜色可在标签管理中修改。"
        )
        splits = result.split_counts
        self._keep_splits.setEnabled(bool(splits))
        split_line = ("划分: " + " · ".join(f"{split_label(k or None)} {splits[k]}"
                                             for k in (*SPLITS, "") if splits.get(k))
                      if splits else "划分: 未识别到 train / val / test 子目录（导入后可在数据集页划分）")
        lines = [
            f"布局: {LAYOUT_LABELS.get(result.layout, result.layout)}",
            f"原图 {len(result.pairs)} 张，已配对 {result.n_paired} 张，未配对原图 {len(result.pairs) - result.n_paired} 张，"
            f"未匹配掩码 {len(result.orphan_masks)} 个",
            split_line,
        ]
        lines += [f"冲突: {c}" for c in result.conflicts[:20]]
        lines += [f"读取失败: {p}: {e}" for p, e in scan.errors[:20]]
        lines += [f"未匹配掩码: {p}" for p in result.orphan_masks[:20]]
        self._summary.setPlainText("\n".join(lines))
        if not self._name.text().strip():
            self._name.setText(src.name)
        self._scanned = True

    def _accept(self) -> None:
        if not self._scanned:
            self._scan()
            if not self._scanned:
                return
        if not self._name.text().strip():
            QMessageBox.warning(self, "导入", "请填写数据集名称。")
            return
        self.accept()

    def tool_inputs(self) -> dict:
        mapping, classes = self._table.result()
        return {
            "source_dir": self._source.text().strip(),
            "name": self._name.text().strip(),
            "rules": self._rules().to_dict(),
            "mapping": mapping,
            "classes": [c.to_dict() for c in classes],
            "mask_status": self._status.currentData(),
            "include_unpaired": self._include_unpaired.isChecked(),
            "keep_splits": self._keep_splits.isEnabled() and self._keep_splits.isChecked(),
        }


class PairMasksDialog(QDialog):
    """按命名规则把外部掩码目录批量配对到当前数据集的样本。"""

    def __init__(self, dataset, parent=None) -> None:
        super().__init__(parent)
        self._ds = dataset
        self.setWindowTitle("批量配对掩码")
        self.resize(560, 420)
        form = QFormLayout(self)
        row = QHBoxLayout()
        self._dir = QLineEdit()
        row.addWidget(self._dir, 1)
        btn = QPushButton("浏览...")
        btn.clicked.connect(self._browse)
        row.addWidget(btn)
        form.addRow("掩码目录:", row)
        self._suffix = QLineEdit("_mask")
        form.addRow("掩码后缀:", self._suffix)
        self._recursive = QCheckBox("包含子目录")
        self._recursive.setChecked(True)
        form.addRow(self._recursive)
        self._overwrite = QCheckBox("覆盖已有掩码的样本")
        form.addRow(self._overwrite)
        mapping_text = ", ".join(f"{k}→{v}" for k, v in sorted(dataset.meta.value_mapping.items()))
        lbl = QLabel(f"使用数据集映射: {mapping_text or '（空）'}")
        lbl.setWordWrap(True)
        lbl.setProperty("mdRole", "caption")
        form.addRow(lbl)
        preview_btn = QPushButton("预览匹配")
        preview_btn.clicked.connect(self._preview)
        form.addRow(preview_btn)
        self._result = QPlainTextEdit()
        self._result.setReadOnly(True)
        form.addRow(self._result)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("开始配对")
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _browse(self) -> None:
        path = browse_directory(self, "选择掩码目录", "last_dir/dataset_pair_masks")
        if path:
            self._dir.setText(path)

    def _preview(self) -> bool:
        d = Path(self._dir.text().strip())
        if not d.is_dir():
            QMessageBox.warning(self, "批量配对", "请选择有效的掩码目录。")
            return False
        report = importer.pair_masks_from_dir(
            self._ds, d, self._suffix.text().strip(), self._recursive.isChecked(),
            overwrite=self._overwrite.isChecked(), only_preview=True,
        )
        masks = list(index_masks(d, self._suffix.text().strip(), self._recursive.isChecked()).values())
        unknown = set()
        for p in masks[:200]:
            try:
                unknown |= set(mask_io.unique_keys(mask_io.read_raw_mask(p))) - set(self._ds.meta.value_mapping)
            except Exception:
                continue
        lines = [f"可配对 {report['matched']} 个样本；已有掩码而跳过 {report['skipped_annotated']} 个"]
        if unknown:
            lines.append(f"⚠ 掩码中存在数据集映射未覆盖的值 {sorted(unknown)[:20]}，这些文件会配对失败；"
                         "请先在标签页「编辑映射」中补充。")
        lines += [f"冲突: {c}" for c in report["conflicts"][:20]]
        lines += [f"{x['sample']} ← {Path(x['mask']).name}" for x in report.get("preview", [])[:50]]
        self._result.setPlainText("\n".join(lines))
        return True

    def _accept(self) -> None:
        if self._preview():
            self.accept()

    def tool_inputs(self) -> dict:
        return {
            "mask_dir": self._dir.text().strip(),
            "mask_suffix": self._suffix.text().strip(),
            "recursive": self._recursive.isChecked(),
            "overwrite": self._overwrite.isChecked(),
        }


class MappingEditDialog(QDialog):
    """编辑数据集的像素值/颜色 → 类别映射（用于之后指定或批量配对外部掩码）。"""

    def __init__(self, dataset, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑像素值 / 颜色映射")
        self.resize(460, 420)
        self._ids = sorted(dataset.meta.class_ids)
        self._names = {c.id: c.name for c in dataset.meta.classes}
        layout = QVBoxLayout(self)
        tip = QLabel("外部掩码中的像素值（如 255）或颜色（如 #ff0000）映射到类别编号。")
        tip.setWordWrap(True)
        tip.setProperty("mdRole", "caption")
        layout.addWidget(tip)
        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(("源值 / 颜色", "类别"))
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table, 1)
        for key, cid in sorted(dataset.meta.value_mapping.items()):
            self._add_row(key, cid)
        row = QHBoxLayout()
        add = QPushButton("新增行")
        add.clicked.connect(lambda: self._add_row("", self._ids[0] if self._ids else 0))
        rm = QPushButton("删除选中行")
        rm.clicked.connect(self._remove_rows)
        row.addWidget(add)
        row.addWidget(rm)
        row.addStretch(1)
        layout.addLayout(row)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _add_row(self, key: str, cid: int) -> None:
        r = self._table.rowCount()
        self._table.insertRow(r)
        self._table.setItem(r, 0, QTableWidgetItem(key))
        combo = QComboBox()
        for i in self._ids:
            combo.addItem(f"{i}  {self._names.get(i, '')}", i)
        combo.setCurrentIndex(max(0, combo.findData(cid)))
        self._table.setCellWidget(r, 1, combo)

    def _remove_rows(self) -> None:
        for r in sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True):
            self._table.removeRow(r)

    def mapping(self) -> dict[str, int]:
        out = {}
        for r in range(self._table.rowCount()):
            key = (self._table.item(r, 0).text() if self._table.item(r, 0) else "").strip().lower()
            if key:
                out[key] = int(self._table.cellWidget(r, 1).currentData())
        return out

    def _accept(self) -> None:
        for key in self.mapping():
            ok = key.isdigit() or (key.startswith("#") and len(key) == 7 and
                                   all(ch in "0123456789abcdef" for ch in key[1:]))
            if not ok:
                QMessageBox.warning(self, "映射", f"无效的源值: {key}（应为整数或 #rrggbb）")
                return
        self.accept()


class SplitDialog(QDialog):
    """按比例随机划分 train / val / test；文件会移动到数据集的对应子目录。"""

    _DEFAULT_PERCENT = {"train": 70, "val": 20, "test": 10}

    def __init__(self, dataset, selected_ids: list[str], parent=None) -> None:
        super().__init__(parent)
        self._ds = dataset
        self._selected = list(selected_ids)
        self.setWindowTitle("随机划分 train / val / test")
        form = QFormLayout(self)

        self._percent: dict[str, QSpinBox] = {}
        for split in SPLITS:
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setSuffix(" %")
            spin.setValue(self._DEFAULT_PERCENT[split])
            spin.valueChanged.connect(self._update_preview)
            self._percent[split] = spin
            form.addRow(f"{split_label(split)} ({split}):", spin)

        self._scope = QComboBox()
        n_unsplit = sum(1 for s in dataset.samples if not s.split)
        self._scope.addItem(f"全部样本（{len(dataset.samples)} 个，重新划分）", "all")
        self._scope.addItem(f"仅未划分样本（{n_unsplit} 个，已划分的保持不变）", "unsplit")
        if self._selected:
            self._scope.addItem(f"仅选中样本（{len(self._selected)} 个）", "selected")
        if dataset.has_splits and n_unsplit:
            self._scope.setCurrentIndex(1)
        self._scope.currentIndexChanged.connect(self._update_preview)
        form.addRow("划分范围:", self._scope)

        self._seed = QSpinBox()
        self._seed.setRange(0, 2_147_483_647)
        self._seed.setValue(42)
        self._seed.setToolTip("相同种子与样本得到相同划分，便于复现")
        form.addRow("随机种子:", self._seed)

        self._preview = QLabel("")
        self._preview.setWordWrap(True)
        form.addRow(self._preview)
        note = QLabel("按样本状态（已标注 / 预测 / 未标注）分层抽样，使各子集的已标注比例接近。"
                      "原图与掩码会移动到数据集的 train/、val/、test/ 子目录。")
        note.setWordWrap(True)
        note.setProperty("mdRole", "caption")
        form.addRow(note)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("开始划分")
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)
        self._update_preview()

    def _ratios(self) -> dict[str, float]:
        return {k: float(spin.value()) for k, spin in self._percent.items() if spin.value() > 0}

    def _sample_ids(self) -> list[str] | None:
        scope = self._scope.currentData()
        if scope == "unsplit":
            return [s.id for s in self._ds.samples if not s.split]
        if scope == "selected":
            return self._selected
        return None

    def _update_preview(self) -> None:
        ratios = self._ratios()
        total = sum(ratios.values())
        ids = self._sample_ids()
        n = len(self._ds.samples) if ids is None else len(ids)
        if not ratios:
            self._preview.setText("⚠ 请至少为一个子集设置大于 0 的比例。")
            return
        counts = allocate_counts(n, ratios)
        warn = "" if total == 100 else f"（比例合计 {total:.0f}%，将按比例归一化）"
        self._preview.setText("预计: " + " · ".join(
            f"{split_label(k)} {counts.get(k, 0)}" for k in SPLITS) + warn)

    def _accept(self) -> None:
        if not self._ratios():
            QMessageBox.warning(self, "随机划分", "请至少为一个子集设置大于 0 的比例。")
            return
        ids = self._sample_ids()
        if ids is not None and not ids:
            QMessageBox.information(self, "随机划分", "所选范围内没有样本。")
            return
        self.accept()

    def tool_inputs(self) -> dict:
        inputs = {"mode": "random", "ratios": self._ratios(), "seed": self._seed.value()}
        ids = self._sample_ids()
        if ids is not None:
            inputs["sample_ids"] = ids
        return inputs
