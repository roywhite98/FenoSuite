# -*- coding: utf-8 -*-
"""图像分割 Tab：选择权重与图像目录，进程内批量预测；支持单张预览（原图+mask 透明叠加）、←→ 切换图片。"""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QKeyEvent, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QMessageBox,
    QGroupBox,
    QGridLayout,
    QCheckBox,
    QSplitter,
    QSlider,
    QStackedWidget,
    QComboBox,
    QInputDialog,
    QSizePolicy,
)

from plant_phenotype.gui.utils import show_yaml_editor_dialog, browse_directory, browse_file, match_width
from plant_phenotype.image_io import DEFAULT_IMAGE_EXTENSIONS
from plant_phenotype.gui.segmentation_preview import PreviewCanvasView, numpy_rgb_to_qimage
from plant_phenotype.segmentation.facade import get_facade

PKG_ROOT = Path(__file__).resolve().parents[3]


class SegmentationTab(QWidget):
    """图像分割页：配置路径、图像目录、输出目录，启动批量预测并显示日志。"""

    def __init__(self, parent=None, platform_app=None):
        super().__init__(parent)
        self._platform_app = platform_app
        self._task_id: str | None = None
        self._task_kind: str | None = None
        self._task_logs_shown = 0
        self._preview_image_rgb: np.ndarray | None = None
        self._preview_mask: np.ndarray | None = None
        self._preview_image_list: list = []
        self._preview_index: int = -1
        self._preview_mask_alpha: float = 0.45
        self._preview_show_mask: bool = True
        self.setProperty("mdDensity", "compact")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 12)

        grp = QGroupBox("路径")
        grp.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        grid = QGridLayout(grp)
        grid.setVerticalSpacing(6)
        grid.setHorizontalSpacing(8)
        grid.setContentsMargins(8, 8, 8, 8)
        default_config = get_facade().default_config_path()
        grid.addWidget(QLabel("配置文件:"), 0, 0)
        self._config_edit = QLineEdit()
        self._config_edit.setPlaceholderText("用于解析相对 checkpoint 路径，可选")
        if default_config.is_file():
            self._config_edit.setText(str(default_config))
        grid.addWidget(self._config_edit, 0, 1)
        btn_cfg = QPushButton("浏览...")
        btn_cfg.clicked.connect(self._browse_config)
        grid.addWidget(btn_cfg, 0, 2)
        btn_preview_yaml = QPushButton("预览 YAML")
        btn_preview_yaml.clicked.connect(self._preview_yaml)
        grid.addWidget(btn_preview_yaml, 0, 3)

        grid.addWidget(QLabel("权重文件:"), 1, 0)
        self._ckpt_edit = QLineEdit()
        self._ckpt_edit.setPlaceholderText("best_model.pth 等")
        if platform_app is not None:
            checkpoint = platform_app.context.get_last_checkpoint_path()
            if checkpoint:
                self._ckpt_edit.setText(checkpoint)
        grid.addWidget(self._ckpt_edit, 1, 1)
        btn_ckpt = QPushButton("浏览...")
        btn_ckpt.clicked.connect(self._browse_checkpoint)
        grid.addWidget(btn_ckpt, 1, 2)

        grid.addWidget(QLabel("图像来源:"), 2, 0)
        self._source_combo = QComboBox()
        self._source_combo.addItem("当前数据集", "dataset")
        self._source_combo.addItem("自定义目录", "dir")
        grid.addWidget(self._source_combo, 2, 1)

        grid.addWidget(QLabel("图像目录:"), 3, 0)
        self._image_edit = QLineEdit()
        self._image_edit.setPlaceholderText("包含待分割图片的目录")
        grid.addWidget(self._image_edit, 3, 1)
        self._btn_img = QPushButton("浏览...")
        self._btn_img.clicked.connect(self._browse_image_dir)
        grid.addWidget(self._btn_img, 3, 2)

        grid.addWidget(QLabel("输出目录:"), 4, 0)
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("结果将写入 输出目录/masks/；使用数据集时可留空")
        grid.addWidget(self._output_edit, 4, 1)
        btn_out = QPushButton("浏览...")
        btn_out.clicked.connect(self._browse_output)
        grid.addWidget(btn_out, 4, 2)

        layout.addWidget(grp)

        cpu_row = QHBoxLayout()
        self._no_cuda_check = QCheckBox("仅使用 CPU（不占用 GPU）")
        cpu_row.addWidget(self._no_cuda_check)
        cpu_row.addWidget(QLabel("结果入库:"))
        self._store_combo = QComboBox()
        self._store_combo.addItem("不入库（仅写输出目录）", "none")
        self._store_combo.addItem("写回当前数据集（仅未标注/预测样本）", "current")
        self._store_combo.addItem("另存为新数据集", "new")
        self._store_combo.setToolTip(
            "写回：结果标记为「预测/候选」，人工标注的样本不会被覆盖，预测结果默认不参与训练；\n"
            "另存：用原图与结果新建「背景/根」数据集，便于复核"
        )
        cpu_row.addWidget(self._store_combo)
        self._start_btn = QPushButton("开始分割")
        self._start_btn.setProperty("mdRole", "filled")
        self._start_btn.clicked.connect(self._start_prediction)
        cpu_row.addWidget(self._start_btn)
        cpu_row.addStretch()
        layout.addLayout(cpu_row)

        layout.setSpacing(8)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        preview_grp = QGroupBox("单张预览（← → 键切换图片）")
        preview_layout = QVBoxLayout(preview_grp)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(6)
        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel("预览图像:"))
        self._preview_image_edit = QLineEdit()
        self._preview_image_edit.setPlaceholderText("选择一张图像进行分割预览")
        preview_row.addWidget(self._preview_image_edit)
        btn_preview_browse = QPushButton("浏览...")
        btn_preview_browse.clicked.connect(self._browse_preview_image)
        preview_row.addWidget(btn_preview_browse)
        self._preview_btn = QPushButton("运行预览")
        self._preview_btn.setProperty("mdRole", "tonal")
        self._preview_btn.clicked.connect(self._run_preview)
        preview_row.addWidget(self._preview_btn)
        preview_layout.addLayout(preview_row)
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Mask 透明度:"))
        self._preview_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._preview_alpha_slider.setRange(0, 100)
        self._preview_alpha_slider.setValue(45)
        self._preview_alpha_slider.valueChanged.connect(self._on_preview_alpha_changed)
        ctrl_row.addWidget(self._preview_alpha_slider, 1)
        self._preview_alpha_label = QLabel("45%")
        ctrl_row.addWidget(self._preview_alpha_label)
        self._preview_show_mask_check = QCheckBox("显示 Mask")
        self._preview_show_mask_check.setChecked(True)
        self._preview_show_mask_check.toggled.connect(self._on_preview_show_mask_toggled)
        ctrl_row.addWidget(self._preview_show_mask_check)
        preview_layout.addLayout(ctrl_row)
        self._preview_info_label = QLabel("")
        self._preview_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_info_label.setProperty("mdRole", "caption")
        preview_layout.addWidget(self._preview_info_label)
        self._preview_canvas = PreviewCanvasView()
        self._preview_canvas.setObjectName("previewCanvas")
        self._preview_canvas.setMinimumSize(280, 200)
        self._preview_placeholder = QLabel()
        self._preview_placeholder.setObjectName("previewPlaceholder")
        self._preview_placeholder.setMinimumSize(280, 200)
        self._preview_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_placeholder.setText("预览结果将显示在此。← → 键切换图片；滚轮缩放画布。")
        self._preview_stack = QStackedWidget()
        self._preview_stack.addWidget(self._preview_canvas)
        self._preview_stack.addWidget(self._preview_placeholder)
        self._preview_stack.setCurrentWidget(self._preview_placeholder)
        preview_layout.addWidget(self._preview_stack, 1)
        main_splitter.addWidget(preview_grp)
        # 用 QShortcut 保证在 Tab 内任意控件有焦点时 ← → 都能切换预览
        self._shortcut_prev = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self._shortcut_prev.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._shortcut_prev.activated.connect(lambda: self._preview_go(-1))
        self._shortcut_next = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self._shortcut_next.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._shortcut_next.activated.connect(lambda: self._preview_go(1))

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("分割日志将在此显示…")
        right_layout.addWidget(self._log)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([420, 380])
        layout.addWidget(main_splitter)
        match_width(
            [btn_cfg, btn_preview_yaml, btn_ckpt, self._btn_img, btn_out, btn_preview_browse, self._preview_btn],
            self._start_btn,
        )
        if platform_app is not None:
            platform_app.tasks.task_state_changed.connect(self._on_task_state_changed)

        self._pending_store: dict | None = None
        service = self._dataset_service()
        if service is None:
            self._source_combo.setCurrentIndex(self._source_combo.findData("dir"))
            self._source_combo.setEnabled(False)
            self._store_combo.setEnabled(False)
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        self._on_source_changed()

    def _dataset_service(self):
        return getattr(self._platform_app, "datasets", None)

    def _use_dataset(self) -> bool:
        return self._source_combo.currentData() == "dataset"

    def _on_source_changed(self, *_args) -> None:
        use_ds = self._use_dataset()
        self._image_edit.setEnabled(not use_ds)
        self._btn_img.setEnabled(not use_ds)
        item = self._store_combo.model().item(self._store_combo.findData("current"))
        item.setEnabled(use_ds)
        if not use_ds and self._store_combo.currentData() == "current":
            self._store_combo.setCurrentIndex(self._store_combo.findData("none"))

    def _browse_config(self) -> None:
        path = browse_file(self, "选择配置文件", "last_dir/seg_config",
                           "YAML (*.yaml *.yml);;All (*)", str(PKG_ROOT))
        if path:
            self._config_edit.setText(path)

    def _preview_yaml(self) -> None:
        path = self._config_edit.text().strip() or str(get_facade().default_config_path())
        show_yaml_editor_dialog(self, path)

    def _browse_checkpoint(self) -> None:
        path = browse_file(self, "选择权重文件", "last_dir/seg_ckpt",
                           "PyTorch (*.pth *.pt);;All (*)", str(PKG_ROOT))
        if path:
            self._ckpt_edit.setText(path)

    def _browse_image_dir(self) -> None:
        path = browse_directory(self, "选择图像目录", "last_dir/seg_image")
        if path:
            self._image_edit.setText(path)

    def _browse_output(self) -> None:
        path = browse_directory(self, "选择输出目录", "last_dir/seg_output")
        if path:
            self._output_edit.setText(path)

    def _browse_preview_image(self) -> None:
        path = browse_file(self, "选择预览图像", "last_dir/seg_preview_image",
                           "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All (*)")
        if path:
            self._preview_image_edit.setText(path)

    def _run_preview(self) -> None:
        if self._task_id is not None:
            return
        config_path = self._config_edit.text().strip() or str(get_facade().default_config_path())
        checkpoint_path = self._ckpt_edit.text().strip()
        image_path = self._preview_image_edit.text().strip()
        if not checkpoint_path or not Path(checkpoint_path).is_file():
            QMessageBox.warning(self, "预览", "请先选择有效的权重文件。")
            return
        if not Path(config_path).is_file():
            QMessageBox.warning(self, "预览", "配置文件不存在。")
            return
        if not image_path or not Path(image_path).is_file():
            QMessageBox.warning(self, "预览", "请选择一张要预览的图像。")
            return
        self._preview_btn.setEnabled(False)
        self._start_btn.setEnabled(False)
        self._preview_placeholder.setText("正在预测…")
        self._preview_stack.setCurrentWidget(self._preview_placeholder)
        self._task_kind = "preview"
        self._task_logs_shown = 0
        self._task_id = self._platform_app.tasks.run_tool(
            "segmentation.preview",
            {
                "config_path": config_path,
                "checkpoint_path": checkpoint_path,
                "image_path": image_path,
                "no_cuda": self._no_cuda_check.isChecked(),
            },
        )

    def _on_preview_ok(self, image_rgb: np.ndarray, mask: np.ndarray) -> None:
        self._preview_btn.setEnabled(True)
        self._start_btn.setEnabled(True)
        self._preview_image_rgb = image_rgb
        self._preview_mask = mask
        path = self._preview_image_edit.text().strip()
        if path:
            p = Path(path)
            if p.is_file() and p.parent.is_dir():
                # 仅同级目录内图像，← → 切换不跨级、不换路径
                exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
                self._preview_image_list = sorted(
                    [f for f in p.parent.iterdir() if f.is_file() and f.suffix.lower() in exts],
                    key=lambda x: x.name,
                )
                try:
                    self._preview_index = self._preview_image_list.index(p)
                except ValueError:
                    self._preview_index = 0
            else:
                self._preview_image_list = []
                self._preview_index = -1
        else:
            self._preview_image_list = []
            self._preview_index = -1
        self._refresh_preview_display()
        self._refresh_preview_info()

    def _refresh_preview_info(self) -> None:
        if not self._preview_image_list or self._preview_index < 0:
            self._preview_info_label.setText("")
            return
        name = self._preview_image_list[self._preview_index].name
        total = len(self._preview_image_list)
        idx = self._preview_index + 1
        self._preview_info_label.setText(f"{name}  [{idx} / {total}]")

    def _refresh_preview_display(self) -> None:
        if self._preview_image_rgb is None or self._preview_mask is None:
            return
        if self._preview_show_mask:
            display = get_facade().overlay_mask(
                self._preview_image_rgb, self._preview_mask, color=(0, 255, 0), alpha=self._preview_mask_alpha
            )
        else:
            display = self._preview_image_rgb.copy()
        qimg = numpy_rgb_to_qimage(display)
        pix = QPixmap.fromImage(qimg)
        self._preview_canvas.set_pixmap(pix)
        self._preview_stack.setCurrentWidget(self._preview_canvas)

    def _on_preview_alpha_changed(self, value: int) -> None:
        self._preview_mask_alpha = value / 100.0
        self._preview_alpha_label.setText(f"{value}%")
        self._refresh_preview_display()

    def _on_preview_show_mask_toggled(self, checked: bool) -> None:
        self._preview_show_mask = checked
        self._refresh_preview_display()

    def _on_preview_error(self, msg: str) -> None:
        self._preview_btn.setEnabled(True)
        self._start_btn.setEnabled(True)
        self._preview_placeholder.setText(f"预览失败: {msg}")
        self._preview_stack.setCurrentWidget(self._preview_placeholder)
        QMessageBox.warning(self, "预览失败", msg)

    def _preview_go(self, delta: int) -> None:
        if not self._preview_image_list or self._task_kind == "preview":
            return
        idx = self._preview_index + delta
        if idx < 0 or idx >= len(self._preview_image_list):
            return
        self._preview_index = idx
        path = self._preview_image_list[self._preview_index]
        self._preview_image_edit.setText(str(path))
        self._refresh_preview_info()
        self._run_preview()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            delta = -1 if event.key() == Qt.Key.Key_Left else 1
            self._preview_go(delta)
            event.accept()
            return
        super().keyPressEvent(event)

    def _start_prediction(self) -> None:
        if self._task_id is not None:
            return
        checkpoint_path = self._ckpt_edit.text().strip()
        if not checkpoint_path:
            QMessageBox.warning(self, "分割", "请选择权重文件。")
            return
        if not Path(checkpoint_path).is_file():
            QMessageBox.warning(self, "分割", f"权重文件不存在: {checkpoint_path}")
            return
        config_path = self._config_edit.text().strip()
        if not config_path:
            config_path = str(get_facade().default_config_path())
        if not Path(config_path).is_file():
            QMessageBox.warning(self, "分割", f"配置文件不存在: {config_path}")
            return
        output_dir = self._output_edit.text().strip()
        store_mode = self._store_combo.currentData()
        source_ds = None
        image_files = None
        if self._use_dataset():
            try:
                source_ds = self._dataset_service().current()
            except Exception as e:
                QMessageBox.warning(self, "分割", f"数据集加载失败: {e}")
                return
            if source_ds is None:
                QMessageBox.warning(self, "分割", "尚未选择数据集，请在「数据集」页选择，或改用自定义目录。")
                return
            # 样本可能分布在 images/ 与 train|val|test/images/ 下，按样本列表逐个预测
            image_dir = str(source_ds.path)
            image_files = [str(p) for p in (source_ds.abs(s.image) for s in source_ds.samples) if p.is_file()]
            if not image_files:
                QMessageBox.warning(self, "分割", f"数据集「{source_ds.name}」中没有可分割的原图。")
                return
            if not output_dir:
                output_dir = str(
                    self._dataset_service().root / ".cache" / "predictions" / source_ds.id
                    / time.strftime("%Y%m%d-%H%M%S")
                )
        else:
            image_dir = self._image_edit.text().strip()
            if not image_dir:
                QMessageBox.warning(self, "分割", "请选择图像目录。")
                return
            if not Path(image_dir).is_dir():
                QMessageBox.warning(self, "分割", f"图像目录不存在: {image_dir}")
                return
            if not output_dir:
                QMessageBox.warning(self, "分割", "请选择输出目录。")
                return

        self._pending_store = None
        if store_mode == "current":
            from plant_phenotype.datasets.store import DatasetError

            try:
                source_ds.binary_foreground_class()
            except DatasetError as e:
                QMessageBox.warning(self, "分割", f"不能写回数据集「{source_ds.name}」:\n{e}\n\n可改为「另存为新数据集」。")
                return
            self._pending_store = {"target": "current", "dataset_id": source_ds.id}
        elif store_mode == "new":
            base = source_ds.name if source_ds is not None else Path(image_dir).name
            name, ok = QInputDialog.getText(
                self, "另存为新数据集", "新数据集名称:", text=f"{base} 预测 {time.strftime('%m%d-%H%M')}"
            )
            if not ok or not name.strip():
                return
            prov = {"source_dataset_id": source_ds.id, "source_name": source_ds.name} if source_ds else {}
            self._pending_store = {"target": "new", "name": name.strip(), "provenance": prov}
            if source_ds is not None and source_ds.has_splits:
                self._pending_store["splits"] = {str(source_ds.abs(s.image)): s.split for s in source_ds.samples}
        if self._pending_store is not None:
            self._pending_store.update({
                "image_dir": image_dir,
                "image_files": image_files,
                "mask_dir": str(Path(output_dir) / "masks"),
                "info": f"checkpoint={checkpoint_path}",
            })
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self._log.clear()
        self._start_btn.setEnabled(False)
        self._preview_btn.setEnabled(False)
        self._task_kind = "batch"
        self._task_logs_shown = 0
        self._task_id = self._platform_app.tasks.run_tool(
            "segmentation.batch_predict",
            {
                "config_path": config_path,
                "checkpoint_path": checkpoint_path,
                "image_dir": image_dir,
                **({"image_files": image_files} if image_files is not None else {}),
                "output_dir": output_dir,
                "no_cuda": self._no_cuda_check.isChecked(),
            },
        )

    def wait_workers(self) -> None:
        """关闭前等待所有后台线程结束，避免 QThread destroyed while still running。"""
        if self._task_id is not None:
            self._platform_app.tasks.wait_task(self._task_id)

    def _append_log(self, line: str) -> None:
        self._log.appendPlainText(line)

    def _on_finished_ok(self, success_count: int, failed_list: list) -> None:
        self._start_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)
        msg = f"完成：成功 {success_count} 张"
        if failed_list:
            msg += f"，失败 {len(failed_list)} 张"
        self._log.appendPlainText(msg)
        if failed_list:
            QMessageBox.warning(self, "分割", msg + "\n失败: " + ", ".join(failed_list[:10]) + (" …" if len(failed_list) > 10 else ""))
        pending, self._pending_store = self._pending_store, None
        if pending is not None and success_count > 0:
            self._start_store(pending, set(failed_list))

    def _start_store(self, pending: dict, failed: set) -> None:
        exts = {e.lower() for e in DEFAULT_IMAGE_EXTENSIONS}
        image_dir = Path(pending.pop("image_dir"))
        image_files = pending.pop("image_files", None)
        if image_files is not None:
            pending["images"] = [f for f in image_files if Path(f).name not in failed]
        else:
            pending["images"] = sorted(
                str(f) for f in image_dir.iterdir()
                if f.is_file() and f.suffix.lower() in exts and f.name not in failed
            )
        pending["datasets_root"] = str(self._dataset_service().root)
        self._log.appendPlainText("正在把分割结果写入数据集…")
        self._start_btn.setEnabled(False)
        self._preview_btn.setEnabled(False)
        self._task_kind = "store"
        self._task_logs_shown = 0
        self._task_id = self._platform_app.tasks.run_tool("datasets.import_predictions", pending)

    def _on_store_finished(self, result) -> None:
        self._start_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)
        self._log.appendPlainText(result.message)
        dataset_id = (result.metrics or {}).get("dataset_id")
        self._dataset_service().notify_changed(dataset_id)
        report = (result.metrics or {}).get("report") or {}
        failed = report.get("failed") or []
        text = result.message + "\n\n可在「数据集」页或各页右侧面板查看、复核（状态为「预测/候选」）。"
        if failed:
            text += "\n\n失败:\n" + "\n".join(f"{f.get('file')}: {f.get('error')}" for f in failed[:10])
        QMessageBox.information(self, "分割结果入库", text)

    def _on_finished_error(self, msg: str) -> None:
        self._log.appendPlainText(f"错误: {msg}")
        self._start_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)
        QMessageBox.critical(self, "分割失败", msg)

    def _on_task_state_changed(self, state) -> None:
        if state.tool_id == "training.local_train" and state.status == "success":
            checkpoint = self._platform_app.context.get_last_checkpoint_path()
            if checkpoint:
                self._ckpt_edit.setText(checkpoint)
        if state.id != self._task_id:
            return
        for line in state.logs[self._task_logs_shown:]:
            self._append_log(line)
        self._task_logs_shown = len(state.logs)
        if state.status not in ("success", "failed", "cancelled"):
            return
        kind = self._task_kind
        self._task_id = None
        self._task_kind = None
        if state.status != "success":
            message = state.error or state.message
            self._pending_store = None
            if kind == "preview":
                self._on_preview_error(message)
            else:
                self._on_finished_error(message)
            return
        if kind == "store":
            self._on_store_finished(state.result)
        elif kind == "preview":
            artifacts = {item.id: item.data for item in state.result.artifacts}
            self._on_preview_ok(artifacts["preview_rgb"], artifacts["preview_mask"])
        else:
            metrics = state.result.metrics
            self._on_finished_ok(metrics["success"], metrics["failed"])


def build_segmentation_tab(parent=None, app=None) -> QWidget:
    """构建图像分割 Tab 控件（供插件注册调用）。"""
    return SegmentationTab(parent, platform_app=app)
