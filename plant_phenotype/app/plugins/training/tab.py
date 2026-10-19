# -*- coding: utf-8 -*-
"""模型训练 Tab：选择配置与数据来源（当前数据集或自定义目录），本地训练或提交到远程服务并显示日志。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from plant_phenotype.gui.utils import browse_directory, browse_file, match_width, show_yaml_editor_dialog
from plant_phenotype.training.facade import DEFAULT_REMOTE_URL, get_facade
from plant_phenotype.training.ssh_client import DEFAULT_REMOTE_DIR, DEFAULT_REMOTE_PYTHON

# 默认配置路径（包内）
DEFAULT_CONFIG = get_facade().default_config_path()

MODE_LOCAL = "本地训练"
MODE_SSH = "SSH 远程"
MODE_HTTP = "HTTP 服务"


class TrainingTab(QWidget):
    """模型训练页：本地训练或远程服务，配置路径、数据来源、保存目录，启动训练并显示日志。"""

    def __init__(self, parent=None, platform_app=None):
        super().__init__(parent)
        self._platform_app = platform_app
        self._task_id: str | None = None
        self._task_is_check = False
        self._task_logs_shown = 0
        self.setProperty("mdDensity", "compact")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(8)

        settings = QSettings()
        mode_grp = QGroupBox("训练方式")
        mode_ly = QHBoxLayout(mode_grp)
        mode_ly.setContentsMargins(8, 8, 8, 8)
        mode_ly.setSpacing(8)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([MODE_LOCAL, MODE_SSH, MODE_HTTP])
        self._mode_combo.setCurrentText(str(settings.value("training/mode", MODE_LOCAL)))
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_ly.addWidget(QLabel("方式:"))
        mode_ly.addWidget(self._mode_combo)
        self._remote_url_label = QLabel("服务地址:")
        mode_ly.addWidget(self._remote_url_label)
        self._remote_url_edit = QLineEdit()
        self._remote_url_edit.setPlaceholderText("http://localhost:8765")
        self._remote_url_edit.setText(DEFAULT_REMOTE_URL)
        mode_ly.addWidget(self._remote_url_edit, 1)
        self._use_cpu_check = QCheckBox("仅使用 CPU（不占用 GPU）")
        self._use_cpu_check.setChecked(False)
        self._use_cpu_check.setToolTip("本地训练时生效，勾选后强制使用 CPU")
        mode_ly.addWidget(self._use_cpu_check)
        mode_ly.addStretch(0)
        layout.addWidget(mode_grp)

        self._ssh_grp = QGroupBox("SSH 服务器")
        self._ssh_grp.setToolTip("服务器只需装好 PyTorch；训练代码与数据在每次训练时自动上传，结束后产物下载到保存目录")
        ssh_grid = QGridLayout(self._ssh_grp)
        ssh_grid.setContentsMargins(8, 8, 8, 8)
        ssh_grid.setVerticalSpacing(6)
        ssh_grid.setHorizontalSpacing(8)
        ssh_grid.addWidget(QLabel("连接:"), 0, 0)
        self._ssh_addr_edit = QLineEdit(str(settings.value("training/ssh_address", "")))
        self._ssh_addr_edit.setPlaceholderText("如 ssh -p 2222 root@192.168.1.100，或只填 IP（端口 22、用户 root）")
        self._ssh_addr_edit.setToolTip(
            "可直接粘贴云平台给出的 ssh 登录指令；也支持 root@IP:端口、ssh://root@IP:端口，"
            "或只填 IP / 主机名（默认端口 22、用户 root）"
        )
        ssh_grid.addWidget(self._ssh_addr_edit, 0, 1, 1, 2)
        ssh_grid.addWidget(QLabel("密码:"), 0, 3)
        self._ssh_pwd_edit = QLineEdit()
        self._ssh_pwd_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._ssh_pwd_edit.setPlaceholderText("使用私钥时留空；不会保存")
        ssh_grid.addWidget(self._ssh_pwd_edit, 0, 4, 1, 2)

        ssh_grid.addWidget(QLabel("私钥:"), 1, 0)
        self._ssh_key_edit = QLineEdit(str(settings.value("training/ssh_key", "")))
        self._ssh_key_edit.setPlaceholderText("可选，如 ~/.ssh/id_rsa")
        ssh_grid.addWidget(self._ssh_key_edit, 1, 1)
        btn_key = QPushButton("浏览...")
        btn_key.clicked.connect(self._browse_ssh_key)
        ssh_grid.addWidget(btn_key, 1, 2)
        ssh_grid.addWidget(QLabel("远端 Python:"), 1, 3)
        self._ssh_python_edit = QLineEdit(str(settings.value("training/ssh_python", DEFAULT_REMOTE_PYTHON)))
        self._ssh_python_edit.setToolTip("已装好 PyTorch 的解释器，如 python 或 /root/miniconda3/envs/xxx/bin/python")
        ssh_grid.addWidget(self._ssh_python_edit, 1, 4, 1, 2)

        ssh_grid.addWidget(QLabel("远端目录:"), 2, 0)
        self._ssh_dir_edit = QLineEdit(str(settings.value("training/ssh_remote_dir", DEFAULT_REMOTE_DIR)))
        self._ssh_dir_edit.setToolTip("每次训练在其下的 jobs/ 中新建任务目录；AutoDL/SeetaCloud 建议放到数据盘，如 /root/autodl-tmp/fenosuite")
        ssh_grid.addWidget(self._ssh_dir_edit, 2, 1, 1, 2)
        self._ssh_install_check = QCheckBox("自动安装缺少的依赖")
        self._ssh_install_check.setToolTip("用远端 pip 安装缺少的训练依赖（不包括 PyTorch）")
        self._ssh_install_check.setChecked(settings.value("training/ssh_auto_install", True, type=bool))
        ssh_grid.addWidget(self._ssh_install_check, 2, 3, 1, 2)
        self._ssh_test_btn = QPushButton("测试连接")
        self._ssh_test_btn.clicked.connect(self._test_ssh)
        ssh_grid.addWidget(self._ssh_test_btn, 2, 5)
        ssh_grid.setColumnStretch(1, 3)
        ssh_grid.setColumnStretch(4, 2)
        layout.addWidget(self._ssh_grp)

        grp = QGroupBox("路径")
        grp.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        grid = QGridLayout(grp)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setVerticalSpacing(6)
        grid.setHorizontalSpacing(8)
        grid.addWidget(QLabel("配置文件:"), 0, 0)
        self._config_edit = QLineEdit()
        self._config_edit.setPlaceholderText("YAML 配置路径")
        if DEFAULT_CONFIG.is_file():
            self._config_edit.setText(str(DEFAULT_CONFIG))
        grid.addWidget(self._config_edit, 0, 1)
        btn_cfg = QPushButton("浏览...")
        btn_cfg.clicked.connect(self._browse_config)
        grid.addWidget(btn_cfg, 0, 2)
        btn_preview_yaml = QPushButton("预览 YAML")
        btn_preview_yaml.clicked.connect(self._preview_yaml)
        grid.addWidget(btn_preview_yaml, 0, 3)

        grid.addWidget(QLabel("数据来源:"), 1, 0)
        self._data_edit = QLineEdit()
        if self._dataset_service() is not None:
            self._data_edit.setPlaceholderText("留空使用当前数据集；或选择包含 images/ 与 masks/ 的目录")
            self._data_edit.setToolTip("当前数据集在「数据集」页或各页右侧面板中切换")
        else:
            self._data_edit.setPlaceholderText("包含 images/ 与 masks/ 的目录")
        grid.addWidget(self._data_edit, 1, 1)
        btn_data = QPushButton("浏览...")
        btn_data.clicked.connect(self._browse_data)
        grid.addWidget(btn_data, 1, 2)

        grid.addWidget(QLabel("保存目录:"), 2, 0)
        self._save_edit = QLineEdit()
        self._save_edit.setPlaceholderText("可选，默认 workspace/runs/training")
        grid.addWidget(self._save_edit, 2, 1)
        btn_save = QPushButton("浏览...")
        btn_save.clicked.connect(self._browse_save)
        grid.addWidget(btn_save, 2, 2)

        grid.addWidget(QLabel("继续训练权重:"), 3, 0)
        self._resume_edit = QLineEdit()
        self._resume_edit.setPlaceholderText("可选，选择已有 .pth 从该参数继续训练")
        grid.addWidget(self._resume_edit, 3, 1)
        btn_resume = QPushButton("浏览...")
        btn_resume.clicked.connect(self._browse_resume)
        grid.addWidget(btn_resume, 3, 2)

        layout.addWidget(grp)

        btn_layout = QHBoxLayout()
        self._start_btn = QPushButton("开始训练")
        self._start_btn.setProperty("mdRole", "filled")
        self._start_btn.clicked.connect(self._start_training)
        btn_layout.addWidget(self._start_btn)
        self._stop_btn = QPushButton("停止")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_task)
        btn_layout.addWidget(self._stop_btn)
        btn_layout.addStretch(1)
        layout.addLayout(btn_layout)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("训练日志将在此显示…")
        layout.addWidget(self._log)
        match_width([btn_cfg, btn_preview_yaml, btn_data, btn_save, btn_resume, btn_key, self._stop_btn], self._start_btn)

        self._on_mode_changed(self._mode_combo.currentText())
        if platform_app is not None and platform_app.tasks is not None:
            platform_app.tasks.task_state_changed.connect(self._on_task_state_changed)

    def _dataset_service(self):
        return getattr(self._platform_app, "datasets", None)

    def _use_dataset(self) -> bool:
        """数据来源留空时使用当前数据集，填写目录时使用自定义目录。"""
        return not self._data_edit.text().strip() and self._dataset_service() is not None

    def _resolve_dataset_root(self) -> Optional[str]:
        """校验当前数据集可用于二值训练，返回其目录；不可用时提示并返回 None。"""
        from plant_phenotype.datasets.store import DatasetError
        from plant_phenotype.datasets.training_source import binary_training_splits, describe_splits

        try:
            ds = self._dataset_service().current()
        except Exception as e:
            QMessageBox.warning(self, "训练", f"数据集加载失败: {e}")
            return None
        if ds is None:
            QMessageBox.warning(self, "训练", "尚未选择数据集。请在「数据集」页导入或选择数据集，或在「数据来源」中选择一个目录。")
            return None
        try:
            groups = binary_training_splits(ds.path)
        except DatasetError as e:
            QMessageBox.warning(self, "训练", f"数据集「{ds.name}」不能用于当前训练:\n{e}")
            return None
        if not (groups["train"] or groups[""]):
            QMessageBox.warning(self, "训练", f"数据集「{ds.name}」中没有训练集或未划分的已标注样本"
                                              "（验证集、测试集样本不用于训练）。")
            return None
        n_pairs = sum(len(v) for v in groups.values())
        self._log.appendPlainText(
            f"使用数据集「{ds.name}」：{n_pairs} 对已标注样本，{describe_splits(groups)} ({ds.path})"
        )
        return str(ds.path)

    def _on_mode_changed(self, text: str) -> None:
        is_http = text == MODE_HTTP
        self._remote_url_label.setVisible(is_http)
        self._remote_url_edit.setVisible(is_http)
        self._ssh_grp.setVisible(text == MODE_SSH)
        self._use_cpu_check.setVisible(text == MODE_LOCAL)
        self._resume_edit.setEnabled(not is_http)

    def _browse_ssh_key(self) -> None:
        path = browse_file(self, "选择 SSH 私钥", "last_dir/ssh_key", "All (*)", str(Path.home() / ".ssh"))
        if path:
            self._ssh_key_edit.setText(path)

    def _ssh_inputs(self) -> Optional[dict]:
        """SSH 表单输入；地址为空时提示并返回 None。保存除密码外的设置。"""
        address = self._ssh_addr_edit.text().strip()
        if not address:
            QMessageBox.warning(self, "训练", "请填写 SSH 连接，例如: ssh -p 2222 root@192.168.1.100，或只填服务器 IP")
            return None
        inputs = {
            "ssh_address": address,
            "password": self._ssh_pwd_edit.text(),
            "key_path": self._ssh_key_edit.text().strip(),
            "python": self._ssh_python_edit.text().strip() or DEFAULT_REMOTE_PYTHON,
            "remote_dir": self._ssh_dir_edit.text().strip() or DEFAULT_REMOTE_DIR,
            "auto_install": self._ssh_install_check.isChecked(),
        }
        settings = QSettings()
        settings.setValue("training/ssh_address", address)
        settings.setValue("training/ssh_key", inputs["key_path"])
        settings.setValue("training/ssh_python", inputs["python"])
        settings.setValue("training/ssh_remote_dir", inputs["remote_dir"])
        settings.setValue("training/ssh_auto_install", inputs["auto_install"])
        return inputs

    def _run_task(self, tool_id: str, inputs: dict, *, is_check: bool = False) -> None:
        if self._platform_app is None or self._platform_app.tasks is None:
            self._log.appendPlainText("平台任务调度器不可用")
            return
        self._set_running(True)
        self._task_is_check = is_check
        self._task_logs_shown = 0
        self._task_id = self._platform_app.tasks.run_tool(tool_id, inputs)

    def _set_running(self, running: bool) -> None:
        self._start_btn.setEnabled(not running)
        self._ssh_test_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)

    def _stop_task(self) -> None:
        if self._task_id is not None and self._platform_app is not None and self._platform_app.tasks is not None:
            self._log.appendPlainText("正在停止…")
            self._stop_btn.setEnabled(False)
            self._platform_app.tasks.cancel(self._task_id)

    def _test_ssh(self) -> None:
        if self._task_id is not None:
            return
        inputs = self._ssh_inputs()
        if inputs is None:
            return
        self._log.clear()
        self._log.appendPlainText("正在连接并检查远端训练环境…")
        self._run_task("training.ssh_check", inputs, is_check=True)

    def _browse_config(self) -> None:
        path = browse_file(
            self,
            "选择配置文件",
            "last_dir/train_config",
            "YAML (*.yaml *.yml);;All (*)",
            str(DEFAULT_CONFIG.parent),
        )
        if path:
            self._config_edit.setText(path)

    def _preview_yaml(self) -> None:
        path = self._config_edit.text().strip() or str(DEFAULT_CONFIG)
        show_yaml_editor_dialog(self, path)

    def _browse_data(self) -> None:
        path = browse_directory(self, "选择数据来源目录", "last_dir/train_data")
        if path:
            self._data_edit.setText(path)

    def _browse_save(self) -> None:
        path = browse_directory(self, "选择保存目录", "last_dir/train_save")
        if path:
            self._save_edit.setText(path)

    def _browse_resume(self) -> None:
        path = browse_file(
            self,
            "选择继续训练权重（.pth）",
            "last_dir/train_resume",
            "PyTorch 权重 (*.pth);;All (*)",
        )
        if path:
            self._resume_edit.setText(path)

    def _start_training(self) -> None:
        if self._task_id is not None:
            return
        config_path = self._config_edit.text().strip()
        if not config_path:
            QMessageBox.warning(self, "训练", "请选择配置文件。")
            return
        if not Path(config_path).is_file():
            QMessageBox.warning(self, "训练", f"配置文件不存在: {config_path}")
            return
        self._log.clear()
        if self._use_dataset():
            data_root = self._resolve_dataset_root()
            if data_root is None:
                return
        else:
            data_root = self._data_edit.text().strip() or None
            if data_root and not Path(data_root).is_dir():
                QMessageBox.warning(self, "训练", f"数据来源目录不存在: {data_root}")
                return
        save_dir = self._save_edit.text().strip() or None
        mode = self._mode_combo.currentText()
        QSettings().setValue("training/mode", mode)

        if mode == MODE_HTTP:
            self._start_remote(config_path, data_root)
            return

        if save_dir and not Path(save_dir).is_dir():
            try:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.warning(self, "训练", f"无法创建保存目录: {e}")
                return

        resume_from = self._resume_edit.text().strip() or None
        if resume_from and not Path(resume_from).is_file():
            QMessageBox.warning(self, "训练", f"继续训练权重文件不存在: {resume_from}")
            return

        if mode == MODE_SSH:
            if not data_root:
                QMessageBox.warning(self, "训练", "SSH 远程训练需要数据来源（当前数据集或指定目录），数据将打包上传。")
                return
            inputs = self._ssh_inputs()
            if inputs is None:
                return
            inputs.update(config_path=config_path, data_root=data_root, save_dir=save_dir, resume_from=resume_from)
            self._run_task("training.ssh_train", inputs)
            return

        self._run_task(
            "training.local_train",
            {
                "config_path": config_path,
                "data_root": data_root,
                "save_dir": save_dir,
                "use_cpu": self._use_cpu_check.isChecked(),
                "resume_from": resume_from,
            },
        )

    def _start_remote(self, config_path: str, data_root: Optional[str]) -> None:
        base_url = self._remote_url_edit.text().strip()
        if not base_url:
            QMessageBox.warning(self, "训练", "请填写远程服务地址。")
            return
        if not data_root:
            QMessageBox.warning(self, "训练", "使用远程服务时请在「数据来源」中指定目录（将打包上传）。")
            return
        self._run_task(
            "training.remote_train",
            {
                "base_url": base_url,
                "config_path": config_path,
                "data_root": data_root,
            },
        )

    def _on_task_state_changed(self, state) -> None:
        if state.id != self._task_id:
            return
        for line in state.logs[self._task_logs_shown:]:
            self._append_log(line)
        self._task_logs_shown = len(state.logs)
        if state.status not in ("success", "failed", "cancelled"):
            return
        self._task_id = None
        self._set_running(False)
        if state.status == "cancelled":
            self._log.appendPlainText("已停止")
        elif state.status == "failed":
            self._on_finished_error(state.error or state.message)
        elif self._task_is_check:
            self._log.appendPlainText(f"=== {state.message} ===")
        else:
            metrics = state.result.metrics if state.result is not None else {}
            self._on_finished_ok(metrics.get("save_dir"))

    def _append_log(self, line: str) -> None:
        self._log.appendPlainText(line)

    def _on_finished_ok(self, save_path: Optional[str] = None) -> None:
        log_text = self._log.toPlainText()
        dice_str = "N/A"
        save_path = save_path or self._save_edit.text().strip() or "workspace/runs/training"
        for line in reversed(log_text.splitlines()):
            if "dice" in line.lower():
                m = re.search(r"dice[:\s=]+([0-9]+\.[0-9]+)", line, re.IGNORECASE)
                if m:
                    dice_str = m.group(1)
                break
        self._log.appendPlainText(
            f"\n=== 训练完成 | 最终 Dice: {dice_str} | 保存路径: {save_path} ==="
        )

    def _on_finished_error(self, msg: str) -> None:
        self._log.appendPlainText(f"错误: {msg}")
        QMessageBox.critical(self, "连接失败" if self._task_is_check else "训练失败", msg)

    def wait_workers(self) -> None:
        """关闭前等待后台线程结束。"""
        if self._task_id is not None and self._platform_app is not None and self._platform_app.tasks is not None:
            self._platform_app.tasks.wait_task(self._task_id)


def build_training_tab(parent=None, app=None) -> QWidget:
    """构建模型训练 Tab 控件（供插件注册调用）。"""
    return TrainingTab(parent, platform_app=app)
