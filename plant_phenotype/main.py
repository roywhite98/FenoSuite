# -*- coding: utf-8 -*-
"""植物表型分析软件入口：启动统一 GUI。"""
import sys
import warnings
import logging

# 尽早屏蔽 albumentations 联网版本检查超时等无关警告（该库导入时会请求版本信息）
warnings.filterwarnings("ignore", message=".*Error fetching version.*")
warnings.filterwarnings("ignore", message=".*version info.*")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def main():
    try:
        from PyQt6.QtWidgets import QApplication
        from plant_phenotype.gui.main_window import MainWindow
        from plant_phenotype.gui.theme import apply_app_theme
    except ImportError as e:
        logging.error("导入失败，请安装依赖: pip install -r requirements.txt")
        logging.error("具体错误: %s", e)
        raise SystemExit(1) from e

    app = QApplication(sys.argv)
    app.setApplicationName("PlantPhenotype")
    app.setOrganizationName("PlantPhenotype")
    apply_app_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
