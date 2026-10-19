# -*- coding: utf-8 -*-
"""启动远端训练服务。默认 http://0.0.0.0:8765（从仓库根目录运行）。"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    from plant_phenotype.server.app import main
    main()
