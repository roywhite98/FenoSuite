# -*- coding: utf-8 -*-
"""从仓库根目录启动植物表型分析 GUI。"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    from plant_phenotype.main import main
    main()
