# -*- coding: utf-8 -*-
"""
安装依赖。若为 Python 3.14 会提示：stringzilla 无 3.14 预编译包，建议使用 Python 3.10–3.12。
"""
import subprocess
import sys
from pathlib import Path

# 仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "requirements.txt"


def main():
    py_version = sys.version_info
    if py_version.major == 3 and py_version.minor >= 14:
        print("当前为 Python 3.14。部分依赖（如 stringzilla）暂无 3.14 预编译包，")
        print("在 Windows 上会尝试从源码编译并需要 Microsoft C++ Build Tools。")
        print("建议使用 Python 3.10、3.11 或 3.12 创建虚拟环境后再安装：")
        print("  py -3.12 -m venv .venv   # 或从 python.org 安装 3.12")
        print("  .venv\\Scripts\\activate")
        print("  在仓库根目录执行: pip install -r requirements.txt")
        print()
        try:
            s = input("是否仍继续安装？(y/N): ").strip().lower()
            if s != "y" and s != "yes":
                sys.exit(0)
        except EOFError:
            sys.exit(0)

    ret = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        cwd=str(REPO_ROOT),
    )
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
