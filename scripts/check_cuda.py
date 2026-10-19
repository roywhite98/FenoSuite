# -*- coding: utf-8 -*-
"""CUDA 诊断：检查当前 PyTorch 是否支持 GPU，并给出安装建议。"""
from __future__ import annotations

import sys


def main() -> None:
    print("=" * 60)
    print("PyTorch / CUDA 诊断")
    print("=" * 60)
    try:
        import torch
    except ImportError:
        print("未安装 PyTorch。请先: pip install -r requirements.txt")
        print("若需 GPU 支持，请使用下方「安装带 CUDA 的 PyTorch」命令。")
        sys.exit(1)

    print(f"PyTorch 版本:     {torch.__version__}")
    cuda_built = getattr(torch.version, "cuda", None)
    print(f"PyTorch 编译 CUDA: {cuda_built if cuda_built else '无（当前为 CPU 版）'}")
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {available}")

    if available:
        print(f"GPU 数量:        {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        print("CUDA 可用，训练/分割将使用 GPU。")
    else:
        print()
        print("未检测到 CUDA。常见原因：")
        print("1. 安装的是 PyTorch CPU 版（pip install torch 默认可能是 CPU 版）")
        print("2. 本机 NVIDIA 驱动过旧，或未安装驱动")
        print("3. PyTorch 编译时的 CUDA 版本与驱动不匹配")
        print()
        print("解决步骤：")
        print("(1) 确认已安装 NVIDIA 显卡驱动（nvidia-smi 可查看版本与 GPU）")
        print("(2) 卸载当前 PyTorch 后安装带 CUDA 的版本，例如：")
        print("    pip uninstall torch torchvision -y")
        print("    # CUDA 12.1（多数新卡推荐）：")
        print('    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121')
        print("    # 或 CUDA 11.8：")
        print('    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118')
        print("    更多版本见: https://pytorch.org/get-started/locally")
        print("=" * 60)
        sys.exit(0)
    print("=" * 60)


if __name__ == "__main__":
    main()
