"""
训练可视化与发表级统计图生成
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 关闭 matplotlib 字体查找提示，避免无 Arial 时的 findfont 警告
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# 设置发表级图表样式（使用 matplotlib 自带的 DejaVu Sans，避免 Linux 无 Arial 报错）
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Training Progress",
) -> None:
    """
    绘制训练/验证损失与指标曲线（发表级别）。

    Args:
        history: 字典，包含 'train_loss', 'val_loss', 'val_dice', 'val_iou', 'lr' 等列表
        save_path: 输出图片路径
        title: 图表标题
    """
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 损失曲线
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2, color="#2E86AB")
    ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2, color="#A23B72")
    ax.set_xlabel("Epoch", fontweight='bold')
    ax.set_ylabel("Loss", fontweight='bold')
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_title("Loss Curves", fontweight='bold')

    # Dice 系数
    ax = axes[0, 1]
    ax.plot(epochs, history["val_dice"], label="Val Dice", linewidth=2, color="#F18F01", marker='o', markersize=3)
    ax.set_xlabel("Epoch", fontweight='bold')
    ax.set_ylabel("Dice Coefficient", fontweight='bold')
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_title("Dice Coefficient", fontweight='bold')
    ax.set_ylim([0, 1])

    # IoU / mIoU
    ax = axes[1, 0]
    ax.plot(epochs, history["val_iou"], label="Val IoU (mIoU)", linewidth=2, color="#6A994E", marker='s', markersize=3)
    ax.set_xlabel("Epoch", fontweight='bold')
    ax.set_ylabel("IoU", fontweight='bold')
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_title("Mean IoU (mIoU)", fontweight='bold')
    ax.set_ylim([0, 1])

    # 学习率
    ax = axes[1, 1]
    if "lr" in history and history["lr"]:
        ax.plot(epochs, history["lr"], label="Learning Rate", linewidth=2, color="#BC4B51")
        ax.set_xlabel("Epoch", fontweight='bold')
        ax.set_ylabel("Learning Rate", fontweight='bold')
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_title("Learning Rate Schedule", fontweight='bold')
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"训练曲线已保存: {save_path}")


def plot_confusion_style_metrics(
    val_dice: float,
    val_iou: float,
    save_path: str,
) -> None:
    """绘制最佳验证指标条形图（Dice / mIoU 均为训练过程中最佳值，用于论文展示）。"""
    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = ["Dice", "IoU"]
    values = [val_dice, val_iou]
    colors = ["#F18F01", "#6A994E"]

    bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_ylim([0, 1])
    ax.set_ylabel("Score", fontweight='bold', fontsize=14)
    ax.set_title("Best Validation Metrics", fontweight='bold', fontsize=16)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"最终指标图已保存: {save_path}")


def save_training_log(
    history: Dict,
    cfg: Dict,
    save_dir: Path,
) -> None:
    """保存训练日志为 JSON（包含超参数和指标历史）。"""
    log = {
        "config": cfg,
        "history": history,
        "best_val_dice": max(history.get("val_dice", [0])),
        "best_val_iou": max(history.get("val_iou", [0])),
    }
    log_path = save_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"训练日志已保存: {log_path}")
