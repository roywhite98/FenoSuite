# -*- coding: utf-8 -*-
"""训练读取接口：从数据集目录给出明确的 (原图, 掩码) 列表。"""
from __future__ import annotations

from pathlib import Path

from plant_phenotype.datasets.schema import DATASET_FILE, SPLITS
from plant_phenotype.datasets.store import DatasetError, DatasetProtectedError, DatasetStore

Pair = tuple[str, str]


def is_dataset_dir(path: str | Path) -> bool:
    return (Path(path) / DATASET_FILE).is_file()


def binary_training_splits(dataset_dir: str | Path, include_predicted: bool = False) -> dict[str, list[Pair]]:
    """按划分分组的二值语义训练配对：键为 train / val / test，``""`` 为未划分。

    多类别/实例数据集会被拒绝，避免类别被静默合并。
    """
    dataset_dir = Path(dataset_dir)
    ds = DatasetStore(dataset_dir.parent).load_from_path(dataset_dir)
    reason = ds.protection_reason()
    if reason:
        raise DatasetProtectedError(reason + " 多类别/实例训练需另行扩展。")
    statuses = {"annotated", "predicted"} if include_predicted else {"annotated"}
    groups: dict[str, list[Pair]] = {"": [], **{s: [] for s in SPLITS}}
    for s in ds.samples:
        if s.mask and s.status in statuses:
            img, mask = ds.abs(s.image), ds.abs(s.mask)
            if img.is_file() and mask.is_file():
                groups[s.split or ""].append((str(img), str(mask)))
    if not any(groups.values()):
        hint = "" if include_predicted else "（预测/候选状态的样本默认不参与训练）"
        raise DatasetError(f"数据集「{ds.name}」中没有可用于训练的已标注样本{hint}")
    return groups


def binary_training_pairs(dataset_dir: str | Path, include_predicted: bool = False) -> list[Pair]:
    """二值语义训练使用的全部配对（不区分划分）。"""
    groups = binary_training_splits(dataset_dir, include_predicted)
    return [p for k in ("", *SPLITS) for p in groups[k]]


def describe_splits(groups: dict[str, list[Pair]]) -> str:
    """训练时各划分的用法说明，如 “训练集 70 + 未划分 3 用于训练，验证集 20 用于验证，测试集 10 不参与训练”。"""
    train, extra, val, test = (len(groups.get(k, [])) for k in ("train", "", "val", "test"))
    if not (train or val or test):
        return f"未划分 {extra} 对，训练时按比例随机划出验证集"
    used = f"训练集 {train}" + (f" + 未划分 {extra}" if extra else "") + " 用于训练"
    used += f"，验证集 {val} 用于验证" if val else "，无验证集，从训练样本中按比例随机划出"
    if test:
        used += f"，测试集 {test} 不参与训练"
    return used
