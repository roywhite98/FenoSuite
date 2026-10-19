# -*- coding: utf-8 -*-
"""封装批量预测与单张预览。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable, List, Tuple

import numpy as np

from plant_phenotype.image_io import load_image_uint8_rgb

from plant_phenotype.segmentation.predict import (
    run_batch_prediction as _run_batch_prediction,
    ModelHolder,
    run_prediction,
)


def _load_image_rgb_uint8(path: str) -> np.ndarray:
    return load_image_uint8_rgb(path)


def overlay_mask_on_image(
    image_rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.45
) -> np.ndarray:
    """将二值 mask 以半透明颜色叠加到原图上。mask 为 0/255，返回 RGB 图。"""
    out = image_rgb.astype(np.float32)
    m = (mask > 0).astype(np.float32)[:, :, np.newaxis]
    c = np.array(color, dtype=np.float32)
    out = out * (1 - alpha * m) + (c * alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def run_single_preview(
    config_path: str,
    checkpoint_path: str,
    image_path: str,
    no_cuda: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """单张图像预测，用于预览。返回 (原图 RGB HWC, 二值 mask 0/255)。"""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    model, cfg = ModelHolder.get(checkpoint_path, device)
    predict_cfg = cfg.get("predict", {})
    threshold = predict_cfg.get("threshold", 0.5)
    remove_small = predict_cfg.get("remove_small_objects", 0)
    train_cfg = cfg.get("train", {})
    raw_ps = train_cfg.get("patch_size", [512, 512])
    if isinstance(raw_ps, (list, tuple)) and len(raw_ps) >= 2:
        patch_size = (max(1, int(raw_ps[0])), max(1, int(raw_ps[1])))
    elif isinstance(raw_ps, (int, float)):
        p = max(1, int(raw_ps))
        patch_size = (p, p)
    else:
        patch_size = (512, 512)
    mask = run_prediction(model, image_path, device, patch_size, threshold, remove_small)
    img_rgb = _load_image_rgb_uint8(image_path)
    return img_rgb, mask


def run_batch_prediction(
    config_path: str,
    checkpoint_path: str,
    image_dir: str,
    output_dir: str,
    segment_task_root: Optional[Path] = None,
    no_cuda: bool = False,
    line_callback: Optional[Callable[[str], None]] = None,
    image_files: Optional[List[str]] = None,
) -> tuple[int, List[str]]:
    """
    对 image_dir 下所有图片（或 image_files 指定的文件）做语义分割，结果写入 output_dir/masks/。
    在当前进程中调用预测逻辑。
    返回 (成功数, 失败文件名列表)。
    """
    # segment_task_root 已废弃，保留参数兼容
    return _run_batch_prediction(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        image_dir=image_dir,
        output_dir=output_dir,
        no_cuda=no_cuda,
        line_callback=line_callback,
        image_files=image_files,
    )
