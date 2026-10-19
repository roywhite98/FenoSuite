# -*- coding: utf-8 -*-
"""批量预测：进程内调用，支持 run_batch_prediction 供 segmentation/runner 使用。"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Optional, Callable, List, Tuple

import cv2
import numpy as np
import torch

from plant_phenotype.image_io import DEFAULT_IMAGE_EXTENSIONS, load_image_uint8_rgb
from plant_phenotype.models.build_model import build_model_from_config
from plant_phenotype.models.outputs import segmentation_logits


def _load_image_uint8_rgb(path: str) -> np.ndarray:
    return load_image_uint8_rgb(path)


class ModelHolder:
    """进程内缓存已加载的 (model, cfg)，避免批量预测重复读盘。"""

    _cache: dict[tuple[str, str, int], tuple] = {}

    @classmethod
    def get(cls, checkpoint_path: str, device: torch.device):
        path = Path(checkpoint_path).resolve()
        device_key = str(device)
        key = (str(path), device_key, path.stat().st_mtime_ns)
        if key not in cls._cache:
            cls._cache = {k: v for k, v in cls._cache.items() if k[:2] != key[:2]}
            cls._cache[key] = load_model_and_config(checkpoint_path, device)
        return cls._cache[key]

    @classmethod
    def clear(cls) -> None:
        cls._cache.clear()


def load_model_and_config(checkpoint_path: str, device: torch.device):
    """从 checkpoint 加载模型和配置。"""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config")
    if cfg is None:
        raise ValueError("Checkpoint 中缺少 config，请使用本项目保存的 best_model.pth")
    model = build_model_from_config(cfg["model"], initialize_encoder=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    return model, cfg


def preprocess_image(image: np.ndarray, patch_size: tuple) -> tuple:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"图像尺寸无效 (h={h}, w={w})，无法预测。请检查图像文件是否损坏或为空。")
    orig_h, orig_w = h, w
    ph, pw = patch_size
    if ph <= 0 or pw <= 0:
        raise ValueError(f"patch_size 须为正数，当前为 {patch_size}。")
    new_h = ((h + ph - 1) // ph) * ph
    new_w = ((w + pw - 1) // pw) * pw
    if new_h <= 0 or new_w <= 0:
        raise ValueError(f"对齐后尺寸无效 (new_h={new_h}, new_w={new_w})，无法 resize。")
    if (new_h, new_w) != (h, w):
        dsize = (int(new_w), int(new_h))
        if dsize[0] <= 0 or dsize[1] <= 0:
            raise ValueError(f"preprocess_image: 无效 dsize {dsize} (new_w={new_w}, new_h={new_h})")
        try:
            image = cv2.resize(image, dsize, interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
            raise RuntimeError(f"preprocess_image resize 失败: dsize={dsize}, image.shape={image.shape}") from e
    x = image.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
    return x, (orig_h, orig_w), (new_h, new_w)


def postprocess(logits: np.ndarray, orig_size: tuple, pred_size: tuple, threshold: float = 0.5, remove_small: int = 0) -> np.ndarray:
    from skimage import morphology
    from skimage.measure import label
    probs = 1.0 / (1.0 + np.exp(-logits))
    mask = (probs[0] > threshold).astype(np.uint8)
    # 保证 2D：logits 可能为 (1,1,H,W)，probs[0] 为 (1,H,W)，cv2.resize 需要 (H,W)
    mask = np.squeeze(mask)
    if mask.ndim == 3:
        mask = mask[0] if mask.shape[0] == 1 else mask.reshape(mask.shape[-2], mask.shape[-1])
    if remove_small > 0:
        labeled = label(mask, connectivity=2)
        # max_size 表示移除「小于等于该值」的物体；与旧版 min_size（移除严格小于）等价用 max_size=remove_small-1
        mask = morphology.remove_small_objects(labeled.astype(bool), max_size=remove_small - 1, connectivity=2)
        mask = (mask > 0).astype(np.uint8)
    orig_h, orig_w = int(orig_size[0]), int(orig_size[1])
    pred_h, pred_w = pred_size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"原始尺寸无效 (orig_w={orig_w}, orig_h={orig_h})，无法 resize mask。")
    if mask.size == 0 or mask.shape[0] <= 0 or mask.shape[1] <= 0:
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    elif (pred_h, pred_w) != (orig_h, orig_w):
        dsize = (orig_w, orig_h)
        if dsize[0] <= 0 or dsize[1] <= 0:
            raise ValueError(f"postprocess: 无效 dsize {dsize}")
        try:
            mask = cv2.resize(mask, dsize, interpolation=cv2.INTER_NEAREST)
        except cv2.error as e:
            raise RuntimeError(
                f"postprocess resize 失败: dsize={dsize}, mask.shape={mask.shape}, orig_size={orig_size}"
            ) from e
    out = np.squeeze((mask * 255).astype(np.uint8))
    if out.ndim > 2:
        out = out.max(axis=-1)
    return np.ascontiguousarray(out)


def run_prediction(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    patch_size: tuple,
    threshold: float,
    remove_small: int,
) -> np.ndarray:
    """单张图像预测，返回二值 mask (H,W) 0/255。"""
    image = _load_image_uint8_rgb(image_path)
    if image.size == 0 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"图像为空或尺寸为 0，无法预测: {image_path}")
    x, orig_size, pred_size = preprocess_image(image, patch_size)
    x = x.to(device)
    with torch.inference_mode():
        logits = segmentation_logits(model(x))
    logits = logits.cpu().numpy()
    return postprocess(logits, orig_size, pred_size, threshold, remove_small)


def run_batch_prediction(
    config_path: str,
    checkpoint_path: str,
    image_dir: str,
    output_dir: str,
    no_cuda: bool = False,
    line_callback: Optional[Callable[[str], None]] = None,
    image_files: Optional[List[str]] = None,
) -> Tuple[int, List[str]]:
    """
    进程内批量预测。结果写入 output_dir/masks/。
    image_files 给出时只预测这些文件（可分布在不同目录，主名需唯一），忽略 image_dir 的内容。
    返回 (成功数, 失败文件名列表)。
    """
    def log(msg: str) -> None:
        if line_callback:
            line_callback(msg)
        else:
            print(msg)

    if os.path.isabs(checkpoint_path):
        ckpt_abs = checkpoint_path
    else:
        config_abspath = os.path.abspath(config_path)
        project_root = os.path.dirname(os.path.dirname(config_abspath))
        ckpt_abs = os.path.normpath(os.path.join(project_root, checkpoint_path))

    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    model, cfg = ModelHolder.get(ckpt_abs, device)
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

    output_dir_p = Path(output_dir)
    if image_files is not None:
        image_paths = [Path(f) for f in image_files]
    else:
        image_dir_p = Path(image_dir)
        if not image_dir_p.is_dir():
            raise FileNotFoundError(f"图像目录不存在: {image_dir}")
        image_exts = set(ext.lower() for ext in DEFAULT_IMAGE_EXTENSIONS)
        image_paths = sorted(f for f in image_dir_p.iterdir() if f.is_file() and f.suffix.lower() in image_exts)
    masks_dir = output_dir_p / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    if not image_paths:
        log(f"未在 {image_dir} 下找到图像文件")
        return 0, []

    log(f"共 {len(image_paths)} 张图像，输出目录: {output_dir}")
    success_count = 0
    failed: List[str] = []

    for image_path in image_paths:
        name = image_path.name
        stem = image_path.stem
        mask_out_path = masks_dir / f"{stem}_mask.png"
        try:
            mask = run_prediction(model, str(image_path), device, patch_size, threshold, remove_small)
            # cv2.imwrite 在 Windows 非 ASCII 路径下会静默失败
            ok, buf = cv2.imencode(".png", mask)
            if not ok:
                raise RuntimeError("PNG 编码失败")
            buf.tofile(str(mask_out_path))
            log(f"  {name} -> {mask_out_path.name}")
            success_count += 1
        except Exception as e:
            failed.append(name)
            log(f"  [错误] {name}: {e}")

    return success_count, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="批量根系分割")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "resources" / "configs" / "unet.yaml"),
    )
    parser.add_argument("--checkpoint", type=str, default="workspace/runs/training/best_model.pth")
    parser.add_argument("--image_dir", type=str, default="workspace/datasets/data/images")
    parser.add_argument("--output_dir", type=str, default="workspace/outputs")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()
    run_batch_prediction(
        args.config, args.checkpoint, args.image_dir, args.output_dir,
        no_cuda=args.no_cuda, line_callback=print,
    )


if __name__ == "__main__":
    main()
