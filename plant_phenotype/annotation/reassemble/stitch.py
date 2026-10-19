"""从切块目录按 meta.json 拼接并还原边缘。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image


def load_meta(tile_dir: Path) -> dict:
    with open(tile_dir / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


def reassemble_from_tiles(
    tile_dir: Path,
    output_image_path: Optional[Path] = None,
    output_mask_path: Optional[Path] = None,
    use_predicted_mask: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Restore tiles, preferring saved annotations over preprocessed masks.

    ``use_predicted_mask`` selects masks from the ``masks/`` subdirectory,
    matching the batch prediction output layout. Missing selected masks remain
    background instead of silently mixing predictions with annotations. If no
    predicted tile is found, raise ``FileNotFoundError``.
    """
    tile_dir = Path(tile_dir)
    meta = load_meta(tile_dir)
    ts = meta["tile_size"]
    cols, rows = meta["grid_size"][0], meta["grid_size"][1]
    resized_size = meta["resized_size"]
    original_size = meta["original_size"]
    w_resized, h_resized = resized_size[0], resized_size[1]
    w_orig, h_orig = original_size[0], original_size[1]
    need_restore = (w_resized != w_orig) or (h_resized != h_orig)

    img_canvas = np.zeros((h_resized, w_resized, 3), dtype=np.uint8)
    mask_canvas = np.zeros((h_resized, w_resized, 3), dtype=np.uint8)
    mask_dir = tile_dir / "masks" if use_predicted_mask else tile_dir
    loaded_masks = 0
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * ts, (r + 1) * ts
            x0, x1 = c * ts, (c + 1) * ts
            img_path = tile_dir / f"img_{r}_{c}.png"
            mask_path = next(
                (
                    path for path in (
                        mask_dir / f"img_{r}_{c}_mask.png",
                        mask_dir / f"mask_{r}_{c}.png",
                    )
                    if path.is_file()
                ),
                None,
            )
            if img_path.exists():
                with Image.open(img_path) as image:
                    img_canvas[y0:y1, x0:x1] = np.array(image.convert("RGB"))
            if mask_path is not None:
                with Image.open(mask_path) as mask:
                    m = np.array(mask)
                if m.ndim == 2:
                    m = np.stack([m] * 3, axis=-1)
                mask_canvas[y0:y1, x0:x1] = m
                loaded_masks += 1

    if use_predicted_mask and loaded_masks == 0:
        raise FileNotFoundError(f"未找到预测切块掩码: {mask_dir}")

    out_img = img_canvas
    out_mask = mask_canvas
    if need_restore:
        out_img = _restore_edges(out_img, w_resized, h_resized, w_orig, h_orig)
        out_mask = _restore_edges(
            out_mask, w_resized, h_resized, w_orig, h_orig,
            resample=Image.Resampling.NEAREST,
        )

    if output_image_path:
        Image.fromarray(out_img).save(output_image_path)
    if output_mask_path:
        Image.fromarray(out_mask).save(output_mask_path)
    return out_img, out_mask


def _restore_edges(
    img: np.ndarray,
    w_resized: int,
    h_resized: int,
    w_orig: int,
    h_orig: int,
    *,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> np.ndarray:
    if (w_resized, h_resized) == (w_orig, h_orig):
        return img
    pil = Image.fromarray(img)
    pil = pil.resize((w_orig, h_orig), resample)
    return np.array(pil)
