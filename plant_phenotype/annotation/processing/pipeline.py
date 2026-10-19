"""预处理管线：转 PNG、二值化、切块、密度过滤。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from plant_phenotype.annotation.config import (
    DEFAULT_TILE_SIZE,
    DEFAULT_MIN_FOREGROUND_RATIO,
    IMAGE_EXTENSIONS,
    MASK_SUFFIX,
)


def _ensure_png_path(image_path: Path, out_dir: Path) -> Path:
    stem = image_path.stem
    out_path = out_dir / f"{stem}.png"
    img = Image.open(image_path).convert("RGB")
    img.save(out_path, "PNG")
    return out_path


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        is_black = (mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)
        out = np.zeros_like(mask)
    else:
        is_black = mask == 0
        out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[~is_black] = 255
    return out


def to_png_and_binarize(
    image_path: Path,
    mask_array: np.ndarray,
    out_dir: Path,
) -> Tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = _ensure_png_path(Path(image_path), out_dir)
    stem = png_path.stem
    mask_path = out_dir / f"{stem}{MASK_SUFFIX}.png"
    if mask_array.ndim == 2:
        mask_array = np.stack([mask_array] * 3, axis=-1)
    bin_mask = binarize_mask(mask_array)
    Image.fromarray(bin_mask).save(mask_path)
    return png_path, mask_path


def _resize_to_multiple(img: np.ndarray, tile_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    nw = ((w + tile_size - 1) // tile_size) * tile_size
    nh = ((h + tile_size - 1) // tile_size) * tile_size
    if (nw, nh) == (w, h):
        return img
    pil = Image.fromarray(img)
    pil = pil.resize((nw, nh), Image.Resampling.LANCZOS)
    return np.array(pil)


def tile_and_filter(
    image_path: Path,
    mask_path: Optional[Path],
    tiles_base_dir: Path,
    tile_size: int = DEFAULT_TILE_SIZE,
    min_foreground_ratio: float = DEFAULT_MIN_FOREGROUND_RATIO,
) -> dict:
    tiles_base_dir = Path(tiles_base_dir)
    img = np.array(Image.open(image_path).convert("RGB"))
    if mask_path is not None and mask_path.exists():
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask_fg = (mask[:, :, 0] > 0) | (mask[:, :, 1] > 0) | (mask[:, :, 2] > 0)
        else:
            mask_fg = mask > 0
    else:
        mask_fg = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
    original_size = [img.shape[1], img.shape[0]]
    img = _resize_to_multiple(img, tile_size)
    mask_pil = Image.fromarray((mask_fg.astype(np.uint8) * 255))
    mask_pil = mask_pil.resize((img.shape[1], img.shape[0]), Image.Resampling.NEAREST)
    mask_arr = np.array(mask_pil)
    mask_fg = (mask_arr[:, :, 0] > 0) if mask_arr.ndim == 3 else (mask_arr > 0)
    if mask_fg.shape[0] != img.shape[0] or mask_fg.shape[1] != img.shape[1]:
        mask_pil = Image.fromarray((mask_fg.astype(np.uint8) * 255))
        mask_pil = mask_pil.resize((img.shape[1], img.shape[0]), Image.Resampling.NEAREST)
        mask_arr = np.array(mask_pil)
        mask_fg = (mask_arr[:, :, 0] > 0) if mask_arr.ndim == 3 else (mask_arr > 0)
    resized_size = [img.shape[1], img.shape[0]]
    cols = img.shape[1] // tile_size
    rows = img.shape[0] // tile_size
    grid_size = [cols, rows]
    basename = image_path.stem
    tile_dir = tiles_base_dir / basename
    tile_dir.mkdir(parents=True, exist_ok=True)
    has_annotation = np.any(mask_fg)
    excluded_tiles: List[List[int]] = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * tile_size, (r + 1) * tile_size
            x0, x1 = c * tile_size, (c + 1) * tile_size
            img_patch = img[y0:y1, x0:x1]
            mask_patch = mask_fg[y0:y1, x0:x1]
            ratio = float(np.sum(mask_patch)) / mask_patch.size
            img_path = tile_dir / f"img_{r}_{c}.png"
            Image.fromarray(img_patch).save(img_path)
            if has_annotation:
                mask_tile_path = tile_dir / f"mask_{r}_{c}.png"
                hw = (img_patch.shape[0], img_patch.shape[1])
                mask_tile = np.zeros((hw[0], hw[1], 3), dtype=np.uint8)
                ph, pw = min(mask_patch.shape[0], hw[0]), min(mask_patch.shape[1], hw[1])
                mask_tile[:ph, :pw, :] = (np.asarray(mask_patch[:ph, :pw], dtype=np.uint8).reshape(ph, pw, 1) * 255).repeat(3, axis=-1)
                Image.fromarray(mask_tile).save(mask_tile_path)
            if ratio < min_foreground_ratio:
                excluded_tiles.append([r, c])
    meta = {
        "original_size": original_size,
        "tile_size": tile_size,
        "grid_size": grid_size,
        "resized_size": resized_size,
        "excluded_tiles": excluded_tiles,
        "has_mask": bool(has_annotation),
    }
    with open(tile_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def run_preprocessing_on_dir(
    annotated_dir: Path,
    tiles_base_dir: Path,
    tile_size: int = DEFAULT_TILE_SIZE,
    min_foreground_ratio: float = DEFAULT_MIN_FOREGROUND_RATIO,
) -> int:
    annotated_dir = Path(annotated_dir)
    tiles_base_dir = Path(tiles_base_dir)
    ext_set = {e.lower() for e in IMAGE_EXTENSIONS}
    stem_to_img: dict = {}
    for p in annotated_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in ext_set:
            continue
        stem = p.stem
        if stem.endswith(MASK_SUFFIX):
            continue
        stem_to_img[stem] = p
    for stem, img_path in stem_to_img.items():
        mask_path = annotated_dir / f"{stem}{MASK_SUFFIX}.png"
        if not mask_path.exists():
            mask_path = None
        tile_and_filter(img_path, mask_path, tiles_base_dir, tile_size, min_foreground_ratio)
    return len(stem_to_img)
