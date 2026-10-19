# -*- coding: utf-8 -*-
"""掩码无损读写、像素值/颜色到类别的映射与彩色显示。

映射键约定：灰度/编号/调色板掩码用十进制像素值（如 ``"255"``），
彩色掩码用 ``"#rrggbb"``。所有读写均经 PIL，兼容中文路径。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image

from plant_phenotype.datasets.schema import LabelClass

LOSSLESS_MASK_EXTENSIONS = (".png", ".tif", ".tiff", ".bmp")
MAX_COLOR_KEYS = 256


def read_image_size(path: str | Path) -> tuple[int, int]:
    """返回 (width, height)，只读文件头。"""
    with Image.open(path) as im:
        return int(im.width), int(im.height)


def read_raw_mask(path: str | Path) -> np.ndarray:
    """读取掩码原始值：灰度/调色板返回 (H,W)，彩色返回 (H,W,3)。三通道相同的 RGB 视为灰度。"""
    with Image.open(path) as im:
        if im.mode in ("1",):
            arr = np.array(im.convert("L"))
        elif im.mode in ("L", "P", "I", "I;16", "I;16B", "I;16L", "I;16N"):
            arr = np.array(im)
        elif im.mode == "LA":
            arr = np.array(im.getchannel(0))
        elif im.mode in ("RGB", "RGBA", "CMYK", "YCbCr"):
            arr = np.array(im.convert("RGB"))
        elif im.mode == "F":
            arr = np.array(im)
            if not np.all(np.mod(arr, 1) == 0):
                raise ValueError(f"浮点掩码含非整数值，无法作为标签读取: {path}")
            arr = arr.astype(np.int64)
        else:
            arr = np.array(im.convert("RGB"))
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 0], arr[:, :, 2]):
            arr = arr[:, :, 0]
    return arr


def read_label_mask(path: str | Path) -> np.ndarray:
    """读取数据集内部的类别编号图，返回 (H,W) 整型数组。"""
    arr = read_raw_mask(path)
    if arr.ndim != 2:
        raise ValueError(f"类别编号图应为单通道，实际为彩色: {path}")
    return arr


def color_key(rgb: Iterable[int]) -> str:
    r, g, b = (int(c) for c in list(rgb)[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def parse_color_key(key: str) -> tuple[int, int, int]:
    key = key.lstrip("#")
    return int(key[0:2], 16), int(key[2:4], 16), int(key[4:6], 16)


def is_color_key(key: str) -> bool:
    return key.startswith("#")


def unique_keys(raw: np.ndarray) -> dict[str, int]:
    """统计掩码中出现的像素值或颜色及其像素数。"""
    if raw.ndim == 2:
        vals, counts = np.unique(raw, return_counts=True)
        return {str(int(v)): int(c) for v, c in zip(vals, counts)}
    flat = raw.reshape(-1, 3).astype(np.uint32)
    packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    vals, counts = np.unique(packed, return_counts=True)
    if len(vals) > MAX_COLOR_KEYS:
        raise ValueError(f"颜色数 {len(vals)} 超过 {MAX_COLOR_KEYS}，掩码可能经过有损压缩或缩放插值")
    return {
        color_key(((int(v) >> 16) & 0xFF, (int(v) >> 8) & 0xFF, int(v) & 0xFF)): int(c)
        for v, c in zip(vals, counts)
    }


def apply_mapping(raw: np.ndarray, mapping: Mapping[str, int]) -> tuple[np.ndarray, dict[str, int]]:
    """按映射把原始掩码转为类别编号图。返回 (编号图, 未映射键→像素数)；未映射像素置 0。"""
    h, w = raw.shape[:2]
    out = np.zeros((h, w), dtype=np.int64)
    unknown: dict[str, int] = {}
    for key, count in unique_keys(raw).items():
        if raw.ndim == 2:
            if is_color_key(key):
                continue
            sel = raw == int(key)
        else:
            sel = np.all(raw == np.array(parse_color_key(key), dtype=raw.dtype), axis=-1)
        if key in mapping:
            out[sel] = int(mapping[key])
        else:
            unknown[key] = count
    return out, unknown


def to_storage_dtype(index: np.ndarray) -> np.ndarray:
    mx = int(index.max()) if index.size else 0
    if index.size and int(index.min()) < 0:
        raise ValueError("类别编号不能为负数")
    if mx <= 255:
        return index.astype(np.uint8)
    if mx <= 65535:
        return index.astype(np.uint16)
    raise ValueError(f"类别编号 {mx} 超出 16 位 PNG 范围")


def write_label_mask(path: str | Path, index: np.ndarray) -> None:
    """写入类别编号图（PNG 无损；≤255 用 8 位，否则 16 位）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = to_storage_dtype(np.asarray(index))
    img = Image.fromarray(arr)
    tmp = path.with_name(path.name + ".tmp.png")
    img.save(tmp, format="PNG")
    tmp.replace(path)


def write_binary_mask(path: str | Path, mask: np.ndarray) -> None:
    """实例二值掩码：0/255 PNG。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.png")
    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255).save(tmp, format="PNG")
    tmp.replace(path)


def write_color_mask(path: str | Path, index: np.ndarray, classes: Iterable[LabelClass]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colorize(index, classes)).save(path, format="PNG")


def preview_mask_rgb(index: np.ndarray) -> np.ndarray:
    """把掩码变成可看的预览图：背景保持 0，前景按最大像素值拉伸到 255。

    数据集里前景常常只写成类别号 1。直接当灰度显示时和黑色背景几乎没有差别。
    """
    arr = np.asarray(index)
    if arr.ndim == 3:
        arr = arr[..., 0]
    h, w = arr.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if arr.size == 0:
        return rgb
    mx = float(np.max(arr))
    if mx <= 0:
        return rgb
    scaled = np.clip(np.rint(arr.astype(np.float64) / mx * 255.0), 0, 255).astype(np.uint8)
    rgb[..., 0] = scaled
    rgb[..., 1] = scaled
    rgb[..., 2] = scaled
    return rgb


def colorize(index: np.ndarray, classes: Iterable[LabelClass]) -> np.ndarray:
    """类别编号图 → RGB；未定义类别显示为品红以便发现。"""
    h, w = index.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    known = np.zeros((h, w), dtype=bool)
    for c in classes:
        sel = index == c.id
        out[sel] = c.color
        known |= sel
    out[~known] = (255, 0, 255)
    return out


def overlay(image_rgb: np.ndarray, index: np.ndarray, classes: Iterable[LabelClass], alpha: float) -> np.ndarray:
    """原图上半透明叠加类别颜色（背景 0 不叠加）。"""
    if image_rgb.shape[:2] != index.shape[:2]:
        return image_rgb.copy()
    color = colorize(index, classes)
    fg = index != 0
    out = image_rgb.astype(np.float32)
    out[fg] = out[fg] * (1.0 - alpha) + color[fg].astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def load_rgb(path: str | Path) -> np.ndarray:
    """读取原图为 8 位 RGB，兼容 16 位/浮点 TIFF、调色板、RGBA。"""
    with Image.open(path) as im:
        if im.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
            arr = np.array(im).astype(np.float64)
            mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
            arr = (arr - mn) / (mx - mn) * 255 if mx > mn else np.zeros_like(arr)
            return np.stack([np.nan_to_num(arr).astype(np.uint8)] * 3, axis=-1)
        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel("A"))
            return np.array(bg)
        return np.array(im.convert("RGB"))
