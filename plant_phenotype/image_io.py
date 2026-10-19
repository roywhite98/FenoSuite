"""Shared image loading and filename conventions for training and inference."""
from typing import List, Tuple, Union


DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".gif")


def load_image_uint8_rgb(path: str):
    """Decode model inputs identically for training, prediction and preview.

    Preserve the training loader's OpenCV conversion (including 16-bit to
    8-bit conversion), so existing checkpoints retain their input scale.
    Pillow handles formats the installed OpenCV build cannot decode.
    """
    import cv2
    import numpy as np
    from PIL import Image

    encoded = np.fromfile(path, dtype=np.uint8)
    try:
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    except cv2.error:
        bgr = None
    if bgr is not None:
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    with Image.open(path) as img:
        if img.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
            arr = np.asarray(img)
            if arr.dtype.kind == "u" and arr.dtype.itemsize == 2:
                arr = (arr.astype(np.uint32) // 256).astype(np.uint8)
            else:
                finite = np.isfinite(arr)
                out = np.zeros(arr.shape, dtype=np.uint8)
                if finite.any():
                    lo, hi = float(arr[finite].min()), float(arr[finite].max())
                    if hi > lo:
                        scaled = (arr[finite].astype(np.float64) - lo) / (hi - lo)
                        out[finite] = np.clip(scaled * 255, 0, 255).astype(np.uint8)
                arr = out
            rgb = np.repeat(arr[:, :, None], 3, axis=2)
        else:
            rgb = np.asarray(img.convert("RGB"))
    if rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError(f"Empty image: {path}")
    return np.ascontiguousarray(rgb)


def normalize_extensions(exts: Union[str, List[str], Tuple[str, ...]]) -> Tuple[str, ...]:
    """Normalize image extensions to lowercase, dot-prefixed values."""
    if isinstance(exts, str):
        exts = (exts,)
    out = []
    for ext in exts:
        ext = ext.strip().lower()
        if ext and not ext.startswith("."):
            ext = "." + ext
        if ext:
            out.append(ext)
    return tuple(out) if out else (".png",)
