# -*- coding: utf-8 -*-
"""魔法棒分割器：加载模型，对点击处候选区域做语义分割。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

from plant_phenotype.annotation.config import MODEL_EXTENSIONS, DEFAULT_RPN_CROP_SIZE


def list_models_in_dir(dir_path: Optional[Path]) -> List[Path]:
    if dir_path is None or not dir_path.is_dir():
        return []
    paths = []
    for ext in MODEL_EXTENSIONS:
        paths.extend(dir_path.glob(f"*{ext}"))
    return sorted(paths, key=lambda p: p.name)


def _output_to_mask(out: np.ndarray) -> np.ndarray:
    if out.ndim == 3:
        pred = out[1] if out.shape[0] >= 2 else out[0]
    elif out.ndim == 2:
        pred = out
    else:
        pred = np.squeeze(out)
    return (pred > 0.5).astype(np.uint8)


class Segmenter:
    def __init__(self):
        self._session = None
        self._pytorch_model = None
        self._backend = None  # "onnx" | "pytorch" | None
        self._input_name: Optional[str] = None
        self._input_shape: Optional[Tuple[int, ...]] = None
        self._model_path: Optional[Path] = None
        self._last_load_error: Optional[str] = None

    def load(
        self,
        model_path: Optional[Path],
        extra_python_path: Optional[Path] = None,
    ) -> bool:
        self._session = None
        self._pytorch_model = None
        self._backend = None
        self._input_name = None
        self._input_shape = None
        self._model_path = None
        self._last_load_error = None
        if model_path is None or not model_path.is_file():
            return False
        suf = model_path.suffix.lower()
        if suf == ".onnx":
            return self._load_onnx(model_path)
        if suf in (".pth", ".pt"):
            return self._load_pytorch(model_path, extra_python_path)
        return False

    def get_last_load_error(self) -> Optional[str]:
        return self._last_load_error

    def _load_onnx(self, model_path: Path) -> bool:
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                str(model_path),
                providers=ort.get_available_providers(),
            )
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            raw = list(inp.shape) if inp.shape else [1, 3, 512, 512]
            if len(raw) < 4:
                raw = [1, 3, 512, 512]
            def _dim(i: int, default: int) -> int:
                d = raw[i] if i < len(raw) else default
                return int(d) if isinstance(d, int) and d > 0 else default
            self._input_shape = (1, 3, _dim(2, 512), _dim(3, 512))
            self._model_path = model_path
            self._backend = "onnx"
            return True
        except Exception:
            return False

    def _load_pytorch(self, model_path: Path, extra_python_path: Optional[Path] = None) -> bool:
        self._last_load_error = None
        import torch

        try:
            loaded = torch.load(str(model_path), map_location="cpu", weights_only=True)
        except Exception:
            loaded = None
        if isinstance(loaded, dict):
            sidecar = model_path.parent / (model_path.name + ".json")
            if not sidecar.is_file():
                self._last_load_error = (
                    f"该文件仅为权重。请在同目录下新建 {model_path.name}.json，"
                    "内容示例：{\"model_class\": \"类名\", \"kwargs\": {\"in_channels\": 3, \"num_classes\": 2}}。"
                )
                return False
            try:
                with open(sidecar, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                model_class = cfg.get("model_class")
                kwargs = cfg.get("kwargs") or {}
                if not model_class:
                    self._last_load_error = f"{sidecar.name} 中缺少 \"model_class\"。"
                    return False
            except Exception as e:
                self._last_load_error = f"读取 {sidecar.name} 失败: {e}"
                return False
            try:
                from .custom_model import get_model
                model = get_model(model_class, **kwargs)
                model.load_state_dict(loaded, strict=True)
                model.eval()
                self._pytorch_model = model
                self._input_shape = (1, 3, DEFAULT_RPN_CROP_SIZE, DEFAULT_RPN_CROP_SIZE)
                self._model_path = model_path
                self._backend = "pytorch"
                return True
            except Exception as e:
                self._last_load_error = f"{type(e).__name__}: {e}"
                return False

        added_path = None
        if extra_python_path is not None and extra_python_path.is_dir():
            added_path = str(extra_python_path.resolve())
            if added_path not in sys.path:
                sys.path.insert(0, added_path)
        try:
            loaded = torch.load(str(model_path), map_location="cpu", weights_only=False)
            if isinstance(loaded, dict):
                self._last_load_error = "该文件为 state_dict，请使用同名 .json 配置加载。"
                return False
            if not hasattr(loaded, "parameters"):
                self._last_load_error = "该文件不是 nn.Module。"
                return False
            model = loaded
            model.eval()
            self._pytorch_model = model
            self._input_shape = (1, 3, DEFAULT_RPN_CROP_SIZE, DEFAULT_RPN_CROP_SIZE)
            self._model_path = model_path
            self._backend = "pytorch"
            return True
        except Exception as e:
            self._last_load_error = f"{type(e).__name__}: {e}"
            return False
        finally:
            if added_path and added_path in sys.path:
                try:
                    sys.path.remove(added_path)
                except ValueError:
                    pass

    def is_loaded(self) -> bool:
        return self._backend is not None

    def run(
        self,
        image_rgb: np.ndarray,
        center_x: int,
        center_y: int,
        crop_size: int = DEFAULT_RPN_CROP_SIZE,
    ) -> Optional[np.ndarray]:
        if not self.is_loaded() or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return None
        h, w = image_rgb.shape[:2]
        half = crop_size // 2
        x0 = max(0, center_x - half)
        y0 = max(0, center_y - half)
        x1 = min(w, center_x - half + crop_size)
        y1 = min(h, center_y - half + crop_size)
        crop = image_rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
        if crop.size == 0:
            return None
        _, _, mh, mw = self._input_shape
        from PIL import Image
        crop_pil = Image.fromarray((crop * 255).astype(np.uint8))
        crop_resized = np.array(crop_pil.resize((mw, mh), Image.Resampling.BILINEAR))
        if crop_resized.ndim == 2:
            crop_resized = np.stack([crop_resized] * 3, axis=-1)
        x_nchw = np.transpose(crop_resized.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]

        pred = None
        if self._backend == "onnx":
            try:
                outputs = self._session.run(None, {self._input_name: x_nchw})
                out = outputs[0]
                pred = out[0] if out.ndim >= 3 else np.squeeze(out)
            except Exception:
                return None
        elif self._backend == "pytorch":
            try:
                import torch
                with torch.no_grad():
                    x = torch.from_numpy(x_nchw).float()
                    out = self._pytorch_model(x)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    pred = out.cpu().numpy()
                    pred = pred[0] if pred.ndim >= 3 else np.squeeze(pred)
            except Exception:
                return None

        if pred is None:
            return None
        pred_u8 = _output_to_mask(pred if pred.ndim >= 2 else pred[np.newaxis, ...])
        ch, cw = crop.shape[0], crop.shape[1]
        pred_pil = Image.fromarray(pred_u8)
        pred_crop = np.array(pred_pil.resize((cw, ch), Image.Resampling.NEAREST))
        full_mask = np.zeros((h, w), dtype=bool)
        full_mask[y0:y1, x0:x1] = pred_crop > 0
        return full_mask

    def run_full_image(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        if not self.is_loaded() or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            return None
        h, w = image_rgb.shape[:2]
        _, _, mh, mw = self._input_shape
        from PIL import Image
        img_pil = Image.fromarray(image_rgb.astype(np.uint8))
        img_resized = np.array(img_pil.resize((mw, mh), Image.Resampling.BILINEAR))
        if img_resized.ndim == 2:
            img_resized = np.stack([img_resized] * 3, axis=-1)
        x_nchw = np.transpose(img_resized.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]

        pred = None
        if self._backend == "onnx":
            try:
                outputs = self._session.run(None, {self._input_name: x_nchw})
                out = outputs[0]
                pred = out[0] if out.ndim >= 3 else np.squeeze(out)
            except Exception:
                return None
        elif self._backend == "pytorch":
            try:
                import torch
                with torch.no_grad():
                    x = torch.from_numpy(x_nchw).float()
                    out = self._pytorch_model(x)
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    pred = out.cpu().numpy()
                    pred = pred[0] if pred.ndim >= 3 else np.squeeze(pred)
            except Exception:
                return None

        if pred is None:
            return None
        pred_u8 = _output_to_mask(pred if pred.ndim >= 2 else pred[np.newaxis, ...])
        pred_pil = Image.fromarray(pred_u8)
        full_mask = np.array(pred_pil.resize((w, h), Image.Resampling.NEAREST)) > 0
        return full_mask
