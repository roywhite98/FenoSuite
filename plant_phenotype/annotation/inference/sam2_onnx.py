# -*- coding: utf-8 -*-
"""SAM2 点提示分割（ONNX），与 X-AnyLabeling 同款 encoder/decoder 导出兼容。

marks 格式: [{"type": "point", "data": [x, y], "label": 1|0}, ...]
  label 1=正点，0=负点。至少需一个正点才会调用 decoder。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numpy import ndarray


def _import_onnxruntime():
    """延迟导入，避免未安装 onnxruntime 时无法启动主程序。"""
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "使用 SAM2 点提示需要安装 onnxruntime，请在环境中执行：pip install onnxruntime"
        ) from e
    return ort


def onnxruntime_providers_preferred() -> List[str]:
    """优先 CUDA，否则 CPU。"""
    ort = _import_onnxruntime()
    avail = set(ort.get_available_providers())
    order = []
    if "CUDAExecutionProvider" in avail:
        order.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in avail:
        order.append("CPUExecutionProvider")
    return order if order else list(ort.get_available_providers())


class SegmentAnything2ONNX:
    """Segment Anything 2：双 ONNX（encoder + decoder）。"""

    def __init__(self, encoder_model_path: str, decoder_model_path: str) -> None:
        providers = onnxruntime_providers_preferred()
        self.encoder = SAM2ImageEncoder(encoder_model_path, providers)
        self.decoder = SAM2ImageDecoder(
            decoder_model_path,
            providers,
            self.encoder.input_shape[2:],
        )

    def encode(self, cv_image: np.ndarray) -> Dict[str, Any]:
        original_size = cv_image.shape[:2]
        high_res_feats_0, high_res_feats_1, image_embed = self.encoder(cv_image)
        return {
            "high_res_feats_0": high_res_feats_0,
            "high_res_feats_1": high_res_feats_1,
            "image_embedding": image_embed,
            "original_size": original_size,
        }

    def predict_masks(self, embedding: Dict[str, Any], prompt: List[dict]) -> np.ndarray:
        points: List[List[int]] = []
        labels: List[int] = []
        for mark in prompt:
            if mark.get("type") == "point":
                points.append(mark["data"])
                labels.append(int(mark["label"]))
            elif mark.get("type") == "rectangle":
                d = mark["data"]
                points.append([d[0], d[1]])
                points.append([d[2], d[3]])
                labels.append(2)
                labels.append(3)
        if not points:
            raise ValueError("empty prompt")
        point_arr, label_arr = np.array(points, dtype=np.float32), np.array(labels, dtype=np.float32)

        image_embedding = embedding["image_embedding"]
        high_res_feats_0 = embedding["high_res_feats_0"]
        high_res_feats_1 = embedding["high_res_feats_1"]
        original_size = embedding["original_size"]
        self.decoder.set_image_size(original_size)
        masks, _ = self.decoder(
            image_embedding,
            high_res_feats_0,
            high_res_feats_1,
            point_arr,
            label_arr,
        )
        return masks


class SAM2ImageEncoder:
    def __init__(self, path: str, providers: List[str]) -> None:
        ort = _import_onnxruntime()
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        self.session = ort.InferenceSession(
            path, providers=providers, sess_options=sess_options
        )
        self.get_input_details()
        self.get_output_details()

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.encode_image(image)

    def encode_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_tensor = self.prepare_input(image)
        outputs = self.forward_encoder(input_tensor)
        return self.process_output(outputs)

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.resize(image, (self.input_width, self.input_height))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        input_img = (input_img.astype(np.float32) / 255.0 - mean) / std
        input_img = input_img.transpose(2, 0, 1)
        return input_img[np.newaxis, :, :, :].astype(np.float32)

    def forward_encoder(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        return self.session.run(self.output_names, {self.input_names[0]: input_tensor})

    def process_output(
        self, outputs: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return outputs[0], outputs[1], outputs[2]

    def get_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])

    def get_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


class SAM2ImageDecoder:
    def __init__(
        self,
        path: str,
        providers: List[str],
        encoder_input_size: Tuple[int, ...],
        orig_im_size: Optional[Tuple[int, int]] = None,
        mask_threshold: float = 0.0,
    ) -> None:
        ort = _import_onnxruntime()
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        self.session = ort.InferenceSession(
            path, providers=providers, sess_options=sess_options
        )
        self.orig_im_size: Tuple[int, int] = (
            (int(orig_im_size[0]), int(orig_im_size[1]))
            if orig_im_size is not None
            else (int(encoder_input_size[0]), int(encoder_input_size[1]))
        )
        self.encoder_input_size = encoder_input_size
        self.mask_threshold = mask_threshold
        self.scale_factor = 4
        self.get_input_details()
        self.get_output_details()

    def __call__(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords: Union[List[np.ndarray], np.ndarray],
        point_labels: Union[List[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, ndarray]:
        return self.predict(
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            point_coords,
            point_labels,
        )

    def predict(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords: Union[List[np.ndarray], np.ndarray],
        point_labels: Union[List[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, ndarray]:
        inputs = self.prepare_inputs(
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            point_coords,
            point_labels,
        )
        outputs = self.forward_decoder(inputs)
        return self.process_output(outputs)

    def prepare_inputs(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords: Union[List[np.ndarray], np.ndarray],
        point_labels: Union[List[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, ...]:
        input_point_coords, input_point_labels = self.prepare_points(
            point_coords, point_labels
        )
        num_labels = input_point_labels.shape[0]
        mask_input = np.zeros(
            (
                num_labels,
                1,
                self.encoder_input_size[0] // self.scale_factor,
                self.encoder_input_size[1] // self.scale_factor,
            ),
            dtype=np.float32,
        )
        has_mask_input = np.array([0], dtype=np.float32)
        return (
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            input_point_coords,
            input_point_labels,
            mask_input,
            has_mask_input,
        )

    def prepare_points(
        self,
        point_coords: Union[List[np.ndarray], np.ndarray],
        point_labels: Union[List[np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(point_coords, np.ndarray):
            input_point_coords = point_coords[np.newaxis, ...]
            input_point_labels = point_labels[np.newaxis, ...]
        else:
            max_num_points = max(coords.shape[0] for coords in point_coords)
            input_point_coords = np.zeros(
                (len(point_coords), max_num_points, 2), dtype=np.float32
            )
            input_point_labels = np.ones((len(point_coords), max_num_points), dtype=np.float32) * (
                -1
            )
            for i, (coords, labels) in enumerate(zip(point_coords, point_labels)):
                input_point_coords[i, : coords.shape[0], :] = coords
                input_point_labels[i, : labels.shape[0]] = labels

        input_point_coords[..., 0] = (
            input_point_coords[..., 0]
            / self.orig_im_size[1]
            * self.encoder_input_size[1]
        )
        input_point_coords[..., 1] = (
            input_point_coords[..., 1]
            / self.orig_im_size[0]
            * self.encoder_input_size[0]
        )
        return input_point_coords.astype(np.float32), input_point_labels.astype(np.float32)

    def forward_decoder(self, inputs: Tuple[np.ndarray, ...]) -> List[np.ndarray]:
        feed = {
            self.input_names[i]: inputs[i] for i in range(len(self.input_names))
        }
        return self.session.run(self.output_names, feed)

    def process_output(
        self, outputs: List[np.ndarray]
    ) -> Tuple[np.ndarray, ndarray]:
        scores = outputs[1].squeeze()
        masks = outputs[0][0]
        best_mask = masks[np.argmax(scores)]
        best_mask = cv2.resize(
            best_mask,
            (self.orig_im_size[1], self.orig_im_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.array([[best_mask]]), scores

    def set_image_size(self, orig_im_size: Tuple[int, int]) -> None:
        self.orig_im_size = (int(orig_im_size[0]), int(orig_im_size[1]))

    def get_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

    def get_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


def marks_need_positive(marks: List[dict]) -> bool:
    return any(
        m.get("type") == "point" and int(m.get("label", -1)) == 1 for m in marks
    )


def predict_best_mask(
    model: SegmentAnything2ONNX,
    image_rgb: np.ndarray,
    marks: List[dict],
    embedding: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """返回 (H,W) float32 最佳 mask 与 embedding（若传入 embedding 则原样返回，否则编码后返回）。"""
    if not marks or not marks_need_positive(marks):
        raise ValueError("至少需要一个正点")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb 须为 HxWx3 uint8")
    emb = embedding if embedding is not None else model.encode(image_rgb)
    masks = model.predict_masks(emb, marks)
    if len(masks.shape) == 4:
        m = masks[0][0]
    else:
        m = masks[0]
    return m, emb


class Sam2EmbeddingCache:
    """按 cache_key（如图像路径字符串）缓存 encode 结果。"""

    def __init__(self, max_items: int = 2) -> None:
        self._max = max(1, max_items)
        self._data: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []

    def clear(self) -> None:
        self._data.clear()
        self._order.clear()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._data.get(key)

    def put(self, key: str, embedding: Dict[str, Any]) -> None:
        if key in self._data:
            self._order.remove(key)
        elif len(self._order) >= self._max:
            oldest = self._order.pop(0)
            self._data.pop(oldest, None)
        self._order.append(key)
        self._data[key] = embedding
