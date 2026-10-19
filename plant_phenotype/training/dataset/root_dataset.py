import os
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from plant_phenotype.image_io import DEFAULT_IMAGE_EXTENSIONS, normalize_extensions, load_image_uint8_rgb


def _imread(path: str, flags: int) -> Optional[np.ndarray]:
    """cv2.imread 不支持非 ASCII 路径，改用 imdecode。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, flags) if buf.size else None


def _is_naming_mode(images_dir: str, masks_dir: str) -> bool:
    """子目录均为空或 '.' 时使用命名区分 image/mask（同目录下 xxx.png 与 xxx_mask.png）。"""
    return (not images_dir or images_dir.strip() == ".") and (not masks_dir or masks_dir.strip() == ".")


class RootSegDataset(Dataset):
    """
    根系语义分割数据集

    支持两种方式（与数据标注逻辑一致）：
    1) 命名区分：root_dir 下同一目录，图像 xxx.png 对应 mask xxx_mask.png（mask_suffix 可配）。
    2) 子文件夹（可选）：root_dir/images_dir 与 root_dir/masks_dir，图像与 mask 按文件名 stem 对应。

    图像格式：通过 image_extensions 支持多种后缀（.png, .jpg, .jpeg, .tiff, .tif, .gif, .webp, .bmp 等），
    未指定时可用 image_ext 单扩展名或默认多格式。mask 仍为 stem + mask_suffix + mask_ext（如 .png）。

    3) 显式配对：传入 pairs=[(原图绝对路径, 掩码绝对路径), ...]（来自数据集 samples.json）。

    root_dir 由 UI 或命令行指定，不在 YAML 中必填。支持 return_weight_mask、预计算 has_foreground。
    """

    def __init__(
        self,
        root_dir: str,
        images_dir: str = "",
        masks_dir: str = "",
        image_ext: str = ".png",
        image_extensions: Optional[Union[Tuple[str, ...], List[str]]] = None,
        mask_ext: str = ".png",
        mask_suffix: str = "_mask",
        transform: Optional[Callable] = None,
        file_list: Optional[List[str]] = None,
        return_weight_mask: bool = False,
        foreground_weight: float = 3.0,
        pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self._pair_masks: Optional[dict] = None
        self.mask_ext = mask_ext if mask_ext.startswith(".") else "." + mask_ext
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.return_weight_mask = return_weight_mask
        self.foreground_weight = foreground_weight
        self._naming_mode = _is_naming_mode(images_dir or "", masks_dir or "")
        # 支持多种图像格式：优先 image_extensions，否则用 image_ext 单扩展名
        if image_extensions is not None:
            self._image_extensions = normalize_extensions(image_extensions)
        else:
            self._image_extensions = normalize_extensions(image_ext)

        if self._naming_mode:
            self.images_dir = self.root_dir
            self.masks_dir = self.root_dir
        else:
            self.images_dir = os.path.join(self.root_dir, (images_dir or "images").strip())
            self.masks_dir = os.path.join(self.root_dir, (masks_dir or "masks").strip())

        if pairs is not None:
            self.images_dir = self.masks_dir = self.root_dir
            self.image_files = [img for img, _ in pairs]
            self._pair_masks = {img: mask for img, mask in pairs}
            if not self.image_files:
                raise RuntimeError("配对列表为空，没有可训练的样本")
        elif file_list is not None:
            self.image_files = list(file_list)
        else:
            if self._naming_mode:
                # 同目录：仅把“图像”当样本，排除已是 mask 的文件（stem 以 mask_suffix 结尾）
                all_files = os.listdir(self.images_dir)
                candidates = sorted(
                    [
                        f
                        for f in all_files
                        if any(f.lower().endswith(ext) for ext in self._image_extensions)
                        and not os.path.splitext(f)[0].endswith(self.mask_suffix)
                    ]
                )
            else:
                candidates = sorted(
                    [
                        f
                        for f in os.listdir(self.images_dir)
                        if any(f.lower().endswith(ext) for ext in self._image_extensions)
                    ]
                )
            # 仅保留存在对应 mask 的图像；若全部没有 mask 才报错，允许部分标注
            self.image_files = [
                f for f in candidates
                if os.path.isfile(self._mask_path_for(os.path.splitext(f)[0]))
            ]
            if not self.image_files:
                raise RuntimeError(
                    f"No image with a corresponding mask found in {self.images_dir}. "
                    f"Checked {len(candidates)} image(s). Ensure mask files exist in {self.masks_dir!r} "
                    f"with name: 图像 stem + {self.mask_suffix!r} + {self.mask_ext}（例如 image.png → image{self.mask_suffix}{self.mask_ext}）. "
                )

        self._has_foreground: Optional[List[bool]] = None

    def _mask_path_for_image(self, img_name: str) -> str:
        if self._pair_masks is not None:
            return self._pair_masks[img_name]
        return self._mask_path_for(os.path.splitext(img_name)[0])

    def _mask_path_for(self, base_name: str) -> str:
        if self._naming_mode:
            return os.path.join(self.masks_dir, base_name + self.mask_suffix + self.mask_ext)
        # 子文件夹模式：也支持 mask_suffix（如 images/xxx.png 对应 masks/xxx_mask.png）
        return os.path.join(self.masks_dir, base_name + self.mask_suffix + self.mask_ext)

    def precompute_foreground_flags(self) -> List[bool]:
        """预计算每张图像是否包含前景（根），用于过采样/欠采样。"""
        if self._has_foreground is not None:
            return self._has_foreground
        flags = []
        for img_name in self.image_files:
            mask = _imread(self._mask_path_for_image(img_name), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                flags.append(False)
            else:
                flags.append((mask > 0).any())
        self._has_foreground = flags
        return flags

    def get_sample_weights(
        self,
        oversample_foreground: float = 1.0,
        undersample_background: float = 1.0,
    ) -> List[float]:
        """
        返回每张图像的采样权重，用于 WeightedRandomSampler。
        - oversample_foreground: 包含前景的样本权重（>1 则过采样）
        - undersample_background: 仅背景的样本权重（<1 则欠采样）
        """
        flags = self.precompute_foreground_flags()
        return [
            oversample_foreground if fg else undersample_background
            for fg in flags
        ]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = self._mask_path_for_image(img_name)

        image = load_image_uint8_rgb(img_path)

        mask = _imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        # 将掩码二值化到 {0,1}
        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 统一为 (C,H,W) 和 (1,H,W) 的 float tensor
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        else:
            if image.dim() == 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1).float()
            else:
                image = image.float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).float()
            else:
                mask = mask.float()

        if self.return_weight_mask:
            # 目标区域（根）权重大，背景权重 1
            weight_mask = torch.where(
                mask > 0.5,
                torch.full_like(mask, self.foreground_weight),
                torch.ones_like(mask),
            )
            return image, mask, weight_mask
        return image, mask
