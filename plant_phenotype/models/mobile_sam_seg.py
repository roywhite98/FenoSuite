# -*- coding: utf-8 -*-
"""MobileSAM（TinyViT）整图分割。

与参考实验一致：输入先缩放到 1024，空提示（无点、框、掩码）经官方 mask decoder 出单掩码，
再还原到原图尺寸；按图像逐张解码，避免官方 decoder 把 batch 当成提示维相乘。
本项目数据管道把像素归一化到 (x-0.5)/0.5，这里在进入编码器前还原并改用 SAM 的均值方差。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

SAM_MEAN = (123.675, 116.28, 103.53)
SAM_STD = (58.395, 57.12, 57.375)


class MobileSAMSeg(nn.Module):
    def __init__(self, checkpoint: Optional[str] = None, encoder_size: int = 1024):
        super().__init__()
        from plant_phenotype.vendor.mobile_sam.build_sam import sam_model_registry
        self.encoder_size = int(encoder_size)
        self.sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
        self.register_buffer("sam_mean", torch.tensor(SAM_MEAN).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("sam_std", torch.tensor(SAM_STD).view(1, -1, 1, 1), persistent=False)

    @property
    def encoder(self):
        return self.sam.image_encoder

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        native_size = images.shape[-2:]
        pixels = (images + 1.0) * (0.5 * 255.0)
        normalized = (pixels - self.sam_mean) / self.sam_std
        encoder_input = F.interpolate(normalized, size=(self.encoder_size, self.encoder_size), mode="bilinear", align_corners=False)
        embeddings = self.encoder(encoder_input)
        sparse, dense = self.sam.prompt_encoder(points=None, boxes=None, masks=None)
        positional = self.sam.prompt_encoder.get_dense_pe()
        logits = []
        for embedding in embeddings.split(1):
            mask, _ = self.sam.mask_decoder(
                image_embeddings=embedding,
                image_pe=positional,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            logits.append(mask)
        return F.interpolate(torch.cat(logits), size=native_size, mode="bilinear", align_corners=False)
