# -*- coding: utf-8 -*-
"""冻结编码器的残差阶段。

layer1、layer2 指 ResNet 的前两个残差阶段，不是前两次卷积。stem（conv1、bn1）保持可训练。
只关 requires_grad 冻不住 BN 的运行均值和方差，因此每次 model.train() 后把这些阶段设回 eval。
"""
from __future__ import annotations

from typing import Iterable

import torch.nn as nn


def freeze_encoder_stages(model: nn.Module, stages: Iterable[str]) -> list[str]:
    stages = [str(s) for s in stages if str(s).strip()]
    if not stages:
        return []
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise ValueError(f"模型没有 encoder，无法冻结 {stages}")
    missing = [s for s in stages if not hasattr(encoder, s)]
    if missing:
        raise ValueError(f"编码器没有这些阶段: {missing}（ResNet 为 layer1、layer2、layer3、layer4）")
    for stage in stages:
        getattr(encoder, stage).requires_grad_(False)
    original_train = model.train

    def train(mode: bool = True):
        original_train(mode)
        if mode:
            for stage in stages:
                getattr(model.encoder, stage).eval()
        return model

    model.train = train  # type: ignore[method-assign]
    if model.training:
        train(True)
    return stages
