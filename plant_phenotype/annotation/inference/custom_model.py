# -*- coding: utf-8 -*-
"""
魔法棒用：仅权重（state_dict）加载时的模型定义与注册。
"""
from __future__ import annotations

from typing import Any, Dict, Type

_REGISTRY: Dict[str, Type] = {}


def register(name: str):
    def _reg(cls: Type):
        _REGISTRY[name] = cls
        return cls
    return _reg


def get_model(class_name: str, **kwargs: Any):
    if class_name not in _REGISTRY:
        raise KeyError(
            f"未注册的模型类 '{class_name}'。"
            f"请在 annotation.inference.custom_model 中定义并 @register('{class_name}')。"
            f"已注册: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[class_name](**kwargs)


try:
    import torch
    import torch.nn as nn

    @register("SimpleUNet")
    class SimpleUNet(nn.Module):
        def __init__(self, in_channels: int = 3, num_classes: int = 1):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.dec = nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, num_classes, 1),
            )

        def forward(self, x):
            x = self.enc(x)
            return self.dec(x)

except ImportError:
    pass
