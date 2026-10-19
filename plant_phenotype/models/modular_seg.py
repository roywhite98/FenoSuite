"""
模块化语义分割模型
支持自由组合：编码器 + 上下文模块 + 特征融合 + 解码器
"""
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.encoders import build_encoder, ENCODERS
from .modules.context import build_context_module, CONTEXT_MODULES
from .modules.fusion import build_fusion_module, FUSION_MODULES
from .modules.decoders import build_decoder, DECODERS


class ModularSegModel(nn.Module):
    """
    模块化语义分割模型

    架构：Encoder -> Context Module -> Feature Fusion -> Decoder

    Args:
        encoder: 编码器名称 (vgg16, resnet50, segformer-b2, swin_tiny, ...)
        decoder: 解码器名称 (unet, segformer_head, upernet, lawin, simple)
        context: 上下文模块 (aspp, psp, self_attention, cbam, eca, none)
        fusion: 特征融合方式 (fpn, bifpn, asff, concat, add, none)
        in_channels: 输入通道数
        num_classes: 输出类别数
        pretrained: 是否使用预训练权重
        decoder_channels: 解码器通道数
    """

    def __init__(
        self,
        encoder: str = "resnet50",
        decoder: str = "unet",
        context: str = "none",
        fusion: str = "none",
        in_channels: int = 3,
        num_classes: int = 1,
        pretrained: bool = True,
        decoder_channels: int = 256,
        context_kwargs: Optional[Dict[str, Any]] = None,
        fusion_kwargs: Optional[Dict[str, Any]] = None,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.encoder_name = encoder
        self.decoder_name = decoder
        self.context_name = context
        self.fusion_name = fusion

        context_kwargs = context_kwargs or {}
        fusion_kwargs = fusion_kwargs or {}
        decoder_kwargs = decoder_kwargs or {}

        # 1. 构建编码器
        self.encoder, encoder_channels = build_encoder(encoder, pretrained, in_channels)

        # 2. 构建上下文模块（应用于最深层特征）
        if context.lower() != "none":
            self.context = build_context_module(
                context, encoder_channels[-1], decoder_channels, **context_kwargs
            )
            # 更新最深层通道数
            if hasattr(self.context, 'out_channels'):
                encoder_channels = list(encoder_channels)
                encoder_channels[-1] = self.context.out_channels
            elif context.lower() in ["aspp", "psp"]:
                encoder_channels = list(encoder_channels)
                encoder_channels[-1] = decoder_channels
        else:
            self.context = None

        # 3. 构建特征融合模块
        if fusion.lower() != "none":
            self.fusion = build_fusion_module(
                fusion, encoder_channels, decoder_channels, **fusion_kwargs
            )
            # FPN/BiFPN/ASFF 输出多尺度特征，通道统一
            if fusion.lower() in ["fpn", "bifpn", "asff"]:
                fused_channels = [decoder_channels] * len(encoder_channels)
            else:
                fused_channels = encoder_channels
        else:
            self.fusion = None
            fused_channels = encoder_channels

        # 4. 构建解码器
        self.decoder = build_decoder(
            decoder, list(fused_channels), num_classes,
            embed_dim=decoder_channels, channels=decoder_channels, **decoder_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        # 1. 编码器
        features = self.encoder(x)

        # 2. 上下文模块（仅最深层）
        if self.context is not None:
            features = list(features)
            features[-1] = self.context(features[-1])

        # 3. 特征融合
        if self.fusion is not None:
            features = self.fusion(features)
            # ConcatFusion/AddFusion 返回单个 tensor
            if not isinstance(features, list):
                features = [features]

        # 4. 解码器
        out = self.decoder(features, target_size=input_size)

        return out

    @staticmethod
    def get_available_components():
        """返回所有可用组件"""
        return {
            "encoders": ENCODERS,
            "decoders": DECODERS,
            "context_modules": CONTEXT_MODULES,
            "fusion_modules": FUSION_MODULES,
        }


def build_modular_model(cfg: dict) -> ModularSegModel:
    """
    从配置字典构建模块化模型

    配置示例:
    model:
      type: modular
      encoder: segformer-b2
      decoder: segformer_head
      context: aspp
      fusion: fpn
      in_channels: 3
      classes: 1
      decoder_channels: 256
    """
    return ModularSegModel(
        encoder=cfg.get("encoder", "resnet50"),
        decoder=cfg.get("decoder", "unet"),
        context=cfg.get("context", "none"),
        fusion=cfg.get("fusion", "none"),
        in_channels=cfg.get("in_channels", 3),
        num_classes=cfg.get("classes", 1),
        pretrained=cfg.get("pretrained", True),
        decoder_channels=cfg.get("decoder_channels", 256),
        context_kwargs=cfg.get("context_kwargs", {}),
        fusion_kwargs=cfg.get("fusion_kwargs", {}),
        decoder_kwargs=cfg.get("decoder_kwargs", {}),
    )
