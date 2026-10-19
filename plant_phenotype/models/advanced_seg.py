"""
带注意力和边界细化的高级分割模型
"""
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.encoders import build_encoder
from .modules.context import build_context_module
from .modules.fusion import build_fusion_module
from .modules.decoders import build_decoder
from .modules.attention import build_attention_module
from .modules.refinement import build_refinement_module, BoundaryAwareModule


class AttentionSegModel(nn.Module):
    """
    带注意力机制的分割模型
    在编码器各层添加注意力模块
    """
    def __init__(
        self,
        encoder: str = "resnet50",
        decoder: str = "unet",
        attention: str = "cbam",
        num_classes: int = 1,
        pretrained: bool = True,
        decoder_channels: int = 256,
        insert_attention_at: list = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if insert_attention_at is None:
            insert_attention_at = [1, 2, 3]
        attention_kwargs = attention_kwargs or {}

        # 编码器
        self.encoder, encoder_channels = build_encoder(encoder, pretrained, in_channels=3)

        # 注意力模块（应用于指定层）
        self.attention_modules = nn.ModuleList()
        for i in range(len(encoder_channels)):
            if i in insert_attention_at:
                attn = build_attention_module(attention, encoder_channels[i], **attention_kwargs)
                self.attention_modules.append(attn)
            else:
                self.attention_modules.append(nn.Identity())

        # 解码器
        self.decoder = build_decoder(decoder, list(encoder_channels), num_classes,
                                     embed_dim=decoder_channels, channels=decoder_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        # 编码
        features = self.encoder(x)

        # 应用注意力
        features = [attn(feat) for attn, feat in zip(self.attention_modules, features)]

        # 解码
        out = self.decoder(features, target_size=input_size)

        return out


class RefinementSegModel(nn.Module):
    """
    带边界细化的分割模型
    粗预测 -> 边界细化 -> 精细预测
    """
    def __init__(
        self,
        encoder: str = "resnet50",
        decoder: str = "unet",
        refinement: str = "boundary_refine",
        num_classes: int = 1,
        pretrained: bool = True,
        decoder_channels: int = 256,
    ):
        super().__init__()

        # 编码器
        self.encoder, encoder_channels = build_encoder(encoder, pretrained, in_channels=3)

        # 粗预测解码器
        self.coarse_decoder = build_decoder(decoder, list(encoder_channels), num_classes,
                                           embed_dim=decoder_channels, channels=decoder_channels)

        # 边界细化模块
        self.refinement = build_refinement_module(
            refinement,
            in_channels=encoder_channels[0] + num_classes,  # 低层特征 + 粗预测
            out_channels=num_classes,
            detail_channels=encoder_channels[0],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        # 编码
        features = self.encoder(x)

        # 粗预测
        coarse_pred = self.coarse_decoder(features, target_size=input_size)

        # 边界细化（使用最低层特征）
        low_feat = F.interpolate(features[0], size=input_size, mode='bilinear', align_corners=False)
        concat = torch.cat([low_feat, coarse_pred], dim=1)

        if hasattr(self.refinement, 'forward') and 'prev_pred' in self.refinement.forward.__code__.co_varnames:
            # ResidualRefinement 需要分开传参
            refined_pred = self.refinement(low_feat, coarse_pred)
        else:
            # BoundaryRefinement: forward(coarse, detail)，coarse=concat(低层+粗预测 65ch)，detail=low_feat(64ch)
            refined_pred = self.refinement(concat, low_feat)

        return refined_pred


class BoundaryAwareSegModel(nn.Module):
    """
    边界感知分割模型
    联合学习语义分割和边缘检测
    """
    def __init__(
        self,
        encoder: str = "resnet50",
        decoder: str = "unet",
        num_classes: int = 1,
        pretrained: bool = True,
        decoder_channels: int = 256,
    ):
        super().__init__()

        # 编码器
        self.encoder, encoder_channels = build_encoder(encoder, pretrained, in_channels=3)

        # 解码器（用于提取高层特征）
        self.decoder_features = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )

        # 上采样（先定义，边界感知接收其输出通道数）
        self.upsample = nn.Sequential(
            nn.Conv2d(decoder_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # 边界感知模块：输入为上采样后的 64 通道特征
        self.boundary_aware = BoundaryAwareModule(64, num_classes)

    def forward(self, x: torch.Tensor, return_edge: bool = False) -> torch.Tensor:
        input_size = x.shape[-2:]

        # 编码
        features = self.encoder(x)

        # 解码特征
        decode_feat = self.decoder_features(features[-1])
        decode_feat = F.interpolate(decode_feat, scale_factor=4, mode='bilinear', align_corners=False)
        decode_feat = self.upsample(decode_feat)
        decode_feat = F.interpolate(decode_feat, size=input_size, mode='bilinear', align_corners=False)

        # 边界感知
        seg_pred, edge_pred, refined_pred = self.boundary_aware(decode_feat)

        if return_edge:
            return refined_pred, edge_pred
        return refined_pred


class MultiTaskSegModel(nn.Module):
    """
    多任务分割模型
    同时输出：语义分割 + 边缘检测 + 距离图
    """
    def __init__(
        self,
        encoder: str = "resnet50",
        num_classes: int = 1,
        pretrained: bool = True,
        decoder_channels: int = 256,
    ):
        super().__init__()

        # 共享编码器
        self.encoder, encoder_channels = build_encoder(encoder, pretrained, in_channels=3)

        # 共享特征
        self.shared_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )

        # 分割头
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
        )

        # 边缘检测头
        self.edge_head = nn.Sequential(
            nn.Conv2d(decoder_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

        # 距离图头（到边界的距离）
        self.dist_head = nn.Sequential(
            nn.Conv2d(decoder_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]

        # 编码
        features = self.encoder(x)

        # 共享特征
        shared_feat = self.shared_conv(features[-1])
        shared_feat = F.interpolate(shared_feat, size=input_size, mode='bilinear', align_corners=False)

        # 多任务输出
        seg_pred = self.seg_head(shared_feat)
        edge_pred = self.edge_head(shared_feat)
        dist_pred = self.dist_head(shared_feat)

        return {
            'segmentation': seg_pred,
            'edge': edge_pred,
            'distance': dist_pred,
        }


def build_advanced_model(cfg: dict) -> nn.Module:
    """
    构建高级分割模型

    配置示例:
    model:
      type: attention_seg  # attention_seg, refinement_seg, boundary_aware_seg, multi_task_seg
      encoder: resnet50
      decoder: unet
      attention: cbam  # 用于 attention_seg
      refinement: boundary_refine  # 用于 refinement_seg
      num_classes: 1
      pretrained: true
      decoder_channels: 256
    """
    model_type = cfg.get("type", "attention_seg")

    if model_type == "attention_seg":
        attention = cfg.get("attention", "cbam")
        attention_kwargs = cfg.get("attention_kwargs", {})
        # root_topology 等需额外参数时，可从 model 节传入
        if attention == "root_topology":
            attention_kwargs.setdefault("window_size", cfg.get("window_size", 8))
            attention_kwargs.setdefault("use_topology_gate", cfg.get("use_topology_gate", True))
        return AttentionSegModel(
            encoder=cfg.get("encoder", "resnet50"),
            decoder=cfg.get("decoder", "unet"),
            attention=attention,
            num_classes=cfg.get("num_classes", 1),
            pretrained=cfg.get("pretrained", True),
            decoder_channels=cfg.get("decoder_channels", 256),
            insert_attention_at=cfg.get("insert_attention_at", [1, 2, 3]),
            attention_kwargs=attention_kwargs,
        )

    elif model_type == "refinement_seg":
        return RefinementSegModel(
            encoder=cfg.get("encoder", "resnet50"),
            decoder=cfg.get("decoder", "unet"),
            refinement=cfg.get("refinement", "boundary_refine"),
            num_classes=cfg.get("num_classes", 1),
            pretrained=cfg.get("pretrained", True),
            decoder_channels=cfg.get("decoder_channels", 256),
        )

    elif model_type == "boundary_aware_seg":
        return BoundaryAwareSegModel(
            encoder=cfg.get("encoder", "resnet50"),
            decoder=cfg.get("decoder", "unet"),
            num_classes=cfg.get("num_classes", 1),
            pretrained=cfg.get("pretrained", True),
            decoder_channels=cfg.get("decoder_channels", 256),
        )

    elif model_type == "multi_task_seg":
        return MultiTaskSegModel(
            encoder=cfg.get("encoder", "resnet50"),
            num_classes=cfg.get("num_classes", 1),
            pretrained=cfg.get("pretrained", True),
            decoder_channels=cfg.get("decoder_channels", 256),
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
