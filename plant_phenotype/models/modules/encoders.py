"""
编码器模块：VGG, ResNet, EfficientNet, SegFormer, ConvNeXt, ViT, DeiT 等
"""
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False


ENCODERS = [
    "vgg16", "vgg19",
    "resnet18", "resnet34", "resnet50", "resnet101",
    "efficientnet-b0", "efficientnet-b4",
    "mobilenet_v2",
    "segformer-b0", "segformer-b1", "segformer-b2", "segformer-b3",
    "convnext_tiny", "convnext_small",
    "swin_tiny", "swin_small",
    "vit_tiny", "vit_small", "vit_base",
    "deit_tiny", "deit_small", "deit_base",
]

# timm 中 ViT/DeiT/Swin 的 *_224 变体为固定输入尺寸，需将输入缩放到 224 再进骨干
DEFAULT_VIT_SWIN_IMG_SIZE = 224


def _resize_to_backbone_if_needed(x: torch.Tensor, size: int = DEFAULT_VIT_SWIN_IMG_SIZE) -> torch.Tensor:
    """当输入不是 size×size 时缩放到 size×size，避免 timm 骨干的尺寸断言错误。"""
    h, w = x.shape[-2], x.shape[-1]
    if h != size or w != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x


class VGGEncoder(nn.Module):
    """VGG 编码器，提取多尺度特征"""

    def __init__(self, variant: str = "vgg16", pretrained: bool = True):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm required for VGGEncoder")

        self.backbone = timm.create_model(
            variant, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4)
        )
        self.channels = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)


class SegFormerEncoder(nn.Module):
    """SegFormer 编码器 (MiT backbone)"""

    def __init__(self, variant: str = "mit_b0", pretrained: bool = True):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm required for SegFormerEncoder")

        self.backbone = timm.create_model(
            variant, pretrained=pretrained, features_only=True
        )
        self.channels = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt 编码器"""

    def __init__(self, variant: str = "convnext_tiny", pretrained: bool = True):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm required for ConvNeXtEncoder")

        self.backbone = timm.create_model(
            variant, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3)
        )
        self.channels = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)


class SwinEncoder(nn.Module):
    """Swin Transformer 编码器（固定 224×224 输入，非 224 时自动缩放）"""

    def __init__(self, variant: str = "swin_tiny_patch4_window7_224", pretrained: bool = True):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm required for SwinEncoder")
        self.backbone = timm.create_model(
            variant, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3)
        )
        self.channels = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = _resize_to_backbone_if_needed(x)
        return self.backbone(x)


class ViTEncoder(nn.Module):
    """Vision Transformer (ViT) 编码器（固定 224×224 输入，非 224 时自动缩放）"""

    def __init__(self, variant: str = "vit_tiny_patch16_224", pretrained: bool = True):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm required for ViTEncoder")
        self.backbone = timm.create_model(
            variant, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3)
        )
        self.channels = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = _resize_to_backbone_if_needed(x)
        return self.backbone(x)


class DeiTEncoder(nn.Module):
    """Data-efficient image Transformer (DeiT) 编码器（固定 224×224 输入，非 224 时自动缩放）"""

    def __init__(self, variant: str = "deit_tiny_patch16_224", pretrained: bool = True):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm required for DeiTEncoder")
        self.backbone = timm.create_model(
            variant, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3)
        )
        self.channels = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = _resize_to_backbone_if_needed(x)
        return self.backbone(x)


class SMPEncoder(nn.Module):
    """segmentation_models_pytorch 编码器包装"""

    def __init__(self, encoder_name: str = "resnet34", weights: str = "imagenet", in_channels: int = 3):
        super().__init__()
        if not HAS_SMP:
            raise ImportError("segmentation_models_pytorch required")

        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels, depth=5, weights=weights
        )
        self.channels = self.encoder.out_channels[1:]  # 跳过第一层（stem）

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.encoder(x)
        return features[1:]  # 跳过 stem 输出


def build_encoder(
    name: str,
    pretrained: bool = True,
    in_channels: int = 3,
) -> Tuple[nn.Module, List[int]]:
    """
    构建编码器并返回 (encoder, channels_list)

    Returns:
        encoder: 编码器模块
        channels: 各层输出通道数列表
    """
    name = name.lower()

    if name.startswith("vgg"):
        enc = VGGEncoder(name, pretrained)
        return enc, enc.channels

    elif name.startswith("segformer") or name.startswith("mit"):
        variant_map = {
            "segformer-b0": "mit_b0", "segformer-b1": "mit_b1",
            "segformer-b2": "mit_b2", "segformer-b3": "mit_b3",
        }
        variant = variant_map.get(name, name)
        pvt_fallback = {
            "mit_b0": "pvt_v2_b0", "mit_b1": "pvt_v2_b1",
            "mit_b2": "pvt_v2_b2", "mit_b3": "pvt_v2_b3",
        }
        try:
            enc = SegFormerEncoder(variant, pretrained)
        except RuntimeError as e:
            if "Unknown model" in str(e) and variant in pvt_fallback:
                import warnings
                warnings.warn(
                    f"timm 中无 {variant}，使用 PVT v2 骨干 {pvt_fallback[variant]} 作为替代。",
                    UserWarning,
                    stacklevel=2,
                )
                enc = SegFormerEncoder(pvt_fallback[variant], pretrained)
            else:
                raise
        return enc, enc.channels

    elif name.startswith("convnext"):
        enc = ConvNeXtEncoder(name, pretrained)
        return enc, enc.channels

    elif name.startswith("swin"):
        variant_map = {
            "swin_tiny": "swin_tiny_patch4_window7_224",
            "swin_small": "swin_small_patch4_window7_224",
        }
        variant = variant_map.get(name, name)
        enc = SwinEncoder(variant, pretrained)
        return enc, enc.channels

    elif name.startswith("vit"):
        variant_map = {
            "vit_tiny": "vit_tiny_patch16_224",
            "vit_small": "vit_small_patch16_224",
            "vit_base": "vit_base_patch16_224",
        }
        variant = variant_map.get(name, name)
        enc = ViTEncoder(variant, pretrained)
        return enc, enc.channels

    elif name.startswith("deit"):
        variant_map = {
            "deit_tiny": "deit_tiny_patch16_224",
            "deit_small": "deit_small_patch16_224",
            "deit_base": "deit_base_patch16_224",
        }
        variant = variant_map.get(name, name)
        enc = DeiTEncoder(variant, pretrained)
        return enc, enc.channels

    else:
        weights = "imagenet" if pretrained else None
        enc = SMPEncoder(name, weights, in_channels)
        return enc, list(enc.channels)
