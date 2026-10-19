from copy import deepcopy
from typing import Optional, Dict, Any

import segmentation_models_pytorch as smp
import torch.nn as nn

from .transformer_seg import SwinFPNSegmentation
from .mask_attn_unet import MaskAttnUNet
from .modular_seg import ModularSegModel, build_modular_model
from .advanced_seg import (
    AttentionSegModel, RefinementSegModel,
    BoundaryAwareSegModel, MultiTaskSegModel, build_advanced_model
)
from .dinov3_mlp_seg import DINOv3MLPSegmentation
from .freeze import freeze_encoder_stages
from .mobile_sam_seg import MobileSAMSeg


SUPPORTED_MODELS = [
    # smp 预定义模型
    "unet",
    "unet++",
    "fpn",
    "deeplabv3",
    "deeplabv3+",
    "pspnet",
    "linknet",
    "pan",
    "segformer",
    # 自定义模型
    "swin_fpn",
    "mask_attn_unet",
    # 模块化模型（支持自由组合）
    "modular",
    # 高级模型（注意力 + 边界细化）
    "attention_seg",
    "refinement_seg",
    "boundary_aware_seg",
    "multi_task_seg",
    # DINOv3 + MLP
    "dinov3_mlp",
    # MobileSAM TinyViT，空提示整图分割
    "mobile_sam",
]

# 模块化模型可用组件
MODULAR_COMPONENTS = ModularSegModel.get_available_components()

# 高级模型说明
ADVANCED_MODELS_INFO = {
    "attention_seg": "编码器各层加注意力模块（SE/CBAM/Coordinate等）",
    "refinement_seg": "粗预测 + 边界细化两阶段模型",
    "boundary_aware_seg": "联合学习语义分割和边缘检测",
    "multi_task_seg": "多任务学习（分割+边缘+距离图）",
}


def build_model_from_config(
    model_cfg: Dict[str, Any],
    *,
    initialize_encoder: bool = True,
    checkpoint: Optional[str] = None,
) -> nn.Module:
    """Use the same architecture configuration for training and reloading.

    Reloading a full state dict needs the trained architecture, but no
    pretrained encoder weights. Copy nested options because builders may
    supply defaults in place.
    """
    cfg = deepcopy(model_cfg)
    name = cfg["name"].lower()
    custom_names = {
        "modular", "attention_seg", "refinement_seg", "boundary_aware_seg",
        "multi_task_seg", "dinov3_mlp", "mobile_sam", "mobilesam",
    }
    modular_cfg = None
    if name in custom_names:
        default_encoder = (
            "facebook/dinov3-vits16-pretrain-lvd1689m"
            if name == "dinov3_mlp" else "resnet50"
        )
        cfg.setdefault("encoder", cfg.get("encoder_name", default_encoder))
        # The training entry has always taken these from name/classes.
        # Old checkpoints retain the raw config, including any stale aliases.
        cfg["type"] = name
        cfg["num_classes"] = cfg.get("classes", 1)
        cfg["pretrained"] = (
            cfg.get("pretrained", cfg.get("encoder_weights") is not None)
            if initialize_encoder else False
        )
        modular_cfg = cfg
    return build_model(
        name=name,
        encoder_name=cfg.get("encoder_name", "resnet34"),
        encoder_weights=cfg.get("encoder_weights", "imagenet") if initialize_encoder else None,
        in_channels=cfg.get("in_channels", 3),
        classes=cfg.get("classes", 1),
        window_size=cfg.get("window_size", 8),
        modular_cfg=modular_cfg,
        checkpoint=checkpoint,
        freeze_stages=cfg.get("freeze_encoder_stages") or None,
    )


def build_model(
    name: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    window_size: int = 8,
    modular_cfg: Optional[Dict[str, Any]] = None,
    checkpoint: Optional[str] = None,
    freeze_stages: Optional[list] = None,
) -> nn.Module:
    """
    构建语义分割模型。

    - 基于 segmentation_models_pytorch 的经典 CNN 模型：unet / unet++ / deeplabv3+ 等；
    - 基于 timm Swin Transformer 骨干的模型：swin_fpn；
    - 模块化模型（modular）：自由组合编码器、上下文模块、融合方式、解码器。

    推荐：
    - 精细边界、细长根系：unet / unet++ / fpn
    - 大感受野、复杂背景：deeplabv3 / deeplabv3+
    - 结合 Transformer 上下文能力：swin_fpn
    - 自由组合实验：modular

    模块化模型配置示例（modular_cfg）:
        {
            "encoder": "segformer-b2",
            "decoder": "segformer_head",
            "context": "aspp",
            "fusion": "fpn",
            "decoder_channels": 256,
        }

    可用组件:
        - 编码器: vgg16/19, resnet18/34/50/101, efficientnet-b0/b4, segformer-b0~b3, swin_tiny/small, convnext_tiny/small
        - 解码器: unet, segformer_head, upernet, lawin, simple
        - 上下文: aspp, psp, self_attention, cbam, eca, none
        - 融合: fpn, bifpn, asff, concat, add, none
    """
    name = name.lower()

    # 模块化模型
    if name == "modular":
        cfg = modular_cfg or {}
        cfg.setdefault("in_channels", in_channels)
        cfg.setdefault("classes", classes)
        cfg.setdefault("encoder", encoder_name)
        cfg.setdefault("pretrained", encoder_weights is not None)
        return build_modular_model(cfg)

    # 高级模型（注意力 + 边界细化）
    if name in ["attention_seg", "refinement_seg", "boundary_aware_seg", "multi_task_seg"]:
        cfg = modular_cfg or {}
        cfg.setdefault("type", name)
        cfg.setdefault("encoder", encoder_name)
        cfg.setdefault("num_classes", classes)
        cfg.setdefault("pretrained", encoder_weights is not None)
        cfg.setdefault("decoder_channels", 256)
        return build_advanced_model(cfg)

    if name in ("mobile_sam", "mobilesam"):
        cfg = modular_cfg or {}
        model = MobileSAMSeg(
            checkpoint=checkpoint,
            encoder_size=int(cfg.get("encoder_size", 1024)),
        )
        return _maybe_freeze(model, freeze_stages)

    # DINOv3 + MLP 分割
    if name == "dinov3_mlp":
        cfg = modular_cfg or {}
        return DINOv3MLPSegmentation(
            encoder_name=cfg.get("encoder", encoder_name or "facebook/dinov3-vits16-pretrain-lvd1689m"),
            num_classes=cfg.get("classes", classes),
            decoder_embed_dim=cfg.get("decoder_embed_dim", 256),
            in_channels=cfg.get("in_channels", in_channels),
            encoder_layer_indices=cfg.get("encoder_layer_indices"),
            pretrained=encoder_weights is not None if cfg.get("pretrained") is None else cfg.get("pretrained", True),
            freeze_backbone=cfg.get("freeze_backbone", False),
        )

    if name == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    elif name in ("unet++", "unetplusplus", "unet_plus_plus"):
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    elif name == "fpn":
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    elif name == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    elif name in ("deeplabv3+", "deeplabv3plus", "deeplabv3_plus"):
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    elif name == "pspnet":
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    elif name == "linknet":
        model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    elif name == "segformer":
        model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    elif name == "pan":
        model = smp.PAN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
    elif name == "swin_fpn":
        backbone_name = encoder_name or "swin_tiny_patch4_window7_224"
        model = SwinFPNSegmentation(
            backbone_name=backbone_name,
            in_channels=in_channels,
            num_classes=classes,
            decoder_channels=128,
            pretrained=encoder_weights is not None,
        )
    elif name == "mask_attn_unet":
        base_ch = 64
        if encoder_name and "resnet50" in encoder_name.lower():
            base_ch = 96
        elif encoder_name and "resnet101" in encoder_name.lower():
            base_ch = 128
        model = MaskAttnUNet(
            in_channels=in_channels,
            num_classes=classes,
            base_channels=base_ch,
            window_size=window_size,
        )
    else:
        raise ValueError(f"Unsupported model name: {name}. Supported: {SUPPORTED_MODELS}")

    return _maybe_freeze(model, freeze_stages)


def _maybe_freeze(model: nn.Module, stages: Optional[list]) -> nn.Module:
    if stages:
        freeze_encoder_stages(model, stages)
    return model
