"""
DINOv3 (Meta) Encoder + 多层特征对齐 + MLP Decoder 分割模型。
将 DINOv3 的若干层 hidden_states 对齐到统一分辨率与通道后拼接，经 MLP 直接输出 mask logits。
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    AutoModel = None
    AutoConfig = None
    HAS_TRANSFORMERS = False

# 类名为 DINOv3ViTConfig（大写 O），可选用于 gated 仓库不可用时的本地 config
DINOv3ViTConfig = None
if HAS_TRANSFORMERS:
    try:
        from transformers.models.dinov3_vit import DINOv3ViTConfig  # noqa: F401
    except ImportError:
        DINOv3ViTConfig = None


# DINOv3 ViT 默认输入尺寸（与 config.image_size 一致，用于 resize）
DINOV3_DEFAULT_IMAGE_SIZE = 224


def _patch_tokens_to_2d(
    x: torch.Tensor,
    num_register_tokens: int,
    patch_size: int,
    image_size: int,
) -> torch.Tensor:
    """
    将 (N, 1+num_reg+num_patches, C) 的 hidden_state 取 patch 部分并 reshape 为 (N, C, H, W)。
    """
    # patch 从 index 1+num_register_tokens 开始
    n_reg = num_register_tokens
    patch_tokens = x[:, 1 + n_reg :, :]  # (N, num_patches, C)
    n, l, c = patch_tokens.shape
    h = w = image_size // patch_size
    if l != h * w:
        # 非方形时用 sqrt 近似，或按 config 的 image_size 推算
        h = int(l ** 0.5)
        w = l // h if h > 0 else 1
    return patch_tokens.permute(0, 2, 1).unflatten(2, (h, w))  # (N, C, h, w)


class DINOv3MLPSegmentation(nn.Module):
    """
    DINOv3 (HuggingFace) 作为 Encoder，多层特征对齐到统一分辨率与通道后，
    用 MLP Decoder 直接输出 mask logits。
    """

    def __init__(
        self,
        encoder_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        num_classes: int = 1,
        decoder_embed_dim: int = 256,
        in_channels: int = 3,
        encoder_layer_indices: Optional[List[int]] = None,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required for DINOv3. Install with: pip install transformers")

        self.encoder_name = encoder_name
        self.decoder_embed_dim = decoder_embed_dim
        self.num_classes = num_classes

        # 加载 DINOv3 ViT（AutoModel 会解析为 DINOv3ViTModel）
        # DINOv3 为 gated repo，无授权时用本地默认 config 构建（随机初始化）
        try:
            config = AutoConfig.from_pretrained(encoder_name)
        except (OSError, Exception) as e:
            if "gated" in str(e).lower() or "401" in str(e) or "access" in str(e).lower():
                if DINOv3ViTConfig is not None:
                    config = DINOv3ViTConfig(
                        patch_size=16,
                        hidden_size=384,
                        num_hidden_layers=12,
                        num_register_tokens=4,
                        image_size=224,
                    )
                    self.backbone = AutoModel.from_config(config)
                else:
                    raise
            else:
                raise
        else:
            if pretrained:
                self.backbone = AutoModel.from_pretrained(encoder_name)
            else:
                self.backbone = AutoModel.from_config(config)
        self.config = self.backbone.config

        self.patch_size = getattr(self.config, "patch_size", 16)
        self.num_register_tokens = getattr(self.config, "num_register_tokens", 4)
        self.hidden_size = self.config.hidden_size
        self.image_size = getattr(self.config, "image_size", DINOV3_DEFAULT_IMAGE_SIZE)
        self.num_hidden_layers = self.config.num_hidden_layers

        # 选取的层索引（默认均匀取 4 层）
        if encoder_layer_indices is None:
            step = max(1, self.num_hidden_layers // 4)
            self.layer_indices = [min(i * step, self.num_hidden_layers - 1) for i in range(4)]
        else:
            self.layer_indices = [min(i, self.num_hidden_layers - 1) for i in encoder_layer_indices]

        num_layers = len(self.layer_indices)
        # 每层通道均为 hidden_size
        in_channels_list = [self.hidden_size] * num_layers

        # 特征对齐：每层 1x1 到 embed_dim，再统一 interpolate
        self.linear_fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, decoder_embed_dim, 1),
                nn.BatchNorm2d(decoder_embed_dim),
            )
            for c in in_channels_list
        ])
        # MLP head：拼接后 -> 1x1 -> BN -> ReLU -> 1x1 -> num_classes
        self.mlp_head = nn.Sequential(
            nn.Conv2d(decoder_embed_dim * num_layers, decoder_embed_dim, 1),
            nn.BatchNorm2d(decoder_embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_embed_dim, num_classes, 1),
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """冻结 DINOv3 骨干，仅训练 linear_fuse 与 mlp_head。"""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, H, W)，与 dataloader 一致（已做 normalize，如 0.5/0.5 或 ImageNet）。
        返回: (N, num_classes, H, W) logits。
        """
        target_size = (x.shape[-2], x.shape[-1])
        # 将输入 resize 到 backbone 的 image_size，避免尺寸不匹配
        if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
            x_in = F.interpolate(
                x, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        else:
            x_in = x

        # DINOv3 forward，取多层 hidden_states
        outputs = self.backbone(
            pixel_values=x_in,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states  # tuple: (embed, layer0, layer1, ...)

        features: List[torch.Tensor] = []
        for idx in self.layer_indices:
            # hidden_states[0] 是 embedding 输出，[1:] 是各层
            h = hidden_states[idx + 1]  # +1 因为 hidden_states[0] 是 embed
            feat_2d = _patch_tokens_to_2d(
                h,
                self.num_register_tokens,
                self.patch_size,
                self.image_size,
            )
            features.append(feat_2d)

        # 统一分辨率与通道后拼接（与 SegFormerHead 一致）
        target_h, target_w = features[-1].shape[-2], features[-1].shape[-1]
        outs = []
        for f, fuse in zip(features, self.linear_fuse):
            f = fuse(f)
            if f.shape[-2] != target_h or f.shape[-1] != target_w:
                f = F.interpolate(f, size=(target_h, target_w), mode="bilinear", align_corners=False)
            outs.append(f)
        x_fused = torch.cat(outs, dim=1)
        logits = self.mlp_head(x_fused)

        if logits.shape[-2] != target_size[0] or logits.shape[-1] != target_size[1]:
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return logits
