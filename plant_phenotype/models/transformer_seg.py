from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# timm 中 swin_*_224 等骨干的预训练权重对应固定输入 224x224，需将输入缩放到该尺寸再送入骨干
SWIN_DEFAULT_INPUT_SIZE = 224


class SwinFPNSegmentation(nn.Module):
    """
    基于 timm Swin Transformer 骨干 + 简单 FPN 解码器的二分类分割模型。

    - 使用 timm 的 swin_tiny_patch4_window7_224 作为默认骨干（固定 224x224 输入）；
    - 任意分辨率输入会先下采样到 224x224 再进骨干，解码时上采样回输入分辨率，从而支持 512x512 等 patch 训练。
    - 提取 4 个尺度特征，经过 1x1 conv 统一通道数后做 FPN 融合；
    - 最终上采样到输入分辨率输出一个类别通道（配合 BCEWithLogitsLoss 使用）。
    """

    def __init__(
        self,
        backbone_name: str = "swin_tiny_patch4_window7_224",
        in_channels: int = 3,
        num_classes: int = 1,
        decoder_channels: int = 128,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        backbone_channels: List[int] = self.backbone.feature_info.channels()

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, decoder_channels, kernel_size=1) for c in backbone_channels]
        )

        self.output_convs = nn.ModuleList(
            [
                nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1)
                for _ in backbone_channels
            ]
        )

        self.classifier = nn.Conv2d(
            decoder_channels * len(backbone_channels), num_classes, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (x.shape[-2], x.shape[-1])
        # 骨干为 224x224 固定输入，将输入统一缩放到该尺寸再前向，避免 timm 尺寸断言错误
        if target_size != (SWIN_DEFAULT_INPUT_SIZE, SWIN_DEFAULT_INPUT_SIZE):
            x_backbone = F.interpolate(
                x, size=(SWIN_DEFAULT_INPUT_SIZE, SWIN_DEFAULT_INPUT_SIZE),
                mode="bilinear", align_corners=False
            )
        else:
            x_backbone = x

        feats: List[torch.Tensor] = self.backbone(x_backbone)

        # timm Swin 的 features_only 可能返回 (N, H, W, C)，Conv2d 需要 (N, C, H, W)
        backbone_chs = self.backbone.feature_info.channels()
        feats_nchw = []
        for f, c in zip(feats, backbone_chs):
            if f.dim() == 4 and f.shape[1] != c:
                # 第二维不是通道数，视为 (N,H,W,C) 转为 (N,C,H,W)
                f = f.permute(0, 3, 1, 2).contiguous()
            feats_nchw.append(f)
        feats = feats_nchw

        laterals = [lat_conv(f) for lat_conv, f in zip(self.lateral_convs, feats)]

        # 自顶向下 FPN 融合
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=False)
            laterals[i - 1] = laterals[i - 1] + up

        outs: List[torch.Tensor] = []
        for i, lat in enumerate(laterals):
            out = self.output_convs[i](lat)
            # 上采样到原始输入分辨率（支持 512x512 等任意 patch 尺寸）
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
            outs.append(out)

        # 拼接多尺度特征并输出最终 logits
        fused = torch.cat(outs, dim=1)
        logits = self.classifier(fused)
        return logits
