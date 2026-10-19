"""
MaskAttn-UNet: 基于 Mask 注意力的 U-Net 分割模型

参考: MaskAttn-UNet: A Mask Attention-Driven Framework for Universal Low-Resolution Image Segmentation
https://github.com/Belis0811/MaskUnet

适配本项目的二分类根系分割，支持可配置输入尺寸（如 512x512）。
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):
    """
    窗口注意力：在局部窗口内做自注意力，显存从 O((H*W)²) 降到 O(num_windows * win²)。
    类似 Swin Transformer 的 window attention。
    """

    def __init__(self, channels: int, window_size: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, H, W) -> (N, C, H, W)
        """
        N, C, H, W = x.size()
        w = self.window_size
        # 若尺寸不能整除，先 pad
        pad_h = (w - H % w) % w
        pad_w = (w - W % w) % w
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            _, _, H, W = x.size()

        nh, nw = H // w, W // w
        # (N, C, nh, w, nw, w) -> (N, nh*nw, w*w, C)
        x_win = x.view(N, C, nh, w, nw, w).permute(0, 2, 4, 3, 5, 1).contiguous()
        x_win = x_win.view(N, nh * nw, w * w, C)

        Q = self.query(x_win)
        K = self.key(x_win)
        V = self.value(x_win)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (C ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = self.norm(out + x_win)

        # (N, nh*nw, w*w, C) -> (N, C, H, W)
        out = out.view(N, nh, nw, w, w, C).permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(N, C, H, W)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, : H - pad_h, : W - pad_w]
        return out




class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual
        mid_channels = mid_channels or out_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return F.gelu(x + self.conv_block(x))
        return self.conv_block(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = True,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.attention = (
            WindowAttention(out_channels, window_size=window_size) if use_attention else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.attention(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = True,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels, in_channels // 2),
            nn.BatchNorm2d(out_channels),
        )
        self.attention = (
            WindowAttention(out_channels, window_size=window_size) if use_attention else nn.Identity()
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        x = self.attention(x)
        return x


class MaskAttnUNet(nn.Module):
    """
    MaskAttn-UNet 用于二分类语义分割。
    支持任意 2 的幂次输入尺寸（如 128, 256, 512）。
    使用窗口注意力 (window_size=8) 降低显存，适合大分辨率训练。
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.initial = ConvBlock(in_channels, base_channels)

        self.down1 = DownBlock(base_channels, base_channels * 2, window_size=window_size)   # 1/2
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, window_size=window_size)  # 1/4
        self.down3 = DownBlock(base_channels * 4, base_channels * 4, window_size=window_size)  # 1/8

        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8),
            ConvBlock(base_channels * 8, base_channels * 8),
            ConvBlock(base_channels * 8, base_channels * 4),
        )
        self.dropout = nn.Dropout(0.3)

        self.up1 = UpBlock(base_channels * 8, base_channels * 2, window_size=window_size)   # 512 -> 256
        self.up2 = UpBlock(base_channels * 4, base_channels, window_size=window_size)       # 256 -> 128
        self.up3 = UpBlock(base_channels * 2, base_channels, window_size=window_size)     # 128 -> 64

        self.final = nn.Sequential(
            nn.Conv2d(base_channels, num_classes, 1),
            nn.BatchNorm2d(num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.initial(x)       # (N, 64, H, W)
        x2 = self.down1(x1)        # (N, 128, H/2, W/2)
        x3 = self.down2(x2)        # (N, 256, H/4, W/4)
        x4 = self.down3(x3)        # (N, 256, H/8, W/8)

        x4 = self.bottleneck(x4)   # (N, 256, H/8, W/8)

        x = self.up1(x4, x3)
        x = self.dropout(x)
        x = self.up2(x, x2)
        x = self.dropout(x)
        x = self.up3(x, x1)

        return self.final(x)
