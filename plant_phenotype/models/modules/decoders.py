"""
解码器模块：UNet, SegFormer Head, UPerNet, LawinASPP 等
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


DECODERS = ["unet", "segformer_head", "upernet", "simple", "lawin"]


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetDecoder(nn.Module):
    """UNet 风格解码器（跳跃连接 + 上采样）"""
    def __init__(self, encoder_channels: List[int], decoder_channels: List[int] = None, num_classes: int = 1):
        super().__init__()
        encoder_channels = encoder_channels[::-1]
        n_blocks = len(encoder_channels) - 1
        if decoder_channels is None:
            decs = [c // 2 for c in encoder_channels[:-1]]
            decoder_channels = (decs[:-1] + [64]) if len(decs) > 1 else [64]
        elif len(decoder_channels) > n_blocks:
            decoder_channels = decoder_channels[:n_blocks]
        self.blocks = nn.ModuleList()
        in_ch = encoder_channels[0]
        for i, (enc_ch, dec_ch) in enumerate(zip(encoder_channels[1:], decoder_channels)):
            self.blocks.append(
                nn.Sequential(
                    ConvBNReLU(in_ch + enc_ch, dec_ch),
                    ConvBNReLU(dec_ch, dec_ch),
                )
            )
            in_ch = dec_ch
        self.head = nn.Conv2d(in_ch, num_classes, 1)

    def forward(self, features: List[torch.Tensor], target_size: tuple = None) -> torch.Tensor:
        features = features[::-1]
        x = features[0]
        for i, block in enumerate(self.blocks):
            x = F.interpolate(x, size=features[i + 1].shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, features[i + 1]], dim=1)
            x = block(x)
        x = self.head(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class SegFormerHead(nn.Module):
    """SegFormer MLP 解码器"""
    def __init__(self, in_channels_list: List[int], embed_dim: int = 256, num_classes: int = 1):
        super().__init__()
        self.linear_fuse = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, embed_dim, 1), nn.BatchNorm2d(embed_dim))
            for c in in_channels_list
        ])
        self.linear_pred = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels_list), embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, num_classes, 1),
        )

    def forward(self, features: List[torch.Tensor], target_size: tuple = None) -> torch.Tensor:
        target = features[0].shape[-2:]
        outs = []
        for f, fuse in zip(features, self.linear_fuse):
            f = fuse(f)
            if f.shape[-2:] != target:
                f = F.interpolate(f, size=target, mode="bilinear", align_corners=False)
            outs.append(f)
        x = torch.cat(outs, dim=1)
        x = self.linear_pred(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class PSPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_scales: tuple = (1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for ps in pool_scales
        ])
        self.bottleneck = ConvBNReLU(in_channels + out_channels * len(pool_scales), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        psp_outs = [x]
        for stage in self.stages:
            psp_outs.append(F.interpolate(stage(x), size=size, mode="bilinear", align_corners=False))
        return self.bottleneck(torch.cat(psp_outs, dim=1))


class UPerNetDecoder(nn.Module):
    """UPerNet 解码器"""
    def __init__(self, in_channels_list: List[int], channels: int = 256, num_classes: int = 1, pool_scales: tuple = (1, 2, 3, 6)):
        super().__init__()
        self.psp = PSPPooling(in_channels_list[-1], channels, pool_scales)
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, channels, 1) for c in in_channels_list[:-1]])
        self.fpn_convs = nn.ModuleList([ConvBNReLU(channels, channels) for _ in in_channels_list[:-1]])
        self.fusion_conv = ConvBNReLU(channels * len(in_channels_list), channels)
        self.head = nn.Conv2d(channels, num_classes, 1)

    def forward(self, features: List[torch.Tensor], target_size: tuple = None) -> torch.Tensor:
        psp_out = self.psp(features[-1])
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, features[:-1])]
        laterals.append(psp_out)
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=False
            )
        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals[:-1])]
        fpn_outs.append(laterals[-1])
        target = features[0].shape[-2:]
        upsampled = [F.interpolate(f, size=target, mode="bilinear", align_corners=False) for f in fpn_outs]
        x = torch.cat(upsampled, dim=1)
        x = self.fusion_conv(x)
        x = self.head(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNReLU(in_channels, in_channels // 2),
            ConvBNReLU(in_channels // 2, in_channels // 4),
            nn.Conv2d(in_channels // 4, num_classes, 1),
        )

    def forward(self, features: List[torch.Tensor], target_size: tuple = None) -> torch.Tensor:
        x = features[-1] if isinstance(features, list) else features
        x = self.head(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class LawinBlock(nn.Module):
    def __init__(self, channels: int, window_sizes: List[int] = [2, 4, 8]):
        super().__init__()
        self.window_sizes = window_sizes
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        qkv = self.qkv(x).reshape(N, 3, C, H, W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        outputs = []
        for ws in self.window_sizes:
            if ws > 1:
                q_pool = F.avg_pool2d(q, ws, ws)
                k_pool = F.avg_pool2d(k, ws, ws)
                v_pool = F.avg_pool2d(v, ws, ws)
            else:
                q_pool, k_pool, v_pool = q, k, v
            Hp, Wp = q_pool.shape[-2:]
            q_flat = q_pool.view(N, C, -1).permute(0, 2, 1)
            k_flat = k_pool.view(N, C, -1)
            v_flat = v_pool.view(N, C, -1).permute(0, 2, 1)
            attn = F.softmax(q_flat @ k_flat / (C ** 0.5), dim=-1)
            out = (attn @ v_flat).permute(0, 2, 1).view(N, C, Hp, Wp)
            if ws > 1:
                out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
            outputs.append(out)
        out = sum(outputs) / len(outputs)
        out = self.proj(out)
        out = out + x
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)
        return out


class LawinHead(nn.Module):
    def __init__(self, in_channels_list: List[int], embed_dim: int = 256, num_classes: int = 1):
        super().__init__()
        self.in_convs = nn.ModuleList([nn.Conv2d(c, embed_dim, 1) for c in in_channels_list])
        self.lawin_blocks = nn.ModuleList([LawinBlock(embed_dim, window_sizes=[2, 4, 8]) for _ in in_channels_list])
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels_list), embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features: List[torch.Tensor], target_size: tuple = None) -> torch.Tensor:
        target = features[0].shape[-2:]
        outs = []
        for f, conv, lawin in zip(features, self.in_convs, self.lawin_blocks):
            f = conv(f)
            f = lawin(f)
            if f.shape[-2:] != target:
                f = F.interpolate(f, size=target, mode="bilinear", align_corners=False)
            outs.append(f)
        x = torch.cat(outs, dim=1)
        x = self.fusion(x)
        x = self.head(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


def build_decoder(name: str, in_channels_list: List[int], num_classes: int = 1, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "unet":
        return UNetDecoder(in_channels_list, kwargs.get("decoder_channels"), num_classes)
    elif name == "segformer_head":
        return SegFormerHead(in_channels_list, kwargs.get("embed_dim", 256), num_classes)
    elif name == "upernet":
        return UPerNetDecoder(in_channels_list, kwargs.get("channels", 256), num_classes)
    elif name == "simple":
        return SimpleDecoder(in_channels_list[-1], num_classes)
    elif name == "lawin":
        return LawinHead(in_channels_list, kwargs.get("embed_dim", 256), num_classes)
    raise ValueError(f"Unsupported decoder: {name}. Supported: {DECODERS}")
