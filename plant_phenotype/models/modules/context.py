"""
上下文模块：ASPP, PSP Module, Self-Attention, CBAM 等
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


CONTEXT_MODULES = ["aspp", "psp", "self_attention", "cbam", "eca", "none"]


class ASPPConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        out = self.conv(x)
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256, dilations: tuple = (6, 12, 18)):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for d in dilations:
            modules.append(ASPPConv(in_channels, out_channels, d))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [conv(x) for conv in self.convs]
        return self.project(torch.cat(outs, dim=1))


class PSPModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 512, pool_sizes: tuple = (1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=ps),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), 1, bias=False),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True),
            )
            for ps in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        priors = [F.interpolate(stage(x), size=size, mode="bilinear", align_corners=False) for stage in self.stages]
        return self.bottleneck(torch.cat([x] + priors, dim=1))


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        inter_channels = max(in_channels // reduction, 1)
        self.query = nn.Conv2d(in_channels, inter_channels, 1)
        self.key = nn.Conv2d(in_channels, inter_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        query = self.query(x).view(N, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(N, -1, H * W)
        value = self.value(x).view(N, -1, H * W)
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(N, C, H, W)
        return self.gamma * out + x


class WindowSelfAttention(nn.Module):
    def __init__(self, in_channels: int, window_size: int = 8, reduction: int = 8):
        super().__init__()
        self.window_size = window_size
        inter_channels = max(in_channels // reduction, 1)
        self.query = nn.Conv2d(in_channels, inter_channels, 1)
        self.key = nn.Conv2d(in_channels, inter_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        w = self.window_size
        pad_h, pad_w = (w - H % w) % w, (w - W % w) % w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, Hp, Wp = x.size()
        nh, nw = Hp // w, Wp // w
        x_win = x.view(N, C, nh, w, nw, w).permute(0, 2, 4, 1, 3, 5).contiguous().view(N * nh * nw, C, w, w)
        query = self.query(x_win).view(N * nh * nw, -1, w * w).permute(0, 2, 1)
        key = self.key(x_win).view(N * nh * nw, -1, w * w)
        value = self.value(x_win).view(N * nh * nw, -1, w * w)
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(N * nh * nw, C, w, w)
        out = self.gamma * out + x_win
        out = out.view(N, nh, nw, C, w, w).permute(0, 3, 1, 4, 2, 5).contiguous().view(N, C, Hp, Wp)
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        return x * self.sa(x)


class ECAModule(nn.Module):
    def __init__(self, in_channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        import math
        t = int(abs((math.log2(in_channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * torch.sigmoid(y)


def build_context_module(name: str, in_channels: int, out_channels: Optional[int] = None, **kwargs) -> nn.Module:
    name = name.lower()
    out_channels = out_channels or in_channels
    if name == "aspp":
        return ASPP(in_channels, out_channels, kwargs.get("dilations", (6, 12, 18)))
    if name == "psp":
        return PSPModule(in_channels, out_channels, kwargs.get("pool_sizes", (1, 2, 3, 6)))
    if name == "self_attention":
        return WindowSelfAttention(in_channels, kwargs.get("window_size", 8)) if kwargs.get("window_size", 0) > 0 else SelfAttentionBlock(in_channels)
    if name == "cbam":
        return CBAM(in_channels)
    if name == "eca":
        return ECAModule(in_channels)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported context module: {name}. Supported: {CONTEXT_MODULES}")
