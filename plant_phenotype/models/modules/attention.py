"""
注意力模块：SE, CBAM, ECA, root_topology 等
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


ATTENTION_MODULES = [
    "se", "cbam", "eca", "coordinate", "dual", "non_local",
    "spatial", "multi_scale", "scse", "bam", "triplet",
    "root_topology", "direction_aware",
]


class SEModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1).expand_as(x)


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
        return x * self.sa(x * self.ca(x))


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


class CoordinateAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, in_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, in_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return x * torch.sigmoid(self.conv_h(x_h)) * torch.sigmoid(self.conv_w(x_w))


class PositionAttentionModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        query = self.query(x).view(N, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(N, -1, H * W)
        value = self.value(x).view(N, -1, H * W)
        attn = F.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(value, attn.permute(0, 2, 1))
        return self.gamma * out.view(N, C, H, W) + x


class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        query = x.view(N, C, -1)
        key = x.view(N, C, -1).permute(0, 2, 1)
        value = x.view(N, C, -1)
        attn = F.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(attn, value)
        return self.gamma * out.view(N, C, H, W) + x


class DualAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.pam = PositionAttentionModule(in_channels, reduction)
        self.cam = ChannelAttentionModule()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([self.pam(x), self.cam(x)], dim=1))


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 2, use_scale: bool = True):
        super().__init__()
        self.inter_channels = max(in_channels // reduction, 1)
        self.use_scale = use_scale
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.W = nn.Sequential(nn.Conv2d(self.inter_channels, in_channels, 1), nn.BatchNorm2d(in_channels))
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        g_x = self.g(x).view(N, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(N, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(N, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            f = f / (self.inter_channels ** 0.5)
        f = F.softmax(f, dim=-1)
        y = torch.matmul(f, g_x).permute(0, 2, 1).contiguous().view(N, self.inter_channels, H, W)
        return self.W(y) + x


class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels: int, scales: list = [1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // len(scales), 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels // len(scales)),
                nn.ReLU(inplace=True),
            )
            for _ in scales
        ])
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for scale, conv in zip(self.scales, self.convs):
            if scale == 1:
                feats.append(conv(x))
            else:
                h, w = x.shape[-2:]
                pooled = F.adaptive_avg_pool2d(x, (h // scale, w // scale))
                feats.append(F.interpolate(conv(pooled), size=(h, w), mode="bilinear", align_corners=False))
        return x * self.attention(torch.cat(feats, dim=1))


class scSE(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 2):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cSE(x) + x * self.sSE(x)


class BAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, dilation: int = 4):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.channel_att(x) + self.spatial_att(x))


class ChannelGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * torch.sigmoid(torch.cat([avg, max_], dim=1))


class SpatialGate(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))


class TripletAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.ch_att = ChannelGate()
        self.cw_att = ChannelGate()
        self.hw_att = SpatialGate(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.ch_att(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x2 = self.cw_att(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        x3 = self.hw_att(x)
        return (x1 + x2 + x3) / 3


class DirectionPredictor(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.conv(x), p=2, dim=1)


class DirectionAwareAttention(nn.Module):
    def __init__(self, in_channels: int, window_size: int = 8, sigma: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.dir_predictor = DirectionPredictor(in_channels)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        N, C, H, W = x.size()
        w = self.window_size
        pad_h, pad_w = (w - H % w) % w, (w - W % w) % w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, Hp, Wp = x.size()
        nh, nw = Hp // w, Wp // w
        dir_map = self.dir_predictor(x)
        qkv = self.qkv(x)
        qkv = qkv.view(N, 3, C, nh, w, nw, w).permute(0, 1, 3, 5, 4, 6, 2).contiguous().view(N, 3, nh * nw, w * w, C)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
        dir_win = dir_map.view(N, 2, nh, w, nw, w).permute(0, 2, 4, 3, 5, 1).contiguous().view(N, nh * nw, w * w, 2)
        Lq = w * w
        idx = torch.arange(Lq, device=x.device, dtype=torch.float32)
        qi, qj = idx // w, idx % w
        query_pos = torch.stack([qi, qj], dim=1)
        dk = query_pos.unsqueeze(0) - query_pos.unsqueeze(1)
        dk = dk.unsqueeze(0).unsqueeze(0).to(x.device)
        dir_q = dir_win.unsqueeze(3).unsqueeze(4)
        perp = dk[..., 0] * (-dir_q[..., 1]) + dk[..., 1] * dir_q[..., 0]
        gate = torch.exp(-(perp ** 2) / (2 * self.sigma ** 2))
        if gate.shape != scores.shape:
            gate = torch.ones_like(scores, device=scores.device, dtype=scores.dtype)
        attn = F.softmax(scores, dim=-1) * gate
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        out = torch.matmul(attn, v)
        out = out.view(N, nh, nw, w, w, C).permute(0, 5, 1, 3, 2, 4).contiguous().view(N, C, Hp, Wp)
        out = self.proj(out)
        out = out.view(N, C, -1).permute(0, 2, 1).contiguous()
        out = self.norm(out)
        out = out.permute(0, 2, 1).view(N, C, Hp, Wp)
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return x_in + out


def build_attention_module(name: str, in_channels: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "se":
        return SEModule(in_channels, kwargs.get("reduction", 16))
    if name == "cbam":
        return CBAM(in_channels, kwargs.get("reduction", 16), kwargs.get("kernel_size", 7))
    if name == "eca":
        return ECAModule(in_channels)
    if name == "coordinate":
        return CoordinateAttention(in_channels, kwargs.get("reduction", 32))
    if name == "dual":
        return DualAttention(in_channels, kwargs.get("reduction", 8))
    if name == "non_local":
        return NonLocalBlock(in_channels, kwargs.get("reduction", 2))
    if name == "spatial":
        return SpatialAttention(kwargs.get("kernel_size", 7))
    if name == "multi_scale":
        return MultiScaleAttention(in_channels, kwargs.get("scales", [1, 2, 4]))
    if name == "scse":
        return scSE(in_channels, kwargs.get("reduction", 2))
    if name == "bam":
        return BAM(in_channels, kwargs.get("reduction", 16))
    if name == "triplet":
        return TripletAttention(kwargs.get("kernel_size", 7))
    if name == "root_topology":
        from .root_topology_attention import build_root_topology_attention
        return build_root_topology_attention(
            in_channels=in_channels,
            window_size=kwargs.get("window_size", 8),
            reduction=kwargs.get("reduction", 8),
            use_topology_gate=kwargs.get("use_topology_gate", True),
        )
    if name == "direction_aware":
        return DirectionAwareAttention(
            in_channels, kwargs.get("window_size", 8), kwargs.get("sigma", 1.0)
        )
    raise ValueError(f"Unsupported attention module: {name}. Supported: {ATTENTION_MODULES}")
