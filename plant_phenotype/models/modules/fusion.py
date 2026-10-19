"""
特征融合模块：FPN, BiFPN, ASFF 等
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


FUSION_MODULES = ["fpn", "bifpn", "asff", "concat", "add"]


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FPN(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.output_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])
        self.out_channels = out_channels

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [lat_conv(f) for lat_conv, f in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=False)
            laterals[i - 1] = laterals[i - 1] + up
        return [out_conv(lat) for out_conv, lat in zip(self.output_convs, laterals)]


class BiFPNLayer(nn.Module):
    def __init__(self, channels: int, num_levels: int):
        super().__init__()
        self.num_levels = num_levels
        self.td_weights = nn.ParameterList([nn.Parameter(torch.ones(2)) for _ in range(num_levels - 1)])
        self.bu_weights = nn.ParameterList([nn.Parameter(torch.ones(2 if i == 0 else 3)) for i in range(num_levels - 1)])
        self.td_convs = nn.ModuleList([ConvBNReLU(channels, channels) for _ in range(num_levels - 1)])
        self.bu_convs = nn.ModuleList([ConvBNReLU(channels, channels) for _ in range(num_levels - 1)])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        eps = 1e-4
        td_features = [features[-1]]
        for i in range(self.num_levels - 2, -1, -1):
            w = F.relu(self.td_weights[self.num_levels - 2 - i])
            w = w / (w.sum() + eps)
            up = F.interpolate(td_features[0], size=features[i].shape[-2:], mode="bilinear", align_corners=False)
            td_features.insert(0, self.td_convs[self.num_levels - 2 - i](w[0] * features[i] + w[1] * up))
        bu_features = [td_features[0]]
        for i in range(1, self.num_levels):
            w = F.relu(self.bu_weights[i - 1])
            w = w / (w.sum() + eps)
            down = F.interpolate(bu_features[-1], size=td_features[i].shape[-2:], mode="bilinear", align_corners=False)
            fused = (w[0] * td_features[i] + w[1] * down) if i == self.num_levels - 1 else (w[0] * features[i] + w[1] * td_features[i] + w[2] * down)
            bu_features.append(self.bu_convs[i - 1](fused))
        return bu_features


class BiFPN(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int = 256, num_layers: int = 2):
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels
        self.input_convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.bifpn_layers = nn.ModuleList([BiFPNLayer(out_channels, self.num_levels) for _ in range(num_layers)])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        features = [conv(f) for conv, f in zip(self.input_convs, features)]
        for layer in self.bifpn_layers:
            features = layer(features)
        return features


class ASFF(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels
        self.convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.weight_convs = nn.ModuleList([nn.Conv2d(out_channels * self.num_levels, self.num_levels, 1) for _ in range(self.num_levels)])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        features = [conv(f) for conv, f in zip(self.convs, features)]
        outputs = []
        for i, target_f in enumerate(features):
            target_size = target_f.shape[-2:]
            resized = [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False) for f in features]
            weights = F.softmax(self.weight_convs[i](torch.cat(resized, dim=1)), dim=1)
            outputs.append(sum(w.unsqueeze(1) * f for w, f in zip(weights.unbind(1), resized)))
        return outputs


class ConcatFusion(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(sum(in_channels_list), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        resized = [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False) for f in features]
        return self.conv(torch.cat(resized, dim=1))


class AddFusion(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.out_channels = out_channels

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        features = [conv(f) for conv, f in zip(self.convs, features)]
        resized = [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False) for f in features]
        return sum(resized)


def build_fusion_module(name: str, in_channels_list: List[int], out_channels: int = 256, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "fpn":
        return FPN(in_channels_list, out_channels)
    if name == "bifpn":
        return BiFPN(in_channels_list, out_channels, kwargs.get("num_layers", 2))
    if name == "asff":
        return ASFF(in_channels_list, out_channels)
    if name == "concat":
        return ConcatFusion(in_channels_list, out_channels)
    if name == "add":
        return AddFusion(in_channels_list, out_channels)
    raise ValueError(f"Unsupported fusion module: {name}. Supported: {FUSION_MODULES}")
