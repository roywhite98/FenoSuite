"""
边界增强与细化模块
"""
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


REFINEMENT_MODULES = [
    "edge_detection", "boundary_refine", "detail_aggregate",
    "gated_fusion", "residual_refine", "multi_scale_refine"
]


class EdgeDetectionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 4.0)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = torch.mean(x, dim=1, keepdim=True) if x.size(1) > 1 else x
        edge_x = F.conv2d(gray, self.sobel_x.repeat(gray.size(1), 1, 1, 1), padding=1, groups=gray.size(1))
        edge_y = F.conv2d(gray, self.sobel_y.repeat(gray.size(1), 1, 1, 1), padding=1, groups=gray.size(1))
        return torch.sigmoid(self.edge_conv(x) + torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6))


class BoundaryRefinementModule(nn.Module):
    def __init__(self, coarse_channels: int, detail_channels: int, out_channels: int = 1):
        super().__init__()
        self.coarse_conv = nn.Sequential(nn.Conv2d(coarse_channels, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.detail_conv = nn.Sequential(nn.Conv2d(detail_channels, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, out_channels, 1))

    def forward(self, coarse: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
        if coarse.shape[-2:] != detail.shape[-2:]:
            coarse = F.interpolate(coarse, size=detail.shape[-2:], mode='bilinear', align_corners=False)
        return self.fusion(torch.cat([self.coarse_conv(coarse), self.detail_conv(detail)], dim=1))


class DetailAggregationModule(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, out_channels: int):
        super().__init__()
        self.low_conv = nn.Sequential(nn.Conv2d(low_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.high_conv = nn.Sequential(nn.Conv2d(high_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.refine = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, low_feat: torch.Tensor, high_feat: torch.Tensor) -> torch.Tensor:
        high_up = F.interpolate(high_feat, size=low_feat.shape[-2:], mode='bilinear', align_corners=False)
        return self.refine(self.low_conv(low_feat) + self.high_conv(high_up))


class GatedFusionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1), nn.Sigmoid(),
        )
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if x1.shape[-2:] != x2.shape[-2:]:
            x2 = F.interpolate(x2, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        g = self.gate(torch.cat([x1, x2], dim=1))
        return g * self.conv1(x1) + (1 - g) * self.conv2(x2)


class ResidualRefinementModule(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False), nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False), nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, x: torch.Tensor, prev_pred: torch.Tensor) -> torch.Tensor:
        if prev_pred.shape[-2:] != x.shape[-2:]:
            prev_pred = F.interpolate(prev_pred, size=x.shape[-2:], mode='bilinear', align_corners=False)
        r = self.conv3(self.conv2(self.conv1(torch.cat([x, prev_pred], dim=1))))
        return prev_pred + r


class MultiScaleRefinementModule(nn.Module):
    def __init__(self, in_channels: int, scales: List[float] = [0.5, 1.0, 2.0]):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels + 1, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1))
            for _ in scales
        ])
        self.fusion = nn.Sequential(nn.Conv2d(len(scales), 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 1))

    def forward(self, x: torch.Tensor, coarse_pred: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-2:]
        if coarse_pred.shape[-2:] != orig_size:
            coarse_pred = F.interpolate(coarse_pred, size=orig_size, mode='bilinear', align_corners=False)
        refined_preds = []
        for scale, conv in zip(self.scales, self.convs):
            if scale != 1.0:
                new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
                x_s = F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)
                p_s = F.interpolate(coarse_pred, size=new_size, mode='bilinear', align_corners=False)
                refined = F.interpolate(conv(torch.cat([x_s, p_s], dim=1)), size=orig_size, mode='bilinear', align_corners=False)
            else:
                refined = conv(torch.cat([x, coarse_pred], dim=1))
            refined_preds.append(refined)
        return self.fusion(torch.cat(refined_preds, dim=1)) + coarse_pred


class BoundaryAwareModule(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.edge_branch = EdgeDetectionModule(in_channels, 1)
        self.seg_branch = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(num_classes + 1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        edge_pred = self.edge_branch(x)
        seg_pred = self.seg_branch(x)
        refined_pred = self.fusion(torch.cat([seg_pred, edge_pred], dim=1))
        return seg_pred, edge_pred, refined_pred


def build_refinement_module(name: str, in_channels: int, out_channels: int = 1, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "edge_detection":
        return EdgeDetectionModule(in_channels, out_channels)
    if name == "boundary_refine":
        return BoundaryRefinementModule(in_channels, kwargs.get("detail_channels", in_channels), out_channels)
    if name == "detail_aggregate":
        return DetailAggregationModule(in_channels, kwargs.get("high_channels", in_channels), out_channels)
    if name == "gated_fusion":
        return GatedFusionModule(in_channels, out_channels)
    if name == "residual_refine":
        return ResidualRefinementModule(in_channels, kwargs.get("hidden_channels", 64))
    if name == "multi_scale_refine":
        return MultiScaleRefinementModule(in_channels, kwargs.get("scales", [0.5, 1.0, 2.0]))
    raise ValueError(f"Unsupported refinement module: {name}. Supported: {REFINEMENT_MODULES}")
