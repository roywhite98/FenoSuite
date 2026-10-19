"""
根系拓扑注意力 (Root Topology Attention, RTA)
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.conv(x), p=2, dim=1, eps=1e-6)


class RootTopologyAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8, window_size: int = 0, use_topology_gate: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.use_topology_gate = use_topology_gate
        inter_channels = max(in_channels // reduction, 1)
        self.direction_predictor = DirectionPredictor(in_channels, 32)
        self.query = nn.Conv2d(in_channels, inter_channels, 1)
        self.key = nn.Conv2d(in_channels, inter_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.topology_gate = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 1), nn.Sigmoid(),
        ) if use_topology_gate else None
        self.gamma = nn.Parameter(torch.zeros(1))

    def _forward_global(self, x: torch.Tensor, direction: torch.Tensor, topology_mask: Optional[torch.Tensor]) -> torch.Tensor:
        N, C, H, W = x.size()
        L = H * W
        q = self.query(x).view(N, -1, L).permute(0, 2, 1)
        k = self.key(x).view(N, -1, L)
        v = self.value(x).view(N, -1, L)
        attn = torch.bmm(q, k) / (q.size(-1) ** 0.5)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        return out.view(N, C, H, W)

    def _forward_window(self, x: torch.Tensor, direction: torch.Tensor, topology_mask: Optional[torch.Tensor]) -> torch.Tensor:
        N, C, H, W = x.size()
        w = self.window_size
        pad_h, pad_w = (w - H % w) % w, (w - W % w) % w
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            direction = F.pad(direction, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, Hp, Wp = x.size()
        nh, nw = Hp // w, Wp // w
        x_win = x.view(N, C, nh, w, nw, w).permute(0, 2, 4, 1, 3, 5).contiguous().view(N * nh * nw, C, w, w)
        dir_win = direction.view(N, 2, nh, w, nw, w).permute(0, 2, 4, 1, 3, 5).contiguous().view(N * nh * nw, 2, w, w)
        B_prime, L = x_win.size(0), w * w
        q = self.query(x_win).flatten(2).permute(0, 2, 1)
        k = self.key(x_win).flatten(2)
        v = self.value(x_win).flatten(2)
        attn = torch.bmm(q, k) / (q.size(-1) ** 0.5)
        device = x.device
        ii = torch.arange(w, device=device).float()
        jj = torch.arange(w, device=device).float()
        grid_i, grid_j = torch.meshgrid(ii, jj, indexing="ij")
        pos_h, pos_w = grid_i.reshape(-1), grid_j.reshape(-1)
        dh = pos_h.unsqueeze(1) - pos_h.unsqueeze(0)
        dw = pos_w.unsqueeze(1) - pos_w.unsqueeze(0)
        dist = torch.sqrt(dh**2 + dw**2 + 1e-8)
        unit_dw, unit_dh = dw / dist, -dh / dist
        dir_flat = dir_win.view(B_prime, 2, -1).permute(0, 2, 1)
        unit = torch.stack([unit_dw, unit_dh], dim=-1)
        align = torch.einsum("blc,ljc->blj", dir_flat.float(), unit.float())
        align = (align + 1) / 2
        attn = F.softmax(attn + align, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B_prime, C, w, w)
        out = out.view(N, nh, nw, C, w, w).permute(0, 3, 1, 4, 2, 5).contiguous().view(N, C, Hp, Wp)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        return out

    def forward(self, x: torch.Tensor, topology_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        direction = self.direction_predictor(x)
        if topology_mask is not None:
            direction = F.normalize(direction * topology_mask + (1 - topology_mask) * 1e-3, p=2, dim=1, eps=1e-6)
        out = self._forward_window(x, direction, topology_mask) if self.window_size > 0 else self._forward_global(x, direction, topology_mask)
        if self.topology_gate is not None:
            gate = self.topology_gate(x)
            out = out * gate + x * (1 - gate)
        else:
            out = out + x
        return self.gamma * out + x


def build_root_topology_attention(in_channels: int, window_size: int = 8, reduction: int = 8, use_topology_gate: bool = True) -> RootTopologyAttention:
    return RootTopologyAttention(in_channels=in_channels, reduction=reduction, window_size=window_size, use_topology_gate=use_topology_gate)
