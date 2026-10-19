from collections.abc import Mapping
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from plant_phenotype.models.outputs import segmentation_logits


def _reduce_weighted_loss(loss_per_pixel: torch.Tensor, weight_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """对逐像素损失按 weight_mask 加权后求平均。"""
    if weight_mask is None:
        return loss_per_pixel.mean()
    weight_mask = weight_mask.expand_as(loss_per_pixel)
    return (loss_per_pixel * weight_mask).sum() / (weight_mask.sum() + 1e-8)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (N, 1, H, W) 未经过 sigmoid
        targets: (N, 1, H, W) 0/1
        """
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice
        return loss.mean()


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()
        if pos_weight is not None:
            self.register_buffer("_pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
            self.bce = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight)
        else:
            self._pos_weight = None
            self.bce = nn.BCEWithLogitsLoss()

    def to(self, device):
        self.bce.to(device)
        self.dice.to(device)
        return super().to(device)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (N, 1, H, W)
            targets: (N, 1, H, W) 0/1
            weight_mask: (N, 1, H, W) 可选，目标区域（根）像素权重大于 1，用于缓解样本不平衡
        """
        loss_dice = self.dice(logits, targets)

        if weight_mask is not None:
            # 逐像素 BCE 加权，目标区域损失权重大
            bce_per_pixel = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            loss_bce = _reduce_weighted_loss(bce_per_pixel, weight_mask)
        else:
            loss_bce = self.bce(logits, targets)

        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


class CascadedMultiScaleDiceLoss(nn.Module):
    """
    级联多尺度 DICE 损失：在三个尺度（完整、1/4、1/16）上分别计算 DICE 并线性组合，
    通过跨尺度监督增强模型对细节与整体上下文的特征感知与泛化能力。
    """

    def __init__(
        self,
        scale_factors: tuple = (1, 4, 16),
        scale_weights: Optional[tuple] = None,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.scale_factors = scale_factors
        if scale_weights is not None:
            self.scale_weights = tuple(scale_weights)
        else:
            self.scale_weights = tuple(1.0 / len(scale_factors) for _ in scale_factors)
        self.smooth = smooth
        self._dice = DiceLoss(smooth=smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (N, 1, H, W) 模型预测 logits
            targets: (N, 1, H, W) 二值 mask 0/1
            weight_mask: 未使用，仅为与 BCEDiceLoss 接口一致
        """
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        n, _, h, w = logits.shape
        total = None
        for scale_factor, weight in zip(self.scale_factors, self.scale_weights):
            if scale_factor == 1:
                logits_s = logits
                targets_s = targets
            else:
                h_s = max(1, h // scale_factor)
                w_s = max(1, w // scale_factor)
                size_s = (h_s, w_s)
                logits_s = F.interpolate(
                    logits, size=size_s, mode="bilinear", align_corners=False
                )
                targets_s = F.interpolate(
                    targets.float(), size=size_s, mode="bilinear", align_corners=False
                )
                targets_s = (targets_s >= 0.5).float()
            scale_loss = self._dice(logits_s, targets_s)
            if total is None:
                total = weight * scale_loss
            else:
                total = total + weight * scale_loss
        if total is None:
            total = torch.zeros(1, device=logits.device, dtype=logits.dtype).squeeze(0)
        return total


class BCECascadedMultiScaleDiceLoss(nn.Module):
    """
    BCE + 级联多尺度 DICE：主监督为 BCE，DICE 部分在完整、1/4、1/16 三尺度上线性组合。
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None,
        cascaded_scale_factors: tuple = (1, 4, 16),
        cascaded_scale_weights: Optional[tuple] = None,
        dice_smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.cascaded_dice = CascadedMultiScaleDiceLoss(
            scale_factors=cascaded_scale_factors,
            scale_weights=cascaded_scale_weights,
            smooth=dice_smooth,
        )
        if pos_weight is not None:
            self.register_buffer("_pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
            self.bce = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight)
        else:
            self._pos_weight = None
            self.bce = nn.BCEWithLogitsLoss()

    def to(self, device):
        self.bce.to(device)
        self.cascaded_dice.to(device)
        return super().to(device)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss_dice = self.cascaded_dice(logits, targets, weight_mask)
        if weight_mask is not None:
            bce_per_pixel = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            loss_bce = _reduce_weighted_loss(bce_per_pixel, weight_mask)
        else:
            loss_bce = self.bce(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


def auxiliary_targets(targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inner 8-connected boundary and per-image normalized foreground distance.

    Pixels outside the crop are background. The distance is the Euclidean
    distance to the closest background pixel, divided by each image's maximum;
    an empty mask has zero boundary and distance everywhere.
    """
    from scipy.ndimage import distance_transform_edt

    foreground = (targets.detach() > 0.5).any(dim=1, keepdim=True).float()
    padded = F.pad(foreground, (1, 1, 1, 1), value=0)
    eroded = -F.max_pool2d(-padded, kernel_size=3, stride=1)
    edge = foreground - eroded
    masks = foreground[:, 0].cpu().numpy().astype(bool)
    distances = np.zeros(masks.shape, dtype=np.float32)
    for index, mask in enumerate(masks):
        distance = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        maximum = distance.max()
        if maximum > 0:
            distances[index] = distance / maximum
    distance_target = torch.from_numpy(distances[:, None]).to(device=targets.device)
    return edge, distance_target


class SegmentationWithAuxiliaryLoss(nn.Module):
    """Keep tensor-model losses unchanged; supervise all present auxiliary heads.

    ``loss.edge_weight`` scales boundary BCE and ``loss.distance_weight`` scales
    MSE on the distance head's raw regression output. Both default to 0.2 and
    may be set to zero independently.
    """

    def __init__(self, segmentation_loss: nn.Module, edge_weight: float = 0.2,
                 distance_weight: float = 0.2) -> None:
        super().__init__()
        if not np.isfinite(edge_weight) or not np.isfinite(distance_weight) or min(edge_weight, distance_weight) < 0:
            raise ValueError("Auxiliary loss weights must be finite and nonnegative")
        self.segmentation_loss = segmentation_loss
        self.edge_weight = edge_weight
        self.distance_weight = distance_weight

    def forward(self, outputs, targets: torch.Tensor,
                weight_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = self.segmentation_loss(segmentation_logits(outputs), targets, weight_mask)
        if not isinstance(outputs, Mapping):
            return loss
        use_edge = self.edge_weight > 0 and "edge" in outputs
        use_distance = self.distance_weight > 0 and "distance" in outputs
        if use_edge or use_distance:
            edge_target, distance_target = auxiliary_targets(targets)
            if use_edge:
                loss = loss + self.edge_weight * F.binary_cross_entropy_with_logits(outputs["edge"], edge_target)
            if use_distance:
                if outputs["distance"].shape != distance_target.shape:
                    raise ValueError("Distance prediction and target must have the same shape")
                loss = loss + self.distance_weight * F.mse_loss(outputs["distance"], distance_target)
        return loss


def build_criterion(loss_cfg: dict, device: torch.device) -> nn.Module:
    """
    根据配置构建损失函数，使 loss 在配置中可选。
    支持类型：bce_dice（默认）、bce_cascaded_multiscale_dice。
    """
    loss_cfg = loss_cfg or {}
    loss_type = (loss_cfg.get("type") or "bce_dice").lower().strip()
    if loss_type == "bce_cascaded_multiscale_dice":
        weights_cfg = loss_cfg.get("cascaded_scale_weights")
        cascaded_weights = tuple(weights_cfg) if weights_cfg else None
        criterion = BCECascadedMultiScaleDiceLoss(
            bce_weight=loss_cfg.get("bce_weight", 0.5),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            pos_weight=loss_cfg.get("pos_weight", None),
            cascaded_scale_factors=tuple(loss_cfg.get("cascaded_scale_factors", [1, 4, 16])),
            cascaded_scale_weights=cascaded_weights,
            dice_smooth=loss_cfg.get("dice_smooth", 1.0),
        )
    else:
        criterion = BCEDiceLoss(
            bce_weight=loss_cfg.get("bce_weight", 0.5),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            pos_weight=loss_cfg.get("pos_weight", None),
        )
    return SegmentationWithAuxiliaryLoss(
        criterion,
        edge_weight=float(loss_cfg.get("edge_weight", 0.2)),
        distance_weight=float(loss_cfg.get("distance_weight", 0.2)),
    ).to(device)
