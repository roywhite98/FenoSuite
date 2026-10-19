import torch


@torch.no_grad()
def dice_coef(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    logits: (N,1,H,W)
    targets: (N,1,H,W)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    return dice.mean().item()


@torch.no_grad()
def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.mean().item()
