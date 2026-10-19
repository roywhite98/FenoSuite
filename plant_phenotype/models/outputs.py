"""Shared output contract for tensor and multi-task segmentation models."""
from collections.abc import Mapping

import torch


def segmentation_logits(outputs) -> torch.Tensor:
    """Return unnormalized segmentation scores without discarding auxiliary heads."""
    logits = outputs.get("segmentation") if isinstance(outputs, Mapping) else outputs
    if not isinstance(logits, torch.Tensor):
        raise TypeError("Model output must be a Tensor or a mapping containing a segmentation Tensor")
    return logits
