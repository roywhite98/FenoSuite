"""Regression checks for multi-task supervision and checkpoint continuation."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch import nn

from plant_phenotype.models.outputs import segmentation_logits
from plant_phenotype.training.losses import BCEDiceLoss, auxiliary_targets, build_criterion
from plant_phenotype.training import train


class TinyMultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict({name: nn.Conv2d(3, 1, 1) for name in ("segmentation", "edge", "distance")})

    def forward(self, images):
        return {name: head(images) for name, head in self.heads.items()}


class TestMultiTaskTraining(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.masks = torch.zeros(2, 1, 7, 7)
        self.masks[0, 0, 2:5, 2:5] = 1
        self.images = torch.randn(2, 3, 7, 7)
        self.device = torch.device("cpu")

    def test_shared_output_contract(self):
        logits = torch.zeros_like(self.masks)
        self.assertIs(segmentation_logits(logits), logits)
        self.assertIs(segmentation_logits({"segmentation": logits, "edge": logits}), logits)
        with self.assertRaises(TypeError):
            segmentation_logits({"edge": logits})

    def test_boundary_and_distance_targets(self):
        edge, distance = auxiliary_targets(self.masks)
        self.assertEqual(edge[0].sum().item(), 8)
        self.assertEqual(edge[0, 0, 3, 3].item(), 0)
        self.assertEqual(distance[0, 0, 3, 3].item(), 1)
        self.assertEqual(distance[0, 0, 2, 2].item(), 0.5)
        self.assertEqual(distance[self.masks == 0].sum().item(), 0)
        self.assertEqual(edge[1].sum().item(), 0)
        self.assertTrue(torch.isfinite(distance).all())

    def test_all_foreground_crop_uses_background_outside_crop(self):
        edge, distance = auxiliary_targets(torch.ones(1, 1, 3, 3))
        self.assertEqual(edge.sum().item(), 8)
        self.assertEqual(distance[0, 0, 0, 0].item(), 0.5)
        self.assertEqual(distance[0, 0, 1, 1].item(), 1)

    def test_tensor_model_loss_is_unchanged(self):
        logits = torch.randn_like(self.masks)
        weights = 1 + 2 * self.masks
        criterion = build_criterion({}, self.device)
        self.assertEqual(criterion(logits, self.masks, weights).item(),
                         BCEDiceLoss()(logits, self.masks, weights).item())

    def test_auxiliary_weights_and_gradients(self):
        outputs = {name: torch.zeros_like(self.masks, requires_grad=True)
                   for name in ("segmentation", "edge", "distance")}
        criterion = build_criterion({"edge_weight": 0.3, "distance_weight": 0.7}, self.device)
        edge, distance = auxiliary_targets(self.masks)
        expected = (BCEDiceLoss()(outputs["segmentation"], self.masks)
                    + 0.3 * nn.functional.binary_cross_entropy_with_logits(outputs["edge"], edge)
                    + 0.7 * nn.functional.mse_loss(outputs["distance"], distance))
        loss = criterion(outputs, self.masks)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        for output in outputs.values():
            self.assertGreater(output.grad.abs().sum().item(), 0)

    def test_disabling_auxiliary_losses_preserves_segmentation_loss(self):
        outputs = {name: torch.randn_like(self.masks) for name in ("segmentation", "edge", "distance")}
        criterion = build_criterion({"edge_weight": 0, "distance_weight": 0}, self.device)
        with patch("plant_phenotype.training.losses.auxiliary_targets") as targets:
            loss = criterion(outputs, self.masks)
            targets.assert_not_called()
        torch.testing.assert_close(loss, BCEDiceLoss()(outputs["segmentation"], self.masks))

    def test_training_updates_all_heads_and_validation_uses_segmentation(self):
        model = TinyMultiTaskModel()
        before = {name: head.weight.detach().clone() for name, head in model.heads.items()}
        criterion = build_criterion({}, self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loader = [(self.images, self.masks)]
        loss = train.train_one_epoch(model, loader, criterion, optimizer, self.device, None, 0, 10, 1, 1)
        self.assertGreater(loss, 0)
        for name, head in model.heads.items():
            self.assertFalse(torch.equal(before[name], head.weight), name)
        val_loss, dice, iou = train.validate(model, loader, criterion, self.device)
        expected_logits = model(self.images)["segmentation"]
        self.assertGreater(val_loss, 0)
        self.assertEqual(dice, train.dice_coef(expected_logits, self.masks))
        self.assertEqual(iou, train.iou_score(expected_logits, self.masks))


class TestSchedulerContinuation(unittest.TestCase):
    batches = 3
    epochs = 8
    metrics = (0.5, 0.6, 0.55, 0.54, 0.53, 0.7, 0.65, 0.64)

    def setup_schedule(self, kind):
        model = nn.Linear(1, 1)
        cfg = {"train": {"lr": 0.01, "lr_schedule": kind, "step_size": 2,
                         "step_gamma": 0.5, "plateau_patience": 1, "plateau_factor": 0.5}}
        optimizer, scheduler = train.build_optimizer_scheduler(model, cfg, self.epochs * self.batches, self.epochs)
        return optimizer, scheduler

    def advance(self, optimizer, scheduler, first, last):
        lrs = []
        for index in range(first, last):
            lrs.append(optimizer.param_groups[0]["lr"])
            for _ in range(self.batches):
                optimizer.step()
            train._step_scheduler(scheduler, self.metrics[index], self.batches)
        return lrs

    def assert_continuation(self, kind, legacy=False):
        optimizer, scheduler = self.setup_schedule(kind)
        self.advance(optimizer, scheduler, 0, 3)
        checkpoint = {"epoch": 3, "best_dice": max(self.metrics[:3]),
                      "optimizer_state": copy.deepcopy(optimizer.state_dict())}
        if not legacy:
            checkpoint["scheduler_state"] = copy.deepcopy(scheduler.state_dict())
        expected = self.advance(optimizer, scheduler, 3, self.epochs)
        resumed_optimizer, resumed_scheduler = self.setup_schedule(kind)
        resumed_optimizer.load_state_dict(checkpoint["optimizer_state"])
        train._restore_scheduler_state(resumed_scheduler, checkpoint, self.batches, lambda _: None)
        actual = self.advance(resumed_optimizer, resumed_scheduler, 3, self.epochs)
        self.assertEqual(actual, expected)
        self.assertEqual(resumed_scheduler.last_epoch, scheduler.last_epoch)
        self.assertEqual(resumed_optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])

    def test_saved_scheduler_preserves_all_learning_rate_trajectories(self):
        for kind in ("poly", "cosine", "step", "plateau"):
            with self.subTest(schedule=kind):
                self.assert_continuation(kind)

    def test_legacy_checkpoints_recover_schedule_position(self):
        for kind in ("poly", "cosine", "step"):
            with self.subTest(schedule=kind):
                self.assert_continuation(kind, legacy=True)

    def test_legacy_plateau_preserves_lr_and_reports_missing_history(self):
        optimizer, scheduler = self.setup_schedule("plateau")
        optimizer.param_groups[0]["lr"] = 0.001
        messages = []
        train._restore_scheduler_state(scheduler, {"epoch": 5, "best_dice": 0.7}, self.batches, messages.append)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.001)
        self.assertEqual(scheduler.last_epoch, 5)
        self.assertEqual(scheduler.best, 0.7)
        self.assertEqual(scheduler.num_bad_epochs, 0)
        self.assertIn("无法精确恢复", messages[0])

    def test_training_checkpoint_includes_scheduler_and_scaler(self):
        # Keep all temporary checkpoints in the authorized workspace.
        scratch = _ROOT / "workspace" / "test_tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="training-regression-", dir=scratch) as temporary:
            cfg = {"model": {"name": "multi_task_seg"}, "loss": {},
                   "train": {"epochs": 2, "lr": 0.01, "lr_schedule": "poly", "amp": False,
                             "save_dir": temporary, "enable_visualization": False}}
            loader = [(torch.randn(1, 3, 5, 5), torch.zeros(1, 1, 5, 5))] * 2
            with patch.object(train, "build_transforms", return_value=(None, None)), \
                 patch.object(train, "build_dataloaders", return_value=(loader, loader)), \
                 patch.object(train, "build_model_from_config", side_effect=lambda *a, **k: TinyMultiTaskModel()), \
                 patch.object(train, "validate", side_effect=[(1.0, 0.1, 0.05), (0.9, 0.2, 0.1)]):
                train._run_training_impl(cfg, lambda _: None, use_cpu=True)
            checkpoint = torch.load(Path(temporary) / "best_model.pth", weights_only=False)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(checkpoint["scheduler_state"]["last_epoch"], 4)
            self.assertEqual(checkpoint["scheduler_state"]["_last_lr"], [0.0])
            self.assertIn("scaler_state", checkpoint)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
