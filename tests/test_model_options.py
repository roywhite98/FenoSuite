# -*- coding: utf-8 -*-
"""MobileSAM 注册与 ResNet50 浅层冻结。"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plant_phenotype.models.build_model import SUPPORTED_MODELS, build_model  # noqa: E402


@unittest.skipUnless(importlib.util.find_spec("segmentation_models_pytorch"), "需要 segmentation_models_pytorch")
class TestResNetFreeze(unittest.TestCase):
    def _model(self, stages):
        model = build_model(
            name="unet", encoder_name="resnet50", encoder_weights=None,
            in_channels=3, classes=1, freeze_stages=stages,
        )
        return model

    def test_layer1_only(self) -> None:
        model = self._model(["layer1"])
        model.train()
        self.assertTrue(model.encoder.conv1.weight.requires_grad)
        self.assertTrue(model.encoder.bn1.training)
        self.assertTrue(model.encoder.layer2.training)
        self.assertFalse(model.encoder.layer1.training)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.layer1.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.encoder.layer2.parameters()))
        self.assertTrue(all(not m.training for m in model.encoder.layer1.modules() if isinstance(m, torch.nn.BatchNorm2d)))

    def test_layer1_and_layer2_survive_train_and_step(self) -> None:
        model = self._model(["layer1", "layer2"])
        model.eval().train()
        for stage in (model.encoder.layer1, model.encoder.layer2):
            self.assertFalse(stage.training)
            self.assertTrue(all(not p.requires_grad for p in stage.parameters()))
        self.assertTrue(model.encoder.layer3.training)
        before = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith("encoder.layer1.") or k.startswith("encoder.layer2.")}
        stem_before = model.encoder.conv1.weight.detach().clone()
        images = torch.randn(1, 3, 64, 64)
        target = torch.zeros(1, 1, 64, 64)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(images), target)
        loss.backward()
        for stage in (model.encoder.layer1, model.encoder.layer2):
            self.assertTrue(all(p.grad is None for p in stage.parameters()))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        opt.step()
        after = model.state_dict()
        for key, value in before.items():
            self.assertTrue(torch.equal(after[key], value), key)
        self.assertFalse(torch.equal(model.encoder.conv1.weight, stem_before))


class TestMobileSAMRegistration(unittest.TestCase):
    def test_listed(self) -> None:
        self.assertIn("mobile_sam", SUPPORTED_MODELS)

    def test_forward_shape(self) -> None:
        model = build_model(name="mobile_sam", checkpoint=None)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 32, 48))
        self.assertEqual(tuple(out.shape), (2, 1, 32, 48))


if __name__ == "__main__":
    unittest.main()
