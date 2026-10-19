"""Training/inference input and checkpoint architecture regressions."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from plant_phenotype.image_io import load_image_uint8_rgb
from plant_phenotype.models import dinov3_mlp_seg as dino
from plant_phenotype.models.build_model import build_model_from_config
from plant_phenotype.segmentation.predict import (
    load_model_and_config, preprocess_image, run_prediction,
)
from plant_phenotype.segmentation.runner import _load_image_rgb_uint8
from plant_phenotype.training.dataset.root_dataset import RootSegDataset
from plant_phenotype.training.train import build_transforms


class _LocalBackbone(torch.nn.Module):
    """Avoid a model download while exercising the actual DINO decoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, pixel_values, **kwargs):
        batch = pixel_values.shape[0]
        states = tuple(
            torch.full((batch, 9, 8), i / 10.0, device=pixel_values.device)
            for i in range(13)
        )
        return SimpleNamespace(hidden_states=states)


class TestSegmentationAlgorithms(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "workspace" / "test_tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="segmentation_", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_high_depth_training_prediction_and_preview_agree(self):
        values = np.tile(np.array([1000, 2000], dtype=np.uint16), (32, 16))
        path = self.root / "depth16.tif"
        Image.fromarray(values).save(path)
        Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(self.root / "depth16_mask.png")
        train_tf, _ = build_transforms((32, 32), enable_augmentation=False)
        dataset = RootSegDataset(str(self.root), image_ext=".tif", transform=train_tf)
        train_input, _ = dataset[0]
        rgb = load_image_uint8_rgb(str(path))
        prediction_input, _, _ = preprocess_image(rgb, (32, 32))
        torch.testing.assert_close(train_input, prediction_input[0])
        # Preserve the established training conversion for old checkpoints.
        np.testing.assert_array_equal(np.unique(rgb), [3, 7])
        np.testing.assert_array_equal(_load_image_rgb_uint8(str(path)), rgb)

    def test_rgb_channels_are_preserved(self):
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        rgb[:, :, 1] = 42
        path = self.root / "rgb.png"
        Image.fromarray(rgb).save(path)
        np.testing.assert_array_equal(load_image_uint8_rgb(str(path)), rgb)

    def test_dino_checkpoint_restores_nondefault_layers_and_decoder(self):
        config = SimpleNamespace(
            patch_size=16, hidden_size=8, num_hidden_layers=12,
            num_register_tokens=4, image_size=32,
        )
        with patch.object(dino, "HAS_TRANSFORMERS", True), \
             patch.object(dino, "AutoConfig") as auto_config, \
             patch.object(dino, "AutoModel") as auto_model:
            auto_config.from_pretrained.return_value = config
            auto_model.from_config.side_effect = _LocalBackbone
            for channels in (16, 256):
                with self.subTest(decoder_channels=channels):
                    model_cfg = {
                        "name": "dinov3_mlp", "encoder": "local-test-backbone",
                        "encoder_weights": None, "decoder_embed_dim": channels,
                        "encoder_layer_indices": [1, 4, 7, 10],
                    }
                    model = build_model_from_config(model_cfg).eval()
                    path = self.root / "checkpoint.pth"
                    torch.save({"config": {"model": model_cfg}, "model_state": model.state_dict()}, path)
                    loaded, _ = load_model_and_config(str(path), torch.device("cpu"))
                    self.assertEqual(loaded.layer_indices, [1, 4, 7, 10])
                    self.assertEqual(loaded.decoder_embed_dim, channels)
                    self.assertEqual(loaded.encoder_name, "local-test-backbone")
                    self.assertNotIn("pretrained", model_cfg)
                    with torch.no_grad():
                        image = torch.zeros(1, 3, 32, 32)
                        torch.testing.assert_close(model(image), loaded(image))
            auto_model.from_pretrained.assert_not_called()

    def test_prediction_extracts_segmentation_from_multitask_output(self):
        class MultiOutput(torch.nn.Module):
            def forward(self, x):
                shape = (x.shape[0], 1, *x.shape[-2:])
                return {
                    "segmentation": x.new_full(shape, 10),
                    "edge": x.new_full(shape, -10),
                    "distance": x.new_zeros(shape),
                }

        path = self.root / "sample.png"
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path)
        mask = run_prediction(MultiOutput(), str(path), torch.device("cpu"), (32, 32), 0.5, 0)
        self.assertEqual(mask.shape, (32, 32))
        self.assertTrue(np.all(mask == 255))

    def test_real_multitask_checkpoint_keeps_legacy_config_precedence(self):
        # Old training used name/classes even when their aliases disagreed.
        model_cfg = {
            "name": "multi_task_seg", "type": "attention_seg",
            "encoder": "resnet18", "encoder_weights": None,
            "classes": 1, "num_classes": 2, "decoder_channels": 16,
        }
        model = build_model_from_config(model_cfg).eval()
        image = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            expected = model(image)
        self.assertEqual(set(expected), {"segmentation", "edge", "distance"})
        self.assertEqual(tuple(expected["segmentation"].shape), (1, 1, 32, 32))
        path = self.root / "multitask.pth"
        torch.save({"config": {"model": model_cfg}, "model_state": model.state_dict()}, path)
        loaded, _ = load_model_and_config(str(path), torch.device("cpu"))
        with torch.no_grad():
            actual = loaded(image)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
