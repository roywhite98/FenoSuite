"""Regression coverage for mask selection and discrete-label restoration."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plant_phenotype.annotation.reassemble import reassemble_from_tiles


class TestAnnotationStitchRegressions(unittest.TestCase):
    def setUp(self) -> None:
        scratch = _ROOT / "workspace" / "test_tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self._tmp = tempfile.TemporaryDirectory(prefix="stitch_", dir=scratch)
        self.addCleanup(self._tmp.cleanup)
        self.tiles = Path(self._tmp.name)
        self._write_meta()

    def _write_meta(self, original_size=(8, 8)) -> None:
        (self.tiles / "meta.json").write_text(
            json.dumps({
                "tile_size": 8,
                "grid_size": [1, 1],
                "resized_size": [8, 8],
                "original_size": list(original_size),
            }),
            encoding="utf-8",
        )

    def _save_mask(self, filename: str, value: int) -> np.ndarray:
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = value
        path = self.tiles / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(path)
        return np.repeat(mask[:, :, None], 3, axis=2)

    def test_new_annotation_overrides_preprocessed_mask(self) -> None:
        self._save_mask("mask_0_0.png", 127)
        expected = self._save_mask("img_0_0_mask.png", 255)
        _, restored = reassemble_from_tiles(self.tiles)
        np.testing.assert_array_equal(restored, expected)

    def test_erased_annotation_also_overrides_old_foreground(self) -> None:
        self._save_mask("mask_0_0.png", 255)
        expected = self._save_mask("img_0_0_mask.png", 0)
        _, restored = reassemble_from_tiles(self.tiles)
        np.testing.assert_array_equal(restored, expected)

    def test_legacy_mask_filename_is_supported(self) -> None:
        expected = self._save_mask("mask_0_0.png", 255)
        _, restored = reassemble_from_tiles(self.tiles)
        np.testing.assert_array_equal(restored, expected)

    def test_prediction_flag_selects_batch_output_masks(self) -> None:
        annotation = self._save_mask("img_0_0_mask.png", 127)
        prediction = self._save_mask("masks/img_0_0_mask.png", 255)
        _, selected_annotation = reassemble_from_tiles(self.tiles)
        _, selected_prediction = reassemble_from_tiles(
            self.tiles, use_predicted_mask=True,
        )
        np.testing.assert_array_equal(selected_annotation, annotation)
        np.testing.assert_array_equal(selected_prediction, prediction)

    def test_missing_predictions_report_error_instead_of_using_annotations(self) -> None:
        self._save_mask("img_0_0_mask.png", 255)
        with self.assertRaises(FileNotFoundError):
            reassemble_from_tiles(self.tiles, use_predicted_mask=True)

    def test_partial_predictions_leave_missing_tiles_as_background(self) -> None:
        meta_path = self.tiles / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(grid_size=[2, 1], resized_size=[16, 8], original_size=[16, 8])
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        prediction = self._save_mask("masks/img_0_0_mask.png", 255)
        self._save_mask("img_0_1_mask.png", 255)
        _, restored = reassemble_from_tiles(self.tiles, use_predicted_mask=True)
        np.testing.assert_array_equal(restored[:, :8], prediction)
        self.assertFalse(np.any(restored[:, 8:]))

    def test_non_multiple_size_preserves_binary_mask_and_rgb_interpolation(self) -> None:
        self._write_meta(original_size=(5, 7))
        mask = self._save_mask("img_0_0_mask.png", 255)
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[2:6, 2:6] = [235, 120, 40]
        Image.fromarray(rgb).save(self.tiles / "img_0_0.png")
        image_path, mask_path = self.tiles / "restored.png", self.tiles / "restored_mask.png"
        image, restored = reassemble_from_tiles(self.tiles, image_path, mask_path)
        expected_mask = np.asarray(Image.fromarray(mask).resize((5, 7), Image.Resampling.NEAREST))
        expected_image = np.asarray(Image.fromarray(rgb).resize((5, 7), Image.Resampling.LANCZOS))
        self.assertEqual(restored.shape, (7, 5, 3))
        self.assertEqual(set(np.unique(restored)), {0, 255})
        np.testing.assert_array_equal(restored, expected_mask)
        np.testing.assert_array_equal(image, expected_image)
        with Image.open(mask_path) as saved:
            np.testing.assert_array_equal(np.asarray(saved), expected_mask)
        with Image.open(image_path) as saved:
            np.testing.assert_array_equal(np.asarray(saved), expected_image)


if __name__ == "__main__":
    unittest.main()
