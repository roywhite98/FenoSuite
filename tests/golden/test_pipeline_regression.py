# -*- coding: utf-8 -*-
"""run_one Golden 回归：主根长度与一级边集合。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plant_phenotype.root_phenotype.core.pipeline import run_one  # noqa: E402

from tests.golden.helpers import (  # noqa: E402
    compare_to_golden,
    load_golden,
    result_to_golden_dict,
)

MASK_DIR = _ROOT / "tests" / "fixtures" / "root_masks"
GOLDEN_DIR = _ROOT / "tests" / "golden"


def _golden_cases():
    for path in sorted(GOLDEN_DIR.glob("*.json")):
        if path.name == "helpers.py":
            continue
        data = load_golden(path)
        name = data["name"]
        base = name.replace("_batch", "").replace("_seeded", "")
        mask_path = MASK_DIR / f"{base}.png"
        yield path.stem, mask_path, data


class TestPipelineRegression(unittest.TestCase):
    @unittest.skipUnless(MASK_DIR.is_dir(), "缺少 fixtures，请运行 scripts/generate_golden.py")
    def test_run_one_matches_golden(self) -> None:
        for stem, mask_path, expected in _golden_cases():
            with self.subTest(stem=stem):
                self.assertTrue(mask_path.is_file(), f"mask missing: {mask_path}")
                seeds = expected.get("selected_seeds")
                seed_tuples = None
                if seeds:
                    seed_tuples = [(int(p[0]), int(p[1])) for p in seeds]
                # Match the options used by scripts/generate_golden.py.
                # Other cases exercise component selection without bridging.
                overrides = {"connect_broken_roots": "broken_gap" in stem}
                res = run_one(
                    str(mask_path),
                    selected_seeds=seed_tuples,
                    batch_mode=bool(expected.get("batch_mode", True)),
                    config_overrides=overrides,
                )
                actual = result_to_golden_dict(
                    stem,
                    batch_mode=bool(expected.get("batch_mode", True)),
                    selected_seeds=seed_tuples,
                    res=res,
                )
                compare_to_golden(actual, expected)


if __name__ == "__main__":
    unittest.main()
