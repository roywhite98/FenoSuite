# -*- coding: utf-8 -*-
"""生成 tests/fixtures/root_masks 与 tests/golden/*.json 基准文件。"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from plant_phenotype.root_phenotype.core.pipeline import run_one  # noqa: E402

from tests.fixtures.mask_factory import FIXTURE_BUILDERS  # noqa: E402
from tests.golden.helpers import result_to_golden_dict, save_golden  # noqa: E402

MASK_DIR = REPO / "tests" / "fixtures" / "root_masks"
GOLDEN_DIR = REPO / "tests" / "golden"

CASES = [
    ("straight_horizontal", True, None),
    ("t_junction", True, None),
    ("y_fork", True, None),
    ("main_with_lateral", True, None),
    ("two_components", True, None),
    ("broken_gap", True, None),
    ("main_with_lateral", False, [(36, 24)]),
]


def main() -> None:
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in FIXTURE_BUILDERS.items():
        mask = builder()
        out = MASK_DIR / f"{name}.png"
        Image.fromarray(mask).save(out)

    for name, batch_mode, seeds in CASES:
        mask_path = MASK_DIR / f"{name}.png"
        res = run_one(
            str(mask_path),
            selected_seeds=seeds,
            batch_mode=batch_mode,
            config_overrides={"connect_broken_roots": name == "broken_gap"},
        )
        suffix = "_seeded" if seeds else ("_batch" if batch_mode else "")
        golden_name = f"{name}{suffix}"
        data = result_to_golden_dict(golden_name, batch_mode=batch_mode, selected_seeds=seeds, res=res)
        save_golden(GOLDEN_DIR / f"{golden_name}.json", data)
        print(f"OK {golden_name} len={data['main_root_length']:.2f} L1={len(data['level1_edges'])}")


if __name__ == "__main__":
    main()
