# -*- coding: utf-8 -*-
"""根系表型对外门面（analyze / 描线校正 / 渲染 / 导出 / 默认参数）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from plant_phenotype.root_phenotype.core.pipeline import (
    PipelineResult,
    apply_main_root_correction_with_stroke,
    run_one,
)
from plant_phenotype.root_phenotype.figures import (
    export_visualization as _export_visualization,
    render_result_overlay as _render_result_overlay,
)
from plant_phenotype.root_phenotype.vector.models import (
    BaseMark, RootDocument, RootNode, VectorRoot, VectorAnalysisConfig,
    VectorAnalysisResult, EditResult,
)

# 供 GUI TYPE_CHECKING / 工厂再导出，禁止 gui 直连 core
__all__ = [
    "PhenotypingFacade",
    "PipelineResult",
    "get_facade",
    "BaseMark", "RootDocument", "RootNode", "VectorRoot", "VectorAnalysisConfig",
    "VectorAnalysisResult", "EditResult",
]


def _load_default_params() -> Dict[str, Any]:
    try:
        from plant_phenotype.root_phenotype.config import (
            CONNECT_BROKEN_ROOTS,
            EDGE_WEIGHT,
            PREPROCESS,
            PRIMARY_ROOT,
            STATS,
            TOPOLOGY,
        )

        return {
            "morph_kernel_size": PREPROCESS.get("morph_kernel_size", 3),
            "min_component_area": PREPROCESS.get("min_component_area", 50),
            "morph_open": PREPROCESS.get("morph_open", False),
            "connect_enabled": CONNECT_BROKEN_ROOTS.get("enabled", False),
            "connect_max_search": CONNECT_BROKEN_ROOTS.get("max_search", 120),
            "connect_angle_tol_deg": CONNECT_BROKEN_ROOTS.get("angle_tol_deg", 25.0),
            "connect_min_gap": CONNECT_BROKEN_ROOTS.get("min_gap", 3.0),
            "connect_max_gap": CONNECT_BROKEN_ROOTS.get("max_gap", 200.0),
            "junction_min_neighbors": TOPOLOGY.get("junction_min_neighbors", 3),
            "spur_max_length": int(TOPOLOGY.get("spur_max_length", 8)),
            "weight_length": EDGE_WEIGHT.get("length", 1.0),
            "weight_diameter": EDGE_WEIGHT.get("diameter", 0.5),
            "weight_curvature": EDGE_WEIGHT.get("curvature", -0.2),
            "min_root_length": STATS.get("min_root_length", 5.0),
            "min_root_diameter": STATS.get("min_root_diameter", 1.0),
            "zscore": STATS.get("zscore", True),
            "primary_enabled": PRIMARY_ROOT.get("enabled", False),
            "primary_alpha": PRIMARY_ROOT.get("alpha", 1.0),
            "primary_beta": PRIMARY_ROOT.get("beta", 0.5),
            "primary_gamma": PRIMARY_ROOT.get("gamma", 0.8),
            "primary_tau": PRIMARY_ROOT.get("tau", 0.75),
            "primary_max_level": PRIMARY_ROOT.get("max_level", 3),
            "primary_diameter_ratio_min": PRIMARY_ROOT.get("diameter_ratio_min", 0.3),
            "primary_diameter_floor": PRIMARY_ROOT.get("diameter_floor", 1.0),
        }
    except Exception:
        return {
            "morph_kernel_size": 3,
            "min_component_area": 50,
            "morph_open": False,
            "connect_enabled": False,
            "connect_max_search": 120,
            "connect_angle_tol_deg": 25.0,
            "connect_min_gap": 3.0,
            "connect_max_gap": 200.0,
            "junction_min_neighbors": 3,
            "spur_max_length": 8,
            "weight_length": 1.0,
            "weight_diameter": 0.5,
            "weight_curvature": -0.2,
            "min_root_length": 5.0,
            "min_root_diameter": 1.0,
            "zscore": True,
            "primary_enabled": False,
            "primary_alpha": 1.0,
            "primary_beta": 0.5,
            "primary_gamma": 0.8,
            "primary_tau": 0.75,
            "primary_max_level": 3,
            "primary_diameter_ratio_min": 0.3,
            "primary_diameter_floor": 1.0,
        }


class PhenotypingFacade:
    def analyze_vectors(self, mask, base_marks=None, *, original_image_path=None,
                        config=None, document_id=None, progress=None, check_cancel=None, selection_points=None):
        """New beta pipeline; separate document semantics from legacy analyze()."""
        from .vector.pipeline import analyze_vectors
        return analyze_vectors(mask, base_marks, original_image_path=original_image_path,
                               config=config, document_id=document_id,
                               progress=progress, check_cancel=check_cancel, selection_points=selection_points)

    def resolve_vectors(self, document, decisions, *, check_cancel=None):
        from .vector.editing import resolve_decisions
        return resolve_decisions(document, decisions, check_cancel=check_cancel)

    def apply_vector_edit(self, document, command):
        from .vector.editing import apply_edit
        return apply_edit(document, command)

    def measure_vectors(self, document):
        from .vector.metrics import measure_vectors
        return measure_vectors(document)

    def save_vector_document(self, document, path):
        from .vector.io import save_document
        return save_document(document, path)

    def load_vector_document(self, path):
        from .vector.io import load_document
        return load_document(path)

    def export_vector_tables(self, document, path):
        from .vector.io import export_tables
        return export_tables(document, path)

    def export_vector_svg(self, document, path):
        from .vector.io import export_svg
        return export_svg(document, path)

    def vector_root_color(self, root_id, document=None):
        from .vector.io import root_color
        return root_color(root_id, document)

    def vector_source_matches(self, document):
        """Check the original mask content before reusing stored observations."""
        from .vector.features import load_mask, mask_digest
        path = document.source.mask_path
        if not path or not Path(path).is_file():
            return False
        try:
            return mask_digest(load_mask(path)) == document.source.mask_hash
        except (OSError, ValueError):
            return False

    def analyze(
        self,
        mask: Union[str, Path, np.ndarray],
        seeds: Optional[List[Tuple[int, int]]] = None,
        *,
        mode: str = "interactive",
        original_image_path: Optional[str] = None,
        preprocessed_mask: Optional[np.ndarray] = None,
        config_overrides: Optional[dict] = None,
    ) -> PipelineResult:
        batch_mode = mode == "batch"
        mask_path = str(mask) if isinstance(mask, (str, Path)) else ""
        if isinstance(mask, np.ndarray):
            if preprocessed_mask is None:
                preprocessed_mask = mask
            mask_path = mask_path or "<array>"
        return run_one(
            mask_path,
            original_image_path=original_image_path,
            selected_seeds=seeds,
            preprocessed_mask=preprocessed_mask,
            batch_mode=batch_mode,
            config_overrides=config_overrides,
        )

    def correct_main_root(
        self,
        result: PipelineResult,
        stroke: List[Tuple[int, int]],
        config_overrides: Optional[dict] = None,
    ) -> Optional[PipelineResult]:
        return apply_main_root_correction_with_stroke(result, stroke, config_overrides)

    def render_overlay(
        self,
        result: PipelineResult,
        *,
        base_image: Optional[np.ndarray] = None,
        display_options: Optional[dict] = None,
        **render_kw,
    ) -> np.ndarray:
        opts = dict(display_options or {})
        opts.update(render_kw)
        return _render_result_overlay(
            result,
            base_image=base_image if base_image is not None else getattr(result, "original_image", None),
            **opts,
        )

    def export_figure(
        self,
        result: PipelineResult,
        path: str,
        *,
        base_image: Optional[np.ndarray] = None,
        display_options: Optional[dict] = None,
        **render_kw,
    ) -> None:
        opts = dict(display_options or {})
        opts.update(render_kw)
        _export_visualization(result, path, base_image=base_image, **opts)

    def create_preprocessed_partial(
        self,
        mask: np.ndarray,
        original_image: Optional[np.ndarray] = None,
    ) -> PipelineResult:
        """预处理后、尚未选根分析时的占位结果（无骨架/图）。"""
        return PipelineResult(
            mask=mask,
            skeleton=None,
            G=None,
            endpoints=[],
            junctions=[],
            endpoint_ids={},
            junction_ids={},
            main_path=[],
            main_roots_all=[],
            edge_level={},
            summary_df=None,
            detail_df=None,
            original_image=original_image,
        )

    def default_params(self) -> Dict[str, Any]:
        return dict(_load_default_params())


_default_facade = PhenotypingFacade()


def get_facade() -> PhenotypingFacade:
    return _default_facade
