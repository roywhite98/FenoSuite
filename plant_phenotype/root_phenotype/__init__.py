# -*- coding: utf-8 -*-
"""根系表型：基于 root_phenotype.morphology 的形态统计，参考 rootmask 参数与导出。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from plant_phenotype.root_phenotype.morphology import analyze_mask, analyze_mask_image
from plant_phenotype.root_phenotype.config import STATS, PREPROCESS


def run_analysis(
    mask_path: str,
    pixel_size: float = 1.0,
    min_root_area: Optional[int] = None,
    remove_small_objects: Optional[int] = None,
    morph_kernel_size: Optional[int] = None,
    morph_open: Optional[bool] = None,
    min_root_diameter: Optional[float] = None,
) -> Dict:
    """
    对单张根系掩码做形态分析。
    参考 config.root_phenotype（STATS、PREPROCESS）默认值。
    """
    min_area = min_root_area if min_root_area is not None else int(STATS.get("min_root_length", 5.0) ** 2)
    morph_k = morph_kernel_size if morph_kernel_size is not None else PREPROCESS.get("morph_kernel_size", 0)
    morph_o = morph_open if morph_open is not None else PREPROCESS.get("morph_open", False)
    min_d = min_root_diameter if min_root_diameter is not None else float(STATS.get("min_root_diameter", 0.0))
    return analyze_mask_image(
        mask_path,
        pixel_size=pixel_size,
        min_root_area=min_area,
        remove_small_objects=remove_small_objects,
        morph_kernel_size=morph_k,
        morph_open=morph_o,
        min_root_diameter=min_d,
    )


def run_analysis_batch(
    mask_dir: str,
    pixel_size: float = 1.0,
    min_root_area: Optional[int] = None,
    morph_kernel_size: Optional[int] = None,
    morph_open: Optional[bool] = None,
    min_root_diameter: Optional[float] = None,
    extensions: tuple = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
) -> List[Dict]:
    """对目录下所有掩码图像逐张分析，返回每张的结果列表。"""
    results = []
    p = Path(mask_dir)
    if not p.is_dir():
        return results
    for f in sorted(p.iterdir()):
        if f.suffix.lower() not in extensions:
            continue
        try:
            r = run_analysis(
                str(f),
                pixel_size=pixel_size,
                min_root_area=min_root_area,
                morph_kernel_size=morph_kernel_size,
                morph_open=morph_open,
                min_root_diameter=min_root_diameter,
            )
            r["_path"] = str(f)
            results.append(r)
        except Exception:
            continue
    return results


def write_analysis_log(log_path: str, entries: List[Dict]) -> None:
    """将分析结果摘要写入日志文件（参考 rootmask analysis.log）。"""
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            import time
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"\n{ts} [INFO] 批量表型分析 %d 张\n" % len(entries))
            for r in entries:
                path = r.get("_path", "")
                f.write(
                    "  %s: 根数=%d 总长=%.2f 总面积=%.2f 平均根长=%.2f\n"
                    % (
                        path,
                        r.get("num_roots", 0),
                        r.get("total_root_length", 0),
                        r.get("total_root_area", 0),
                        r.get("mean_root_length", 0),
                    )
                )
    except Exception:
        pass


def export_visualization(mask_path: str, output_path: str) -> None:
    """
    将掩码与骨架叠加导出为图像（参考 rootmask 导出可视化图）。
    掩码灰度显示，骨架用 config 中 skeleton_default 颜色叠加。
    """
    import cv2
    from skimage import morphology

    from plant_phenotype.root_phenotype.config import COLORS

    with open(mask_path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取: {mask_path}")
    binary = (img > 0).astype(np.uint8)
    try:
        sk = morphology.skeletonize(binary, method="lee")
    except Exception:
        sk = morphology.skeletonize(binary.astype(bool)).astype(np.uint8)
    sk_uint8 = (sk > 0).astype(np.uint8) * 255

    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color = COLORS.get("skeleton_default", (255, 192, 203))
    color_bgr = (color[2], color[1], color[0])
    out[sk_uint8 > 0] = color_bgr

    ext = Path(output_path).suffix or ".png"
    ok, buf = cv2.imencode(ext, out)
    if not ok:
        raise IOError("imencode failed")
    with open(output_path, "wb") as f:
        f.write(buf.tobytes())
