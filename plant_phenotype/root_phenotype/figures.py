# -*- coding: utf-8 -*-
"""可视化结果图：在原图或 mask 上叠加骨架与分级着色、交点/端点圈注（来自 rootmask/export/figures）。"""
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List, Tuple, Dict

def _get_colors():
    """每次从 config 读取，与 _archived/SegmentTask2/config.py 配色一致。"""
    try:
        from plant_phenotype.root_phenotype.config import COLORS
        c = COLORS
        return (
            c.get("main_root", (255, 80, 80)),
            c.get("secondary_root", (255, 200, 80)),
            c.get("tertiary_root", (80, 220, 120)),
            c.get("junction_circle", (255, 255, 0)),
            c.get("endpoint_circle", (0, 200, 255)),
            c.get("skeleton_default", (255, 192, 203)),
        )
    except ImportError:
        return (255, 80, 80), (255, 200, 80), (80, 220, 120), (255, 255, 0), (0, 200, 255), (255, 192, 203)


def _get_line_width_main():
    try:
        from plant_phenotype.root_phenotype.config import LINE_WIDTH_MAIN
        return LINE_WIDTH_MAIN
    except ImportError:
        return 3


def render_result_overlay(
    result,
    base_image: Optional[np.ndarray] = None,
    line_width: int = 2,
    circle_radius: int = 8,
    font_scale: float = 0.5,
    show_skeleton: bool = True,
    show_junctions: bool = True,
    show_endpoints: bool = True,
    show_main: bool = True,
    show_secondary: bool = True,
) -> np.ndarray:
    """
    将 result（PipelineResult）绘制到 base_image 上。
    base_image: RGB (H,W,3) 或 None（则用 mask 伪彩）。
    show_* 控制骨架层、交点、根尖、主根、二级根的显示与隐藏。
    返回 RGB 图像。
    """
    if result.error:
        h, w = result.mask.shape[:2] if result.mask.size else (400, 400)
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[:] = (40, 40, 40)
        return out

    if result.G is None:
        mask = (result.mask > 0).astype(np.uint8) * 255 if result.mask.size else np.zeros((400, 400), dtype=np.uint8)
        img = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return img

    if base_image is not None and base_image.size > 0:
        if base_image.ndim == 2:
            base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
        else:
            base_image = cv2.cvtColor(base_image.astype(np.uint8), cv2.COLOR_RGB2BGR) if base_image.shape[-1] == 3 else base_image.astype(np.uint8)
        img = base_image.copy()
    else:
        mask = (result.mask > 0).astype(np.uint8) * 255 if result.mask.size else np.zeros((400, 400), dtype=np.uint8)
        img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    main_c, sec_c, _, junc_c, endp_c, skel_c = _get_colors()
    line_width_main = _get_line_width_main()
    main_c_bgr = (main_c[2], main_c[1], main_c[0])
    sec_c_bgr = (sec_c[2], sec_c[1], sec_c[0])
    junc_c_bgr = (junc_c[2], junc_c[1], junc_c[0])
    endp_c_bgr = (endp_c[2], endp_c[1], endp_c[0])
    skel_c_bgr = (skel_c[2], skel_c[1], skel_c[0])

    edge_level = getattr(result, "edge_level", {}) or {}
    G = result.G

    if show_skeleton:
        skel_lw = max(line_width + 2, line_width_main + 1)
        for (u, v) in G.edges():
            if not G.has_edge(u, v):
                continue
            d = G[u][v]
            pixels = d.get("pixels", [])
            for i in range(len(pixels) - 1):
                r0, c0 = pixels[i]
                r1, c1 = pixels[i + 1]
                cv2.line(img, (int(c0), int(r0)), (int(c1), int(r1)), skel_c_bgr, skel_lw)

    for (u, v), lev in edge_level.items():
        if lev not in (1, 2):
            continue
        if lev == 1 and not show_main:
            continue
        if lev == 2 and not show_secondary:
            continue
        if not G.has_edge(u, v):
            continue
        d = G[u][v]
        pixels = d.get("pixels", [])
        if not pixels:
            continue
        color = main_c_bgr if lev == 1 else sec_c_bgr
        lw = line_width_main if lev == 1 else line_width
        for i in range(len(pixels) - 1):
            r0, c0 = pixels[i]
            r1, c1 = pixels[i + 1]
            cv2.line(img, (int(c0), int(r0)), (int(c1), int(r1)), color, lw)

    if show_junctions:
        for n, i in getattr(result, "junction_ids", {}).items():
            r, c = int(n[0]), int(n[1])
            cv2.circle(img, (c, r), circle_radius, junc_c_bgr, 2)
            cv2.putText(img, f"J{i+1}", (c + circle_radius + 2, r), cv2.FONT_HERSHEY_SIMPLEX, font_scale, junc_c_bgr, 1)
    if show_endpoints:
        for n, i in getattr(result, "endpoint_ids", {}).items():
            r, c = int(n[0]), int(n[1])
            cv2.circle(img, (c, r), circle_radius, endp_c_bgr, 2)
            cv2.putText(img, f"E{i+1}", (c + circle_radius + 2, r), cv2.FONT_HERSHEY_SIMPLEX, font_scale, endp_c_bgr, 1)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _imwrite_unicode(path: str, img: np.ndarray) -> bool:
    ext = Path(path).suffix or ".png"
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def export_visualization(
    result,
    out_path: str,
    base_image: Optional[np.ndarray] = None,
    **render_kw,
) -> None:
    """单图：将 result 叠加到 base_image（或 mask）上并保存。"""
    img = render_result_overlay(result, base_image=base_image or getattr(result, "original_image", None), **render_kw)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    _imwrite_unicode(out_path, img)


def export_visualization_batch(
    results: List[Tuple[str, object]],
    out_dir: str,
    base_images: Optional[Dict[str, np.ndarray]] = None,
    **render_kw,
) -> List[str]:
    """批量：results = [(image_id, PipelineResult), ...]，保存到 out_dir，返回保存路径列表。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_images = base_images or {}
    saved = []
    for image_id, res in results:
        base = base_images.get(image_id) or getattr(res, "original_image", None)
        img = render_result_overlay(res, base_image=base, **render_kw)
        safe_id = Path(image_id).stem if isinstance(image_id, (str, Path)) else str(image_id)
        path = out_dir / f"{safe_id}_vis.png"
        _imwrite_unicode(str(path), img)
        saved.append(str(path))
    return saved
