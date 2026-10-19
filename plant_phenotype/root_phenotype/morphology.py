"""
根系形态分析：对二值分割掩码进行计算机视觉测量。

测量项：最长/最短根的长度与宽度、根长均值、总根长、总根面积、侧根/不定根夹角等。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage import morphology
from skimage.measure import label, regionprops


# 8 邻接下的步长：上下左右=1，对角=sqrt(2)
_STEP_LEN = np.array([
    [math.sqrt(2), 1.0, math.sqrt(2)],
    [1.0, 0.0, 1.0],
    [math.sqrt(2), 1.0, math.sqrt(2)],
], dtype=np.float64)


def _skeleton_length_8conn(skeleton: np.ndarray) -> float:
    """基于 8 邻接的骨架路径长度（像素单位）。"""
    if skeleton.sum() == 0:
        return 0.0
    sk = skeleton.astype(np.uint8)
    # 对骨架每个像素，与 3x3 邻域内骨架像素的距离按 _STEP_LEN 累加一半（避免重复）
    h, w = sk.shape
    length = 0.0
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not sk[r, c]:
                continue
            patch = sk[r - 1 : r + 2, c - 1 : c + 2]
            for dr in range(3):
                for dc in range(3):
                    if (dr, dc) == (1, 1):
                        continue
                    if patch[dr, dc]:
                        length += _STEP_LEN[dr, dc]
    return length / 2.0  # 每条边只算一次


def _skeleton_length_simple(skeleton: np.ndarray) -> float:
    """骨架像素数近似长度（含对角校正）。"""
    n = int(skeleton.sum())
    if n == 0:
        return 0.0
    # 近似：4 邻接与对角混合，约 1.2 倍像素数
    return n * 1.2


def skeleton_length(skeleton: np.ndarray, use_8conn: bool = True) -> float:
    """计算骨架长度（像素）。use_8conn=True 时用 8 邻接步长累加。"""
    if use_8conn:
        return _skeleton_length_8conn(skeleton)
    return _skeleton_length_simple(skeleton)


def _branch_points_8conn(skeleton: np.ndarray) -> np.ndarray:
    """骨架中分支点（8 邻接邻居数 >= 3）的坐标 (N, 2)。"""
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    conv = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)
    # 分支点：骨架点且邻居数 >= 3
    branch = (skeleton > 0) & (conv >= 3)
    return np.column_stack(np.nonzero(branch))


def _direction_from_branch(
    skeleton: np.ndarray,
    br: Tuple[int, int],
    branch_radius: int = 5,
) -> List[np.ndarray]:
    """从分支点 br 出发，沿骨架得到各分支的方向单位向量（最多取 branch_radius 步）。"""
    h, w = skeleton.shape
    sk = skeleton.astype(np.uint8)
    directions = []
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            r, c = br[0] + dr, br[1] + dc
            if r < 0 or r >= h or c < 0 or c >= w or not sk[r, c]:
                continue
            # 沿该方向走 branch_radius 步，取终点与 br 的向量
            path_len = 0
            pr, pc = r, c
            prev = (br[0], br[1])
            while path_len < branch_radius:
                best = None
                best_dist = 0
                for nr in range(pr - 1, pr + 2):
                    for nc in range(pc - 1, pc + 2):
                        if (nr, nc) == prev:
                            continue
                        if 0 <= nr < h and 0 <= nc < w and sk[nr, nc]:
                            d = (nr - br[0]) ** 2 + (nc - br[1]) ** 2
                            if d > best_dist:
                                best_dist = d
                                best = (nr, nc)
                if best is None:
                    break
                prev = (pr, pc)
                pr, pc = best
                path_len += 1
            vec = np.array([pr - br[0], pc - br[1]], dtype=np.float64)
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                directions.append(vec / norm)
    return directions


def _lateral_angles_at_branch(
    skeleton: np.ndarray,
    br: Tuple[int, int],
    branch_radius: int = 5,
) -> List[float]:
    """在分支点处计算两两分支夹角（度），返回列表。"""
    vecs = _direction_from_branch(skeleton, br, branch_radius)
    if len(vecs) < 2:
        return []
    angles = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dot = np.clip(np.dot(vecs[i], vecs[j]), -1.0, 1.0)
            angle_rad = math.acos(dot)
            angles.append(math.degrees(angle_rad))
    return angles


def analyze_mask(
    mask: np.ndarray,
    pixel_size: float = 1.0,
    min_root_area: int = 20,
    remove_small_objects: Optional[int] = None,
    morph_kernel_size: int = 0,
    morph_open: bool = False,
    min_root_diameter: float = 0.0,
) -> Dict:
    """
    对二值根系掩码（前景=1 或 255）进行形态分析。
    参考 rootmask：可选形态学去噪、最小根直径过滤。

    Args:
        mask: (H, W) 二值图，非零为前景
        pixel_size: 像素对应的物理长度（如 mm/pixel），结果会乘以此值
        min_root_area: 小于此面积的连通域不计入根
        remove_small_objects: 若给定，先去除面积小于此值的连通域再分析
        morph_kernel_size: 形态学核大小，>0 时先闭运算再可选开运算
        morph_open: 是否在闭运算后做开运算（可去细噪）
        min_root_diameter: 小于此平均宽度的根剔除（像素，0=不剔除）

    Returns:
        包含各项测量指标的字典（长度/面积/宽度均为物理单位，若 pixel_size=1 则为像素）
    """
    if mask.ndim > 2:
        mask = mask.squeeze()
    binary = (mask > 0).astype(np.uint8)

    if remove_small_objects is not None and remove_small_objects > 0:
        binary = morphology.remove_small_objects(
            label(binary),
            min_size=remove_small_objects,
            connectivity=2,
        )
        binary = (binary > 0).astype(np.uint8)

    if morph_kernel_size > 0:
        import cv2
        k = max(1, morph_kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        if morph_open:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = (binary > 0).astype(np.uint8)

    labeled = label(binary, connectivity=2)
    props = regionprops(labeled)

    root_lengths: List[float] = []
    root_widths: List[float] = []
    root_areas: List[float] = []

    for i, prop in enumerate(props):
        if prop.area < min_root_area:
            continue
        # 单根掩码
        root_mask = (labeled == prop.label).astype(np.uint8)
        # 骨架
        try:
            sk = morphology.skeletonize(root_mask, method="lee")
        except Exception:
            sk = morphology.skeletonize(root_mask.astype(bool)).astype(np.uint8)

        length_px = skeleton_length(sk, use_8conn=True)
        if length_px < 1e-6:
            length_px = float(np.sqrt(prop.area))  # 退化为近似
        area_px = float(prop.area)
        # 平均宽度：面积/长度（等效矩形宽）
        width_px = area_px / length_px if length_px > 0 else 0.0
        if min_root_diameter > 0 and width_px < min_root_diameter:
            continue

        root_lengths.append(length_px * pixel_size)
        root_widths.append(width_px * pixel_size)
        root_areas.append(area_px * pixel_size * pixel_size)

    # 全图骨架用于分支夹角（跨连通域的主根-侧根关系用全图骨架更合理）
    try:
        sk_full = morphology.skeletonize(binary, method="lee")
    except Exception:
        sk_full = morphology.skeletonize(binary.astype(bool)).astype(np.uint8)

    branch_pts = _branch_points_8conn(sk_full)
    lateral_angles_deg: List[float] = []
    for idx in range(branch_pts.shape[0]):
        br = (int(branch_pts[idx, 0]), int(branch_pts[idx, 1]))
        lateral_angles_deg.extend(_lateral_angles_at_branch(sk_full, br, branch_radius=5))

    # 汇总
    total_root_length = sum(root_lengths)
    total_root_area = sum(root_areas)
    n_roots = len(root_lengths)

    result = {
        "num_roots": n_roots,
        "total_root_length": total_root_length,
        "total_root_area": total_root_area,
        "mean_root_length": float(np.mean(root_lengths)) if root_lengths else 0.0,
        "longest_root_length": float(max(root_lengths)) if root_lengths else 0.0,
        "shortest_root_length": float(min(root_lengths)) if root_lengths else 0.0,
        "longest_root_width": float(max(root_widths)) if root_widths else 0.0,
        "shortest_root_width": float(min(root_widths)) if root_widths else 0.0,
        "mean_root_width": float(np.mean(root_widths)) if root_widths else 0.0,
        "lateral_root_angles_deg": lateral_angles_deg,
        "mean_lateral_angle_deg": float(np.mean(lateral_angles_deg)) if lateral_angles_deg else 0.0,
        "min_lateral_angle_deg": float(min(lateral_angles_deg)) if lateral_angles_deg else 0.0,
        "max_lateral_angle_deg": float(max(lateral_angles_deg)) if lateral_angles_deg else 0.0,
    }
    return result


def analyze_mask_image(
    mask_path: str,
    pixel_size: float = 1.0,
    min_root_area: int = 20,
    remove_small_objects: Optional[int] = None,
    morph_kernel_size: int = 0,
    morph_open: bool = False,
    min_root_diameter: float = 0.0,
) -> Dict:
    """从掩码图像路径读入并分析，返回同上。"""
    import cv2

    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    return analyze_mask(
        img,
        pixel_size=pixel_size,
        min_root_area=min_root_area,
        remove_small_objects=remove_small_objects,
        morph_kernel_size=morph_kernel_size,
        morph_open=morph_open,
        min_root_diameter=min_root_diameter,
    )
