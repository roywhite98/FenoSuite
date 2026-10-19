# -*- coding: utf-8 -*-
"""单图全流程编排（来自 rootmask，配置从 plant_phenotype.root_phenotype.config 读取）。"""
import logging
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

from . import preprocess
from . import stats
from . import tree
from .phenotyping_config import PhenotypingConfig
from . import pipeline_stages
from .pipeline_profile import get_timer, profiling_enabled, reset_timer
from .skeleton_cache import clear_skeleton_cache


def _get_logger():
    name = "plant_phenotype.root_phenotype.core.pipeline"
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.DEBUG)
    try:
        repo_root = Path(__file__).resolve().parents[2].parent
        log_dir = repo_root / "workspace" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "analysis.log"
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        log.addHandler(fh)
    except Exception:
        pass
    return log


def _get_config(config_overrides: Optional[dict] = None) -> dict:
    return PhenotypingConfig.from_dict(config_overrides).as_dict()


def _get_pcfg(config_overrides: Optional[dict] = None) -> PhenotypingConfig:
    return PhenotypingConfig.from_dict(config_overrides)


@dataclass
class PipelineResult:
    mask: np.ndarray
    skeleton: Any
    G: Any
    endpoints: List[Tuple[int, int]]
    junctions: List[Tuple[int, int]]
    endpoint_ids: Dict[Tuple[int, int], int]
    junction_ids: Dict[Tuple[int, int], int]
    main_path: List[Tuple[int, int]]
    main_roots_all: List[List[Tuple[int, int]]] = field(default_factory=list)
    edge_level: Dict = field(default_factory=dict)
    summary_df: Any = None
    detail_df: Any = None
    roots: List[Dict[str, Any]] = field(default_factory=list)
    original_image: Optional[np.ndarray] = None
    error: Optional[str] = None


def _imread_unicode(path: str, flags: int = None):
    import cv2
    if flags is None:
        flags = cv2.IMREAD_COLOR
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    return img


def _load_mask(path: str) -> np.ndarray:
    import cv2
    img = _imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取: {path}")
    return img


def _union_mask_from_seeds(
    preprocessed_mask: np.ndarray,
    selected_seeds: List[Tuple[int, int]],
    log,
) -> Optional[np.ndarray]:
    import cv2
    union = np.zeros_like(preprocessed_mask, dtype=np.uint8)
    binary = (np.asarray(preprocessed_mask) > 0).astype(np.uint8)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for row, col in selected_seeds:
        if not (0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]):
            return None
        label_at = labels[row, col]
        if label_at == 0:
            return None
        union[labels == label_at] = 255
    return union


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    import cv2
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    num_labels, labels, stats_arr, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    areas = stats_arr[1:, cv2.CC_STAT_AREA]
    best_label = 1 + int(np.argmax(areas))
    out = np.zeros_like(binary, dtype=np.uint8)
    out[labels == best_label] = 255
    return out


def run_one(
    mask_path: str,
    original_image_path: Optional[str] = None,
    selected_seeds: Optional[List[Tuple[int, int]]] = None,
    preprocessed_mask: Optional[np.ndarray] = None,
    batch_mode: bool = False,
    config_overrides: Optional[dict] = None,
) -> PipelineResult:
    pcfg = _get_pcfg(config_overrides)
    if profiling_enabled():
        reset_timer()
    timer = get_timer()

    log = _get_logger()
    log.info("======== 开始分析: %s ========", mask_path)

    try:
        with timer.stage("preprocess"):
            if preprocessed_mask is not None:
                cleaned = np.asarray(preprocessed_mask)
                if cleaned.dtype != np.uint8 or cleaned.max() <= 1:
                    cleaned = (cleaned > 0).astype(np.uint8) * 255
                mask = cleaned
            else:
                mask = _load_mask(mask_path)
                log.info(
                    "加载 mask: shape=%s, 非零像素=%d",
                    getattr(mask, "shape", None),
                    int(np.count_nonzero(mask)) if mask.size else 0,
                )
                cleaned = preprocess.denoise_and_remove_small_components(
                    mask,
                    morph_kernel_size=pcfg.morph_kernel_size,
                    min_component_area=pcfg.min_component_area,
                    morph_open=pcfg.morph_open,
                )
                mask = cleaned
            log.info("预处理后: 非零像素=%d", int(np.count_nonzero(cleaned)))
    except Exception as e:
        log.exception("加载/预处理失败: %s", e)
        return PipelineResult(
            mask=np.array([]), skeleton=np.array([]), G=None,
            endpoints=[], junctions=[], endpoint_ids={}, junction_ids={}, main_path=[],
            error=str(e),
        )

    seeds = selected_seeds if selected_seeds else []
    if not seeds and not batch_mode:
        return PipelineResult(
            mask=cleaned,
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
            original_image=None,
        )

    selection_mask = cleaned
    connected_skel = None
    connected_cache_key = None
    if pcfg.connect_broken_roots:
        # 先在完整掩码上连接断片，再按种子/最大连通域选择目标根。
        # 提前筛选会丢掉尚未连通的断片，批量与交互模式都会无法补全根系。
        try:
            with timer.stage("connect_skeleton"):
                selection_mask, connected_skel, connected_cache_key = pipeline_stages.skeletonize_union_mask(
                    mask_path, cleaned, pcfg, None
                )
        except Exception as e:
            log.exception("断根连接失败: %s", e)
            return PipelineResult(
                mask=cleaned, skeleton=np.array([]), G=None,
                endpoints=[], junctions=[], endpoint_ids={}, junction_ids={}, main_path=[],
                error=str(e),
            )

    if not seeds:
        union_mask = _largest_component_mask(selection_mask)
        if np.count_nonzero(union_mask) == 0:
            log.warning("批量模式: 无连通域，返回空结果")
            return PipelineResult(
                mask=cleaned, skeleton=np.array([]), G=None,
                endpoints=[], junctions=[], endpoint_ids={}, junction_ids={}, main_path=[],
                error="无连通域",
            )
    else:
        union_mask = _union_mask_from_seeds(selection_mask, seeds, log)
        if union_mask is None or np.count_nonzero(union_mask) == 0:
            log.warning("点选区域无效或为空")
            return PipelineResult(
                mask=cleaned,
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
                original_image=None,
                error="点选位置为背景或无效",
            )

    try:
        seed_key = tuple(sorted(seeds)) if seeds else None
        with timer.stage("skeleton"):
            if connected_skel is not None:
                skel = connected_skel & (union_mask > 0)
                skel_cache_key = connected_cache_key + ("component_selection", seed_key)
            else:
                union_mask, skel, skel_cache_key = pipeline_stages.skeletonize_union_mask(
                    mask_path, union_mask, pcfg, seed_key
                )
        skel_count = int(np.count_nonzero(skel)) if skel.size else 0
        if skel_count == 0:
            log.warning("骨架为空")
            return PipelineResult(
                mask=cleaned,
                skeleton=np.array([]),
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
                original_image=None,
                error="选中区域骨架为空",
            )

        with timer.stage("graph"):
            G, endpoints, junctions, endpoint_ids, junction_ids = pipeline_stages.build_graph_from_skeleton(
                skel, union_mask, pcfg, skel_cache_key=skel_cache_key
            )
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        log.info("拓扑图: 节点数=%d, 边数=%d, 端点数=%d, 交点数=%d", n_nodes, n_edges, len(endpoints), len(junctions))
        if n_edges == 0:
            log.warning("图中边数为 0，无法识别主干")

        with timer.stage("per_component"):
            roots_list, edge_level, main_path_first = pipeline_stages.analyze_components(
                G, union_mask, pcfg, seeds, batch_mode
            )

        level_counts = {}
        for (_, _), lev in edge_level.items():
            level_counts[lev] = level_counts.get(lev, 0) + 1
        log.info(
            "分级: 连通分量=%d, edge_level 边数=%d, 各级数量=%s",
            len(roots_list),
            len(edge_level),
            level_counts,
        )

        with timer.stage("stats"):
            summary_df, detail_df = (
                stats.compute_all_stats(
                    G,
                    edge_level,
                    min_root_length=pcfg.min_root_length,
                    min_root_diameter=pcfg.min_root_diameter,
                    apply_zscore=pcfg.zscore,
                )
                if roots_list
                else (None, None)
            )

        if profiling_enabled():
            timer.log_summary(prefix=f"{mask_path} ")

        original_image = None
        if original_image_path:
            try:
                import cv2
                original_image = _imread_unicode(original_image_path)
                if original_image is not None:
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            except Exception:
                pass

        log.info("分析完成: 主根边数=%d, roots=%d", level_counts.get(1, 0), len(roots_list))
        return PipelineResult(
            mask=cleaned,
            skeleton=skel,
            G=G,
            endpoints=endpoints,
            junctions=junctions,
            endpoint_ids=endpoint_ids,
            junction_ids=junction_ids,
            main_path=main_path_first,
            main_roots_all=[r["main_path"] for r in roots_list],
            edge_level=edge_level,
            summary_df=summary_df,
            detail_df=detail_df,
            roots=roots_list,
            original_image=original_image,
        )
    except Exception as e:
        import traceback
        log.exception("分析异常: %s", e)
        return PipelineResult(
            mask=cleaned,
            skeleton=np.array([]),
            G=None,
            endpoints=[],
            junctions=[],
            endpoint_ids={},
            junction_ids={},
            main_path=[],
            error=f"{e}\n{traceback.format_exc()}",
        )


def _path_from_stroke(
    sub: nx.Graph,
    stroke_points: List[Tuple[int, int]],
    max_sample: int = 150,
    search_radius_px: float = 24.0,
    use_graph_weight: bool = True,
) -> Optional[List[Tuple[int, int]]]:
    """
    根据用户描绘的轨迹，在子图中构造主根路径：每个采样点在半径 r 内取最近骨架节点
    （无命中则回退全局最近），相邻锚点之间用 Dijkstra 按边权 w 连接（可选边数最少）。
    """
    from .nearest import build_kdtree, nearest_in_nodes

    nodes = list(sub.nodes())
    if not nodes or not stroke_points:
        return None
    if len(stroke_points) > max_sample:
        indices = [0] + [int((i + 1) * (len(stroke_points) - 1) / (max_sample - 1)) for i in range(max_sample - 2)] + [len(stroke_points) - 1]
        stroke_points = [stroke_points[i] for i in indices]

    tree, node_list = build_kdtree(nodes)
    radius = search_radius_px if search_radius_px and search_radius_px > 0 else None

    node_seq = []
    for (r, c) in stroke_points:
        n = nearest_in_nodes(node_list, r, c, radius_px=radius, tree=tree)
        if n is not None and (not node_seq or node_seq[-1] != n):
            node_seq.append(n)
    if not node_seq:
        return None
    if len(node_seq) == 1:
        return node_seq

    path = [node_seq[0]]
    for i in range(1, len(node_seq)):
        u, v = path[-1], node_seq[i]
        if u == v:
            continue
        try:
            if use_graph_weight:
                seg = nx.shortest_path(sub, u, v, weight="weight")
            else:
                seg = nx.shortest_path(sub, u, v)
            for x in seg[1:]:
                path.append(x)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    return path if len(path) >= 2 else None


def apply_main_root_correction_with_stroke(
    result: PipelineResult,
    stroke_points: List[Tuple[int, int]],
    config_overrides: Optional[dict] = None,
) -> Optional[PipelineResult]:
    """
    用户通过描绘主根轨迹引导主根确定：在半径 r 内锚定骨架节点，段间按 w 加权最短路拼接。
    返回更新后的 PipelineResult，若无法解析则返回 None。
    """
    G = getattr(result, "G", None)
    roots = getattr(result, "roots", None) or []
    if not G or not roots or len(stroke_points) < 2:
        return None
    pcfg = _get_pcfg(config_overrides)
    cfg = pcfg.as_dict()

    components = list(nx.connected_components(G))
    best_comp_idx = -1
    best_score = 1e9
    for idx, comp in enumerate(components):
        total_d = sum(
            min((r - nr) ** 2 + (c - nc) ** 2 for (nr, nc) in comp)
            for (r, c) in stroke_points
        )
        if total_d < best_score:
            best_score = total_d
            best_comp_idx = idx
    if best_comp_idx < 0:
        return None
    comp = components[best_comp_idx]
    sub = G.subgraph(comp).copy()
    path = _path_from_stroke(
        sub,
        stroke_points,
        search_radius_px=float(cfg.get("stroke_search_radius", 24.0)),
        use_graph_weight=bool(cfg.get("stroke_use_graph_weight", True)),
    )
    if not path:
        return None

    edge_level_comp = tree.build_rooted_tree(
        sub, path,
        min_root_length=cfg.get("min_root_length", 5.0),
        min_root_diameter=cfg.get("min_root_diameter", 1.0),
    )
    summary_comp, detail_comp = stats.compute_all_stats(
        sub, edge_level_comp,
        min_root_length=cfg.get("min_root_length", 5.0),
        min_root_diameter=cfg.get("min_root_diameter", 1.0),
        apply_zscore=cfg.get("zscore", True),
    )
    new_roots = list(roots)
    new_roots[best_comp_idx] = {
        "summary_df": summary_comp,
        "detail_df": detail_comp,
        "main_path": path,
        "edge_level": edge_level_comp,
    }
    new_edge_level = {}
    for r in new_roots:
        new_edge_level.update(r["edge_level"])
    summary_df, detail_df = stats.compute_all_stats(
        G, new_edge_level,
        min_root_length=cfg.get("min_root_length", 5.0),
        min_root_diameter=cfg.get("min_root_diameter", 1.0),
        apply_zscore=cfg.get("zscore", True),
    )
    main_path_first = new_roots[0]["main_path"] if new_roots else []
    return PipelineResult(
        mask=result.mask,
        skeleton=result.skeleton,
        G=G,
        endpoints=result.endpoints,
        junctions=result.junctions,
        endpoint_ids=result.endpoint_ids,
        junction_ids=result.junction_ids,
        main_path=main_path_first,
        main_roots_all=[r["main_path"] for r in new_roots],
        edge_level=new_edge_level,
        summary_df=summary_df,
        detail_df=detail_df,
        roots=new_roots,
        original_image=getattr(result, "original_image", None),
    )
