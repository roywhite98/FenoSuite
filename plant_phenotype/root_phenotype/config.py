# -*- coding: utf-8 -*-
"""根系表型默认参数（来自 rootmask）。"""

# 预处理
PREPROCESS = {
    "morph_kernel_size": 3,
    "min_component_area": 50,
    "morph_open": False,
}

# 断根连接
CONNECT_BROKEN_ROOTS = {
    "enabled": True,
    "max_search": 120,
    "angle_tol_deg": 25.0,
    "min_gap": 3.0,
    "max_gap": 200.0,
}

# 拓扑：关键点判定
TOPOLOGY = {
    "endpoint_neighbors": 1,
    "junction_min_neighbors": 3,
    "curvature_angle_deg": None,
    # 骨架端部短毛刺剪枝：删除叶到分叉前长度 ≤ 该值的链；0 表示关闭
    # 默认 8：可显著削减水稻等根毛稠密样本的交点数，加速建图
    "spur_max_length": 8,
}

# 根尖单点化：图端点空间聚类合并（兜底）
ENDPOINT_MERGE = {
    "enabled": False,
    # 欧氏距离 ≤ 该值的叶端点（且共连同一分叉）合并为一个根尖
    "max_dist_px": 4.0,
}

# 解交叉（伪 X 交叉拆点）
DECROSS = {
    "enabled": False,
    "tensor_sigma": 1.0,
    "min_cluster_sep": 0.35,
    "geom_tensor_blend": 0.5,
}

# 多尺度 ridge（边 ∫R ds）
RIDGE = {
    "enabled": False,
    "sigmas": (1.0, 2.0, 4.0),
}

# 拓扑简化（叶边剪枝）
SIMPLIFY = {
    "enabled": False,
    "min_score": 0.0,
    "score_attr": "ridge_integral",
}

# 图特征（betweenness 无下游消费，默认关闭以提速）
GRAPH_FEATURES = {
    "compute_betweenness": False,
}

# 边权权重
EDGE_WEIGHT = {
    "length": 1.0,
    "diameter": 0.5,
    "curvature": -0.2,
}

# 分叉竞争主根（多主根 / 局部得分，不依赖 Strahler）
PRIMARY_ROOT = {
    "enabled": False,
    "alpha": 1.0,
    "beta": 0.5,
    "gamma": 0.8,
    "tau": 0.75,
    "d_attr": "diameter",
    "lambda1": 1.0,
    "lambda2": 0.5,
    "max_level": 3,
    "diameter_ratio_min": 0.3,
    "diameter_floor": 1.0,
    "normalize_score": True,
}

# 主干识别
MAIN_ROOT = {
    "max_main_roots": 1,
    # True：端点对之间在复合边权 w 上求近似最长路径（反权 Dijkstra，与文稿一致）
    "inverse_weight_for_longest": True,
    # False 时使用 length + λ·曲率 的平滑最短路；端点对仍为像素欧氏最远（见 main_root.find_smooth_main_path）
    "curvature_penalty": 1.0,
    # 用户描绘主根：锚点仅在半径 r（像素）内取最近骨架节点；若无命中则回退全局最近
    "stroke_search_radius": 24.0,
    # 锚点序列段间连接是否按边权 w 做 Dijkstra（False 为边数最少）
    "stroke_use_graph_weight": True,
}

# 数据清洗与统计
STATS = {
    "min_root_length": 5.0,
    "min_root_diameter": 1.0,
    "zscore": True,
}

# 可视化配色（RGB 0-255）
COLORS = {
    "main_root": (80, 80, 255),         # #5050ff 主根
    "secondary_root": (80, 200, 255),   # #50c8ff 二级根
    "tertiary_root": (80, 200, 255),    # #50c8ff 侧根
    "junction_circle": (0, 252, 252),  # #00fcfc 交点
    "endpoint_circle": (255, 200, 0),  # #ffc800 根尖
    "skeleton_default": (200, 200, 200),  # 骨架（灰）
}
LINE_WIDTH_MAIN = 3
