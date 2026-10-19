# -*- coding: utf-8 -*-
"""标注与预处理默认配置（来自 dataset-tool）。"""

# 默认切块大小（像素）
DEFAULT_TILE_SIZE = 512

# mask 密度过滤阈值：白色像素占比低于此值的切块标记为不参与训练
DEFAULT_MIN_FOREGROUND_RATIO = 0.01

# 标注层默认前景色 (R, G, B)
DEFAULT_ANNOTATION_COLOR = (0, 255, 0)

# 预测叠加层显示颜色 (R, G, B)
PREDICTION_OVERLAY_COLOR = (255, 0, 255)

# 支持的图像扩展名（画布显示 TIFF/JPG/GIF 等，mask 仍保存为 PNG）
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".gif")

# mask 文件后缀
MASK_SUFFIX = "_mask"

# 魔法棒：模型管理默认子目录名（相对项目根）
DEFAULT_MODEL_DIR_NAME = "models"

# 支持的模型文件扩展名（下拉列表会扫描这些）
MODEL_EXTENSIONS = (".onnx", ".pt", ".pth")

# 魔法棒区域提议：以点击为中心裁剪的候选区域边长（像素）
DEFAULT_RPN_CROP_SIZE = 512
