# 项目目录

```text
FenoSuite/
├── plant_phenotype/           # 应用与业务代码
│   ├── app/                    # 任务平台和内置插件
│   ├── gui/                    # 主窗口与功能页面
│   ├── annotation/             # 标注、切块与 SAM
│   ├── training/               # 模型训练与训练数据集
│   ├── segmentation/           # 单张及批量分割
│   ├── models/                 # 共享模型定义
│   ├── root_phenotype/         # 根系分析和统计
│   ├── server/                 # 远程训练服务
│   └── resources/              # 默认 YAML 配置与图标
├── tests/                      # 自动化测试和小型样本
├── scripts/                    # CUDA 检查、依赖安装等维护脚本
├── docs/                       # 软件文档
├── workspace/                  # 本地数据与运行产物
├── run.py                      # GUI 入口
└── run_server.py               # 服务入口
```

Python 包直接位于仓库根目录。训练与分割共享 `models/` 和 `image_io.py`；GUI、插件、服务端均从对应业务包调用功能。论文稿、图表和制图脚本不属于软件运行目录。
