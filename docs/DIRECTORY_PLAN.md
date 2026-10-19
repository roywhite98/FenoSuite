# FenoSuite 目录整理

应用包现已位于仓库根目录，不再使用 `src/`。论文稿、论文图表、制图脚本及相关素材已删除。

```text
FenoSuite/
├── plant_phenotype/
│   ├── app/                    # 任务平台、插件
│   ├── gui/                    # 保持扁平
│   ├── annotation/
│   ├── training/               # 训练和数据集
│   ├── segmentation/           # 推理
│   ├── models/                 # 训练/推理共用模型
│   ├── root_phenotype/
│   ├── server/
│   └── resources/              # 默认 YAML、图标
├── tests/
├── scripts/
├── docs/
├── workspace/                  # 本地数据和运行产物
├── run.py
├── run_server.py
└── 项目配置与说明文件
```

原 `segment_task/` 已拆到 `training/`、`segmentation/`、`models/` 和 `root_phenotype/`；`pheno_platform/` 改为 `app/`，四个内置插件改为 `app/plugins/` 下的功能名；默认配置和图标位于包内 `resources/`。

入口、打包配置、模块导入、测试路径及说明文档已同步更新。训练和预测共享的图像扩展名约定放在 `image_io.py`。
