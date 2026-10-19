# 植物表型分析软件

> 版本 v1.0.0 · PyQt6 桌面应用 · 支持 Windows / macOS / Linux

基于深度学习（PyTorch）与图算法（NetworkX）的植物表型分析一体化工具，集成**数据标注、模型训练、图像分割、根系表型**四大功能模块，适用于植物科研与育种数据处理场景。

---

## 目录

- [功能概览](#功能概览)
- [快速开始](#快速开始)
- [安装依赖](#安装依赖)
- [使用说明](#使用说明)
- [项目结构](#项目结构)
- [常见问题 FAQ](#常见问题-faq)
- [贡献指南](#贡献指南)

---

## 功能概览

| 模块 | 功能描述 |
|------|----------|
| **数据集** | 多数据集统一存放于 `workspace/datasets/`，各页共享当前数据集；导入、配对、标签映射、校验、格式转换与导出 |
| **数据标注** | 笔刷 / 橡皮 / 油漆桶标注工具，支持切块与还原，自动叠加已有掩码 |
| **模型训练** | 本地训练与远程服务两种模式，支持 YAML 配置预览编辑与断点续训 |
| **图像分割** | 单张预览与批量分割，结果自动保存为掩码图像 |
| **根系表型** | 骨架分析、拓扑统计、主根校正，支持导出 CSV / Excel 与可视化图 |
| **根系表型beta** | 方向场辅助矢量解析、主侧根父子关系、交汇校正与节点编辑，支持 JSON / CSV / Excel / SVG / PNG |

---

## 快速开始

> 以下 5 步即可完成安装并启动软件。

**第 1 步：克隆仓库**

```bash
git clone <仓库地址>
cd FenoSuite
```

**第 2 步：创建 Conda 环境**

```bash
conda env create -f environment.yml
```

**第 3 步：激活环境**

```bash
conda activate plant_phenotype
```

**第 4 步：（可选）安装 GPU 版 PyTorch**

如需使用 CUDA 加速，请先替换为 GPU 版 PyTorch（以 CUDA 12.1 为例）：

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**第 5 步：启动软件**

```bash
python run.py
```

启动后将显示包含数据集、数据标注、模型训练、图像分割、根系表型、根系表型beta 六个 Tab 的主界面。

---

## 安装依赖

根据使用场景选择对应的安装方式：

### 场景一：完整安装（GUI + 训练 + 分割 + 根系表型）

```bash
conda activate plant_phenotype
pip install -r requirements.txt
```

如需使用 DINOv3 模型，额外执行：

```bash
pip install -r requirements-dinov3.txt
```

### 场景二：仅 GUI 本地使用（不搭建远程服务）

与完整安装相同，使用 `requirements.txt` 即可。Flask 已包含在内，不影响本地运行。

```bash
pip install -r requirements.txt
```

### 场景三：远程 GPU 服务器训练

**SSH 方式（推荐）**：服务器上**不需要部署项目**，只要有一个装好 PyTorch 的 Python 环境（AutoDL、SeetaCloud 等云 GPU 镜像自带）。其余训练依赖可在界面中勾选「自动安装缺少的依赖」，或手动安装：

```bash
pip install numpy opencv-python-headless Pillow PyYAML albumentations segmentation-models-pytorch timm tqdm matplotlib seaborn
```

**HTTP 服务方式（可选）**：需要在服务器上常驻 Flask 服务。无需拷贝整个项目，在本机导出最小服务包后上传、解压即可：

```bash
python -m plant_phenotype.training.bundle --server -o fenosuite_server.zip   # 本机执行
pip install -r requirements-server.txt && python run_server.py              # 服务器上解压后执行
```

> **GPU 安装提示**：默认 pip 安装的 PyTorch 为 CPU 版。如需 GPU 加速，请参考[快速开始第 4 步](#快速开始)替换为对应 CUDA 版本。可通过以下命令诊断 CUDA 是否可用：
> ```bash
> python scripts/check_cuda.py
> ```

---

## 使用说明

### 数据集

1. 在「数据集」页点击「导入...」选择外部目录（同目录 `xxx.png` + `xxx_mask.png`，或 `images/` + `masks/`），确认配对与像素值映射后复制到软件数据目录；也可「新建」空数据集再添加图像。
2. 在列表中选中的数据集即为当前数据集，标注、训练、分割、根系表型页右侧的「数据集」面板同步切换。
3. 右侧「标签」「校验」「转换/导出」分别用于类别管理、问题检查和格式转换。

多类别/实例数据会受到保护，不会被当前的二值标注和训练功能覆盖。存储规范与限制见 [docs/datasets.md](docs/datasets.md)。

### 数据标注

<!-- 截图占位：数据标注界面截图，展示笔刷工具栏、画布区域与工作目录树 -->
> 📷 *截图示意：左侧工作目录树 + 中央标注画布 + 右侧工具栏（笔刷/橡皮/油漆桶）*

1. 在右侧「数据集」面板点击样本打开原图，已有掩码自动叠加；或切到「文件夹」选择工作目录按旧方式打开图像。
2. 选择标注工具（笔刷 / 橡皮 / 油漆桶 / 大津 / 区域增长），在画布上涂抹标注前景。
   - 快捷键：`[` 减小笔刷大小，`]` 增大笔刷大小（步长 5px）
3. 点击「保存」：数据集样本写回数据集并标为已标注；文件夹模式保存为 `<原图名>_mask.png`。
5. 如需切块处理大图，使用「切块」功能指定切块大小与前景比例阈值；使用「还原」将切块拼回整图。

### 模型训练

<!-- 截图占位：模型训练界面截图，展示配置路径、训练日志与进度条 -->
> 📷 *截图示意：配置文件路径输入 + 训练日志滚动区域 + 开始/停止按钮*

**本地训练：**

1. 选择 YAML 配置文件（可点击「预览」在线编辑）。
2. 「数据来源」留空即使用当前数据集（只用已标注样本）；点「浏览...」选择目录则使用该自定义目录（包含图像与对应 `*_mask.png`）。
3. 选择模型保存目录，可选「继续训练」指定已有 `.pth` 权重。
4. 点击「开始训练」，日志实时滚动显示，训练完成后显示 Dice 分数摘要。

**SSH 远程训练（推荐）：**

1. 在「模型训练」Tab 选择「SSH 远程」，在「连接」中直接粘贴登录指令（如 `ssh -p 2222 root@192.168.1.100`），或只填服务器 IP（默认端口 22、用户 root），然后填写密码或选择私钥。
2. 「远端 Python」填写装好 PyTorch 的解释器（默认 `python`）；「远端目录」建议放在数据盘，如 `/root/autodl-tmp/fenosuite`。
3. 点「测试连接」检查远端 Python、PyTorch、GPU 与依赖；点「开始训练」后，最小训练代码（约 30 个文件）、数据与配置自动上传，日志实时回传，结束后 `best_model.pth` 等产物下载到本地「保存目录」。
4. 远端训练在后台运行，网络断开不会中断；点「停止」会终止远端训练进程。

**HTTP 远程服务：**

1. 在远程机器上启动服务：`python run_server.py`（确保防火墙放行 8765 端口）。
2. 在「模型训练」Tab 选择「HTTP 服务」，填写服务地址 `http://<远程IP>:8765`。
3. 选择配置文件与数据目录，点击「开始训练」，数据自动打包上传。

### 图像分割

<!-- 截图占位：图像分割界面截图，展示模型路径、预览区域与批量分割进度 -->
> 📷 *截图示意：模型权重路径 + 单张预览画布（含文件名序号标签）+ 批量分割输出目录*

1. 选择模型权重文件（`.pth`）与配置文件（YAML）。
2. **单张预览**：选择图像目录，使用上/下翻页预览分割效果，状态栏显示 `文件名 [当前/总数]`。
3. **批量分割**：图像来源选当前数据集或自定义目录，点击「开始分割」，完成后显示成功/失败数量统计。
4. **结果入库**（可选）：写回当前数据集（只填充未标注样本，状态为「预测/候选」，不覆盖人工标注），或另存为新数据集供复核。

### 根系表型

<!-- 截图占位：根系表型界面截图，展示掩码图像、骨架可视化与统计表格 -->
> 📷 *截图示意：左侧掩码画布（含骨架叠加）+ 右侧参数面板 + 底部统计表格*

1. 在右侧「数据集」面板选择带掩码的样本（多类别数据集可选「分析类别」），或点击「加载掩码」选择根系掩码图像（`.png`）。
2. 点击「预处理」执行骨架化，完成后状态栏显示检测到的根系数量与主根长度。
3. 在画布上点击根系位置进行选根，支持多根选择与撤销。
4. 调整参数面板中的统计参数（最小根长、最小根径、Z-score 阈值）。
5. 点击「导出」将表型统计结果保存为 CSV / Excel，并生成可视化图。

---

### 根系表型beta

1. 打开独立的「根系表型beta」页，点击「选择掩码」载入二值根系掩码（推荐无损 PNG，非零像素为前景）；可另选同尺寸原图作底图。
2. 在「点选待分析根系」模式下点击根内任意位置，再点击「开始解析」。系统仅解析选中的根系，先连接短断口、清理毛刺，再识别主根与侧根；连接距离和去毛刺长度可调整。
3. 画布按根 ID 着色，右侧根树显示父子关系。对待确认交汇选择接续、分叉或穿越解释，核对画布端口预览后点击「确认该解释」。
4. 「编辑节点」支持拖动节点；右侧可修改直径和父根附着、解除附着、确认或删除根。操作支持撤销与重做。
5. 「保存 JSON」保留几何、父子关系和校正结果，可用「打开文档」继续；CSV/XLSX 输出测量表，SVG/PNG 输出图像。标定输入单位是 mm/px。

主根为红色、一级侧根为蓝色、更高级侧根为紫色，并显示对应名称。根级次以主根为 0、直接侧根为 1，依次递增；主根由长度、粗细和走向连续性自动推断，真实穿越、粘连及闭环可局部校正。直径是掩码距离变换估计，补接处和交汇处不可靠的直径记为缺失。详见 [beta 使用说明](docs/root-beta.md) 和 [架构设计](docs/ROOT_VECTOR_ARCHITECTURE.md)。

## 项目结构

```
FenoSuite/
├── README.md
├── pyproject.toml
├── requirements*.txt
├── plant_phenotype/            # 根目录下的应用包
│   ├── app/                    # 任务平台、当前数据集服务和内置插件
│   ├── datasets/               # 数据集存储、配对、导入、校验与转换
│   ├── gui/                    # 功能页面和控件
│   ├── annotation/             # 标注、SAM 和切块
│   ├── training/               # 训练逻辑和数据集
│   ├── segmentation/           # 图像分割推理
│   ├── models/                 # 共享模型
│   ├── root_phenotype/         # 根系表型分析
│   ├── server/                 # 远程训练服务
│   └── resources/              # 默认配置和应用图标
├── tests/                      # 自动化测试
├── scripts/                    # CUDA 与依赖检查
├── docs/                       # 软件文档
├── workspace/                  # 本地数据（datasets/ 数据集）和训练结果
├── run.py
└── run_server.py
```

## 常见问题 FAQ

**Q1：CUDA 不可用，训练/分割只能用 CPU，速度很慢怎么办？**

首先确认 CUDA 是否可用：

```bash
python scripts/check_cuda.py
```

如果显示 CUDA 不可用，需要安装 GPU 版 PyTorch。先卸载当前版本，再按 CUDA 版本重新安装：

```bash
pip uninstall torch torchvision -y
# CUDA 12.1 示例，请根据实际 CUDA 版本调整
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

安装完成后重新运行 `check_cuda` 验证。如果机器没有 NVIDIA GPU，则只能使用 CPU 模式，可在配置文件中将 `device` 设为 `cpu`。

---

**Q2：安装依赖时报 `stringzilla` 编译错误怎么办？**

`stringzilla` 在 Windows 上需要 C++ 编译环境。有两种解决方案：

- **方案 A（推荐）**：使用 Conda 环境安装，Conda 会提供预编译的二进制包：
  ```bash
  conda env create -f environment.yml
  conda activate plant_phenotype
  ```
- **方案 B**：安装 [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)，然后重新执行 `pip install -r requirements.txt`。

---

**Q3：远程训练连接失败（Connection refused / Timeout）怎么办？**

SSH 方式：先在终端用同样的 `ssh -p <端口> <用户>@<主机>` 确认能登录；提示「未安装 PyTorch」时，把「远端 Python」改为实际环境的解释器路径（可在服务器上执行 `which python` 查看）。

HTTP 服务方式按以下步骤逐一排查：

1. 确认远程机器已启动服务：`python run_server.py`，终端应显示 `Running on http://0.0.0.0:8765`。
2. 确认防火墙已放行 8765 端口（Linux：`sudo ufw allow 8765`；Windows：在防火墙入站规则中添加）。
3. 确认服务地址填写正确，格式为 `http://<远程IP>:8765`（不要遗漏 `http://`）。
4. 如果远程与本机是同一台机器，地址填 `http://localhost:8765`。
5. 可用 `curl http://<远程IP>:8765` 测试连通性。

---

**Q4：训练时报 `DataLoader` 死锁或进程卡住怎么办？**

这是 Windows 平台多进程数据加载的已知问题。在 YAML 配置文件中将 `num_workers` 设为 `0` 即可切换为单进程加载：

```yaml
num_workers: 0
```

---

**Q5：批量分割时报「模型文件加载失败」怎么办？**

常见原因及解决方法：

- **权重文件路径错误**：确认 `.pth` 文件路径存在且可读。
- **配置文件与权重不匹配**：确保 YAML 配置中的模型架构（`model_type`）与训练时一致。
- **PyTorch 版本不兼容**：训练与推理环境的 PyTorch 版本应保持一致，可用 `python -c "import torch; print(torch.__version__)"` 确认。

---

**Q6：根系表型分析时内存溢出或速度极慢怎么办？**

当根系图像分辨率较高（超过 2048×2048）或根系拓扑图节点数超过 5000 时，可能出现此问题：

- 适当降低输入掩码图像的分辨率后再分析。
- 在参数面板中增大「最小根长」阈值，减少噪声根系节点数量。
- 仅修改统计参数（最小根长、最小根径、Z-score）时，软件会复用已缓存的骨架，无需重新计算。

---

**Q7：安装后运行 `python run.py` 报 `ModuleNotFoundError` 怎么办？**

确认已激活正确的 Conda 环境，并在仓库根目录执行：

```bash
conda activate plant_phenotype
pip install -e .   # 以开发模式安装根目录下的应用包
python run.py
```

---

## 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发环境搭建

```bash
git clone <仓库地址>
cd plant_phenotype
conda env create -f environment.yml
conda activate plant_phenotype
pip install -e .
```

### 提交规范

- **Bug 修复**：在 Issue 中描述复现步骤，提交 PR 时关联对应 Issue。
- **新功能**：请先在 Issue 中讨论方案，获得认可后再开始开发。
- **代码风格**：遵循 PEP 8，函数与类需添加 docstring，类型注解尽量完整。
- **提交信息**：使用简洁的中文或英文描述，格式参考：`fix: 修复批量分割时的内存泄漏` 或 `feat: 添加根系直径统计功能`。

### 目录约定

- GUI 相关代码放在 `plant_phenotype/gui/`
- 算法核心代码放在对应模块的 `core/` 子目录
- 新增功能请同步在 `test/` 目录添加对应测试

如有问题，欢迎通过 Issue 反馈。
