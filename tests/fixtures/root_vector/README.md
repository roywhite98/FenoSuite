# 根系矢量回归样本

`combo_012.png` 来自用户指定的本地合成数据集 `synthetic_trunk_branch`，原文件为
`F:/Tools/segmentation_core/archived/data/synthetic_trunk_branch/masks/combo_012.png`。
该文件为未经修改的二值掩码，512×512。生成种子为 12，主干真值两端（x, y）
为 (166, 29) 和 (210, 493)。自动骨架端点允许少量像素回缩。

该样本有主干缺失、交叉和侧根绕路。对应测试验证真实主干两端不会被侧根
端点取代，并验证疑似绕行的主根及其侧根归属进入待确认状态；不能据此认为
中间缺失几何或所有父子关系已经恢复。
