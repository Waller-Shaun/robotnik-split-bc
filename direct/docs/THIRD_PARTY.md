# 来源与修改说明

本项目是 COMP0245 课程实验基础上的小组行为克隆实验整理，不是自行开发的仿真引擎。

| 组件 | 来源 | 本仓库的处理 |
|---|---|---|
| 课程任务与原专家控制逻辑 | [VModugno/LAB_SESSION_COMP_0245_2026_PUBLIC](https://github.com/VModugno/LAB_SESSION_COMP_0245_2026_PUBLIC/tree/main/final/robotnik_arm_task) | 保留控制辅助函数；在采集副本加入记录、阶段输入、保持段等功能 |
| 环境、机器人资产 | [VModugno/RoboEnv](https://github.com/VModugno/RoboEnv) | 安装时单独获取；不把完整上游仓库或网格资产提交进本仓库 |
| 仿真和动力学接口 | [VModugno/simulation_and_control](https://github.com/VModugno/simulation_and_control) | RoboEnv 子模块，通过 editable 包安装使用 |
| 网络和优化器 | [PyTorch](https://pytorch.org/) | 独立训练与推理 |
| 原始专家轨迹、网络、指标与新脚本 | 本组实验 | 发布当前流程精选副本、模型及固定归属清单 |

依赖版本：RoboEnv commit `7bb49439677c2ac111347b40e997025ce87bff50`；simulation_and_control commit `df5850526affcc33b25b15d3096c4d21b614bacd`。环境精确求解信息见 `environment/pixi.lock`。

当前代码保留原实验文件的注释和命名，并非统一风格的框架重写。发布只增加目录准备、路径迁移、文档及校验工具，保持原模型输入输出和控制逻辑。四个检查点的序列化元数据因路径/指纹迁移改变，但网络张量和归一化数组不变。

第三方组件应遵循各自项目的许可及课程使用要求；本仓库不为第三方材料追加或替换许可，没有把整个仓库统一声明为 MIT。贡献者在进一步分发时应保留来源说明。
