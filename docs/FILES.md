# 文件索引

全部实验脚本位于 `LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task/`。

| 文件 | 职责 |
|---|---|
| `robotnik_arm_control.py` | 原专家底盘状态机、机械臂控制及固定手指辅助函数 |
| `robotnik_arm_task_3.py` | 单目标专家采集，保存 23D/11D 轨迹及末端位置，连续保持 1 s |
| `collect_expert_ring.py` | 原 931 个环状目标批量采集；沿用配置测量噪声，额外力矩噪声为 0 |
| `collect_expert_ring_lowz_supplement.py` | 50 个新 XY × 4 个连续低高度的 200 条补采 |
| `split_expert_data_ring.py` | 按目标/高度/半径划分原 931，保留重合旧目标归属 |
| `prepare_expert_data_ring_r08.py` | 原实验的 ≥0.8 m 筛选清单，含历史同目标比较种子 |
| `merge_expert_data_ring_lowz.py` | 原 798 + 新 200 合并为 998；保留旧归属和新增 XY 分组 |
| `robotnik_bodyframe_features.py` | 23D + 末端位置 → 26D 的唯一共享转换实现 |
| `robotnik_navigation_policy.py` | 8D 导航特征、阶段重建、停车点几何和底盘 MLP |
| `train_bc_26d_noise_ring_bodyframe.py` | 从头训练原 26D 联合网络，作为机械臂初始模型 |
| `train_nav_bc_ring_r08.py` | 从头训练 8D/4D 底盘网络；不更新机械臂 |
| `finetune_arm_bc_ring_r08.py` | 固定底盘，只按 7 维力矩损失微调原机械臂 |
| `finetune_arm_bc_ring_lowz_supplement.py` | 当前补采版机械臂微调；固定同一个底盘 |
| `arm_finetune_experiment.py` | 首次机械臂微调与底盘模型的绑定核验 |
| `arm_lowz_supplement_experiment.py` | 当前补采模型、父模型、底盘、998 清单的绑定核验 |
| `test_bc_26d_ring_bodyframe.py` / `batch_test_bc_26d_ring_bodyframe.py` | 原联合网络的单点/批量测试，历史依赖核验使用 |
| `test_bc_split_nav_arm.py` / `batch_test_bc_split_nav_arm.py` | 新底盘配原机械臂的单点/批量测试 |
| `test_bc_split_nav_arm_finetuned.py` / `batch_test_bc_split_nav_arm_finetuned.py` | previous 组合的单点/批量测试 |
| `test_bc_split_nav_arm_lowz_supplement.py` / `batch_test_bc_split_nav_arm_lowz_supplement.py` | 当前 supplement 组合的单点/批量测试 |
| `demo_robotnik_bc.py` | 手动/随机目标 GUI 演示，距离范围和高度模式可选 |
| `configs/robotnik_arm.json` | **专家采集配置**，测量噪声打开，参数值 0.001 |

发布辅助工具：

| 文件 | 职责 |
|---|---|
| `tools/prepare_workspace.py` | 从可迁移快照准备本机运行目录；不修改原实验 |
| `tools/relocate.py` | 迁移元数据路径，按父子依赖更新哈希；不改模型张量 |
| `tools/verify_release.py` | 校验代码、样本、模型和清单；可进一步检查完整数据 |
| `tools/export_dataset.py` | 按固定清单导出原始数据与校验索引，避免 Git 提交大型轨迹 |
| `tools/rebuild_splits.py` | 在单独输出目录重建清单，使用现有数据，拒绝覆盖冻结运行目录 |
| `tools/train_bodyframe.py` | 给原预训练入口增加清单、轮数与输出路径参数，不改训练算法 |
| `tools/collect_supplement.py` | 给原补采入口增加底盘清单路径参数，不改专家控制 |

`artifacts/RoboEnv/configs/robotnik_arm.json` 是**推理/评估配置快照**，与项目目录的专家采集配置分开保留。测试还会按 `--sensor-noise off` 在内存关闭测量噪声，不修改磁盘配置。

`environment/` 和 `artifacts/` 是固定发布快照；训练输出、原始数据、GUI 演示结果都写入 Git 忽略的运行目录，避免混入提交。
