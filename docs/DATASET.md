# 数据结构与固定划分

## 一条文件就是一条完整轨迹

NPZ 文件名按采集时间生成。文件名不是类别标签；目标取自文件内部 `target`，固定归属来自 `data_split.json`。

| 字段 | 形状 | 含义 |
|---|---|---|
| `observations` | `(N,23)` | 控制前的当前观测 |
| `actions` | `(N,11)` | 与当前观测配对的专家动作标签 |
| `ee_positions` | `(N,3)` | 同一步的末端世界坐标，用于派生位置误差 |
| `timestamps` | `(N,)` | 仿真时刻，秒 |
| `target` | `(3,)` | 固定目标世界坐标，米 |

还保留实际施加动作、扰动、底盘姿态、期望关节状态、末端误差和到达/保持状态等诊断字段；示例文件的准确键名可以用下面的代码查看。

```python
import numpy as np
with np.load('examples/expert_sample.npz', allow_pickle=False) as d:
    for name in d.files:
        print(name, d[name].shape, d[name].dtype)
```

仿真步长 0.001 s。数百万训练样本指每条轨迹的时间步总和，不代表数百万个独立目标。连续时间步高度相关；因此不能按帧随机切分训练/验证/测试。

## 磁盘观测 23 维（零起始列索引）

| 列 | 维度 | 内容 |
|---|---:|---|
| `0:3` | 3 | 底盘世界位置 XYZ，m |
| `3:5` | 2 | sin(yaw)、cos(yaw) |
| `5:12` | 7 | 关节角，rad |
| `12:19` | 7 | 关节角速度，rad/s |
| `19:22` | 3 | 目标世界位置 XYZ，m |
| `22` | 1 | `base_active`：导航进行中 1，导航完成 0 |

`base_active` 是规则导航状态，不是对实际轮速做阈值检测；它也不是学习模型输出的分类标签。

## 机械臂网络派生输入 26 维

由 `robotnik_bodyframe_features.py` 从原 23 维和对齐的 `ee_positions` 在内存计算：

```text
原 0:19 列保留
19:22 = Rz(-yaw) × (目标世界位置 - 底盘世界位置)
22    = 原 base_active
23:26 = Rz(-yaw) × (目标世界位置 - 末端世界位置)
```

这是按底盘 yaw 旋转的坐标轴：x 向前、y 向左、z 向上；没有补偿 roll/pitch。底盘世界位置及绝对 yaw 的 sin/cos 仍保留。没有直接把末端绝对位置拼入输入，也没有历史帧。

输入版本必须匹配 `obs23_target_and_ee_error_base_yaw_frame_v1`。原始 NPZ 和划分清单的 `input_dim` 仍为 **23**，不能改成 26；26 是训练加载时的派生维度。

## 底盘网络派生输入 8 维

| 内容 | 维度 |
|---|---:|
| 停车点相对底盘的水平坐标 | 2 |
| 朝向停车点的有符号角误差 | 1 |
| 最终朝向的有符号角误差 | 1 |
| align / drive / orient / done 的 one-hot | 4 |

停车点由目标位置和随目标高度插值的 standoff 距离计算。特征版本为 `parking_body_xy_signed_heading_errors_phase4_v1`。停车点不是机械臂最终目标点。

## 专家动作 11 维

| 列 | 含义 | 单位 |
|---|---|---|
| `0:4` | 右前、左前、左后、右后轮目标角速度 | rad/s |
| `4:11` | J1～J7 控制力矩 | N·m |

训练使用每个模型对应的标准化统计；推理必须同时加载该模型的 `normalization.npz`。二值阶段和底盘 one-hot 不做普通均值方差标准化。执行时的轮速/力矩限幅为 ±3 rad/s / ±30 N·m。

## 目标覆盖与清单

原环状轨迹：水平半径 0～2 m，整圈方位，高度 `[0.2,0.35,0.5,0.75,1.0,1.15,1.3]`，931 条。当前任务选取半径 ≥0.8 m 的 798 条。初始底盘位置和朝向固定。

补采：半径 0.8～2 m 分 5 段、方位分 10 段，共 50 个随机 XY；每个 XY 在 `[0.2,0.275]`、`[0.275,0.35]`、`[0.35,0.425]`、`[0.425,0.5]` 内各随机取一个高度，共 200 条。XY 组按 35/7/8 分配为 140/28/32 条。

| 固定清单位置（运行目录内） | train / val / test | 用途 |
|---|---|---|
| `RoboEnv/bc_training_26d_noise_ring/data_split.json` | 651 / 140 / 140 | 原机械臂预训练 |
| `RoboEnv/bc_training_nav_ring_r08/data_split.json` | 558 / 116 / 124 | 底盘训练、第一次机械臂微调 |
| `RoboEnv/bc_training_arm_ring_r08_lowz_supplement/data_split.json` | 698 / 144 / 156 | 当前补采后机械臂微调 |

合并清单 `data_dir` 指向 RoboEnv 根目录，`splits` 中每个路径包含数据来源目录；前两种清单的文件项通常只有文件名，`data_dir` 则指向对应原始目录。统一使用：

```python
from pathlib import Path
import json
import numpy as np
m = json.loads(Path('RoboEnv/bc_training_arm_ring_r08_lowz_supplement/data_split.json').read_text(encoding='utf-8'))
path = Path(m['data_dir']) / m['splits']['train'][0]
with np.load(path, allow_pickle=False) as d:
    observations, actions, ee = d['observations'], d['actions'], d['ee_positions']
```

合并训练集共 7,306,407 帧，验证集 1,481,313 帧，测试集 1,613,639 帧。归属按目标固定，验证/测试目标不进入梯度训练；验证集用于选模型和早停，因而已经参与开发选择。当前独立测试集 156 点尚无本版完整闭环结果。
