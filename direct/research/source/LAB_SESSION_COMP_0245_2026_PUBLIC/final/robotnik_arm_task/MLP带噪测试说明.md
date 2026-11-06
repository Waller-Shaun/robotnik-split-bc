# 无 GRU 基线：带噪观测、真实末端位置评分

本实验只测试，不训练、不修改模型或任何机器人配置文件。

## 固定模型和标准

| 项目 | 设置 |
|---|---|
| 机械臂 | 补采数据微调后的无 GRU MLP，`run_20261103_141232_983306`，此前无噪声验证为 111/144 |
| 底盘 | 由该机械臂检查点绑定的冻结导航 MLP |
| 验证集 | 相同 144 个目标、原有逐目标随机种子；不重新划分 |
| 时限 | 默认每个目标 20 秒仿真时间 |
| 成功 | 任意时刻真实末端距离严格小于 0.01 m；不要求最短时间或保持时间 |
| 配置 | `F:\UCL\COMP0245 Labs\RoboEnv\configs\robotnik_arm.json` |
| 噪声 | 默认 `--sensor-noise config`，沿用配置中的开关和幅度；若配置未开启则报错 |

模型和阶段切换使用原来的带噪观测，估计末端仍由带噪状态通过正运动学得到。
评分使用 PyBullet 中 `robot_j2s7s300_end_effector` 连杆坐标原点的世界位置，即 `getLinkState(...)[4]`，不是质心位置。真实位置不输入模型。

## 完整验证

```powershell
pixi run -e ml python "F:\UCL\COMP0245 Labs\LAB_SESSION_COMP_0245_2026_PUBLIC\final\robotnik_arm_task\batch_test_bc_split_nav_arm_mlp_noise_parallel.py" --split val --sensor-noise config --workers auto
```

自动模式在启动时测量空闲内存和 CPU 负载：最多 4 个独立仿真子进程；每个子进程的数值计算使用 1 个线程；预留 1.5 GiB 内存，按每进程 800 MiB 估算并发量。这是保守的资源规划，不保证是硬件最快配置。显式 `--workers 4` 可覆盖自动规划；资源紧张时仍建议自动模式。没有 GUI 和逐步实时等待。

结果另存于机械臂模型目录的 `batch_tests/parallel_mlp_noise_truth_val_时间戳`，包含 `summary.json`、`evaluation_config.json`、`results.csv`、每个目标的轨迹和日志。进程错误计入总目标分母。

仅核对模型、配置与并行规划，不启动仿真：

```powershell
pixi run -e ml python "F:\UCL\COMP0245 Labs\LAB_SESSION_COMP_0245_2026_PUBLIC\final\robotnik_arm_task\batch_test_bc_split_nav_arm_mlp_noise_parallel.py" --plan-only
```

## 单点观察

```powershell
pixi run -e ml python "F:\UCL\COMP0245 Labs\LAB_SESSION_COMP_0245_2026_PUBLIC\final\robotnik_arm_task\test_bc_split_nav_arm_mlp_noise.py" --target -0.866025 -0.5 0.2 --duration 20 --sensor-noise config --seed 48
```

单点默认打开 GUI 并绘制真实误差和观测误差；批量子进程自动使用 `--headless`。

## 轨迹字段

- `ee_positions`、`ee_errors`、`final_error`、`success`：真实末端位置对应的评分数据。
- `observed_ee_positions`、`observed_ee_errors`：模型使用的带噪状态所估计的末端位置及误差。
- `observations`：模型实际使用的 26 维带噪特征；`actions` 为实际施加动作。
- `evaluation_position_source=pybullet_link_world_frame`，`true_state_used_for_model_input=False`。
- CSV 额外记录观测最终/最小误差；汇总中的 `observed_reached_but_true_failed` 表示估计曾达标、但真实位置在本次时限内从未达标的目标数。

为保证公平，可用同一新脚本加 `--sensor-noise off` 跑无噪声对照，两组都用相同真实位置标准。不要把不同种子、不同噪声模式的单次结果当作严格的稳健性结论。
