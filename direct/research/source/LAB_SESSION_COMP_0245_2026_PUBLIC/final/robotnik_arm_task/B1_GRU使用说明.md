# B1：机械臂因果历史 GRU 实验

本实验只改变机械臂输入的历史上下文与网络结构。原始NPZ、998划分、底盘网络、状态机、动作限幅、测试成功标准均不改。

## 文件

| 文件 | 用途 |
|---|---|
| robotnik_arm_gru_policy.py | 共用网络、因果历史窗口、模型绑定检查；无训练入口 |
| train_arm_bc_ring_gru.py | B1训练入口；不会自动测试 |
| test_bc_split_nav_arm_gru.py | 单点GUI或无窗口测试；不会训练 |
| batch_test_bc_split_nav_arm_gru.py | 同一144个验证/156个测试目标的独立批量评估 |

## 网络和历史

```text
当前26维 → 原MLP 256/256/128 → 7维力矩 ──────────────┐
                                                 + → 7维最终力矩
过去16个26维观测 → Linear64/ReLU → GRU64 → Linear7 ──┘
```

GRU分支输出的是归一化力矩修正，最后再按原统计量反标准化。Linear7初始权重和偏置为0，所以epoch0与父MLP的输出相同。原MLP及GRU分支都参加训练；这里只使用力矩MSE，没有B2期望关节角辅助损失。

16个观测按每10个仿真步取一次，物理步长仍为1ms，因此覆盖过去150ms，包含当前观测。每个窗口按旧→新排列，GRU在窗口开始时用零隐状态；不跨窗口累积隐状态。推理也完全使用同一种有限窗口。

回合开头不足150ms时重复本回合第一帧，不借用其他回合数据，也不读未来。窗口可以跨越导航完成时刻；只有新回合才清空历史。没有把q_des、qd_des、未来动作放入模型输入。

## 控制变量

- 数据：原798 + 补采200，train/val/test = 698/144/156；不重新划分。
- 起点：与B0补采微调相同的第11轮MLP，**不是从已经训练完的第30轮再继续**。这样两者的起点、数据和力矩损失一致。
- 归一化：原父模型统计量不变；二值phase维不标准化。
- 训练：Adam 1e-5，batch4096，每轮抽样50万窗口末端，z≤0.35与其他高度期望各50%，默认最多30轮、耐心8。
- 每一个标签仍是窗口末端同一时刻的7个专家力矩；只把历史作为额外上下文。
- 验证：全部原验证帧，不做窗口末端下采样；按低/其他高度MSE各50%选择最佳模型。
- 新模型参数更多，因此B1结果同时涉及结构/容量变化。可另外用`--history-length 1`训练相同参数量的“只看当前”分支，对照收益究竟是否来自历史。
- CUDA可用时模型和整套单帧数组放到GPU，窗口按批次临时索引，不把百万帧展开成16倍。此实现改变存储和搬运方式，不改变监督标签和采样分布。没有强制混合精度。

## 开始训练

在现有RoboEnv环境中运行：

```powershell
pixi run -e ml python "F:\UCL\COMP0245 Labs\LAB_SESSION_COMP_0245_2026_PUBLIC\final\robotnik_arm_task\train_arm_bc_ring_gru.py" --device cuda
```

结果新建在`RoboEnv/bc_training_arm_ring_r08_gru_b1/run_时间戳/`，不覆盖B0或已有模型。保存最佳模型、原归一化、清单、配置、代码快照和history.csv。epoch0也保存；若后续没有更好结果，可回退到相同起点。

若显存不足，可加`--batch-size 2048`；应记录此改动。改变最大轮数可加`--epochs 60`，但先与B0的30轮设置对照。

## 训练完成后，单独测试

```powershell
# 默认选择该B1目录下最新完成的正式运行，程序小样本检查不会被自动选中。
pixi run -e ml python "F:\UCL\COMP0245 Labs\LAB_SESSION_COMP_0245_2026_PUBLIC\final\robotnik_arm_task\test_bc_split_nav_arm_gru.py" --target -1.5827733453143615 -0.04169457644129078 0.832204004842676

# 与B0相同的完整144点验证集、同样种子，最长20s，噪声关闭。
pixi run -e ml python "F:\UCL\COMP0245 Labs\LAB_SESSION_COMP_0245_2026_PUBLIC\final\robotnik_arm_task\batch_test_bc_split_nav_arm_gru.py" --split val
```

可用`--arm-run-dir "实际结果目录"`固定模型。单点仍可`--headless`，批量默认无窗口。测试时每1ms都重新计算动作，不是每10ms持有同一个力矩；历史间隔不等于策略执行频率。

先对照完整144点成功率，再分别看原116点/新增28点、高度分组和连续高度单点。离线验证MSE改善不能自动当作到达成功率改善。

本次不自动启动长时间正式训练、不修改机器人配置，也不自动推送尚未完成的新实验到GitHub。
