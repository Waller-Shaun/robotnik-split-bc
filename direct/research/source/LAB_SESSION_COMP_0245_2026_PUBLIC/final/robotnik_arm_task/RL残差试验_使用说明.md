# 冻结120版模型的TD3残差试验

本试验尝试让额外网络通过仿真交互学习精细力矩修正。它与之前的监督残差网络具有相似的部署形式，但训练目标不同：之前拟合专家力矩差值，本次优化仿真中的任务奖励。尚未进行正式训练，也没有成功率提升结论。

## 模型和控制逻辑

冻结机械臂父模型：`bc_training_arm_ring_r08_recovery/run_20261104_152933_829681`，其带噪验证结果为120/144。冻结导航模型仍为原来的8维输入MLP。

| 部分 | 输入 | 输出 | 是否更新参数 |
|---|---|---|---|
| 原机械臂MLP | 原26维带噪观测 | 基础关节力矩；原轮速输出继续弃用 | 否 |
| 新TD3 actor | 同一26维标准化观测 | 7维修正，`2 × tanh(...)` Nm | 是 |
| 两个critic | 26维观测＋7维修正动作 | 预期累计奖励Q值 | 是，仅训练使用 |
| 原导航MLP | 原8维导航观测 | 4个轮速 | 否 |

actor结构为`26 → 64 → 64 → 7 → tanh`，最后一层零初始化。初始修正严格为零。每个critic结构为`33 → 128 → 128 → 1`。

最终机械臂动作是：`原模型力矩 + actor修正`，最后沿用原来的±30 Nm限幅。导航动作沿用原模型及±3 rad/s限幅，不强制停车。

导航阶段继续使用原模型。阶段标志变为0后，actor开始提供修正。物理仿真和基础反馈仍为1 kHz，修正每10 ms更新一次（100 Hz），期间保持上次修正。这一动作保持逻辑在训练和测试中一致。

## 训练数据和奖励

不需要重新采集专家数据。程序读取120版的划分清单，只在原698个**训练目标**上产生在线交互；验证和测试目标不参与训练。默认50%的回合从`z < 0.5 m`的300个训练目标抽样，另50%从其他训练目标抽样，兼顾低高度修正和原有表现。

底盘和机械臂通过冻结模型实际运动到阶段切换位置，不瞬移、不施加专家动作。12秒内未完成导航的回合记录为跳过，不能算作RL成功。训练记录也单独注明：导航前缀是否已经进入1 cm，避免把原模型成功归因于RL。

训练奖励包括末端距离缩小、持续接近目标、在1 cm内保持，以及力矩修正和关节速度惩罚。持续在1 cm内0.2秒获得成功奖励。奖励读取PyBullet真实末端位置，但该位置**不输入actor或critic**；策略仍读取带噪观测构造的26维特征。

训练默认使用`RoboEnv/configs/robotnik_arm.json`当前的传感器噪声。另加的小幅动作探索噪声只在训练中使用，测试时关闭；两个原配置文件均不改动。

## 文件

| 文件 | 用途 |
|---|---|
| `train_arm_rl_residual.py` | 独立训练入口 |
| `robotnik_arm_rl_policy.py` | actor、双critic、经验回放、TD3更新与部署包装 |
| `robotnik_arm_rl_env.py` | 冻结基线的物理交互和奖励 |
| `test_arm_rl_residual.py` | 单个目标测试，可显示仿真 |
| `batch_test_arm_rl_residual_parallel.py` | 独立并行验证/测试，默认最多4个进程 |

沿用工程已有PyTorch，不额外安装Gym或Stable-Baselines3。这是针对当前任务的TD3原型，机制参考[TD3官方实现说明](https://spinningup.openai.com/en/latest/algorithms/td3.html)，不声称复现论文成绩。

## 启动训练

在现有Pixi项目路径执行，不需要切换目录：

```powershell
pixi run -e ml python "F:\UCL\COMP0245 Labs\LAB_SESSION_COMP_0245_2026_PUBLIC\final\robotnik_arm_task\train_arm_rl_residual.py" --device cuda --steps 20000
```

先运行这次20,000个RL时间步的试验。一个RL时间步通常对应10个物理步，即10 ms。这不是20,000个epoch，也不是20,000条完整轨迹。每回合还要实际运行导航前缀，因此总耗时不能只用这200秒的机械臂交互时间估算。

GPU用于批量actor/critic梯度更新，CPU单线程用于物理仿真及单帧冻结模型推理。PyBullet仍可能是耗时主体，GPU利用率低不等于没有CUDA训练。

本版采用固定交互预算，没有BC式15轮验证MSE早停。训练不计算离线验证MSE，也不根据训练回报自动选“最优”模型。

结果保存在：

```text
RoboEnv/bc_training_arm_ring_r08_rl_td3/run_时间戳/
  config.json             实际设置、父模型、配置及代码哈希
  data_split.json         继承原清单
  normalization.npz       继承原统计量
  episodes.csv            每回合目标、导航、误差、奖励和梯度损失
  simulation.log          仿真初始化输出
  initial_model.pt        零修正策略
  last_model.pt           最后训练状态
  best_model.pt           固定预算最后actor的兼容名称，并非验证集最优
  training_finished.json  仅完成预算且发生梯度更新后生成
```

训练目标抽样与噪声种子均记录。父模型和导航权重核验保持冻结。若中断，保存`interrupted_model.pt`；当前入口尚不支持完整恢复经验回放与随机流的续训，不能用BC的resume方式加载。

## 训练后单独测试

验证默认读取最新完成的非smoke正式RL结果，保持144个验证目标、原种子、20秒上限和带噪真实位置评分：

```powershell
pixi run -e ml python "F:\UCL\COMP0245 Labs\LAB_SESSION_COMP_0245_2026_PUBLIC\final\robotnik_arm_task\batch_test_arm_rl_residual_parallel.py" --split val --workers auto
```

单点观察：

```powershell
pixi run -e ml python "F:\UCL\COMP0245 Labs\LAB_SESSION_COMP_0245_2026_PUBLIC\final\robotnik_arm_task\test_arm_rl_residual.py" --target -1 1.732051 0.35 --duration 20
```

需要指定某次训练时，附加`--arm-run-dir "完整run目录"`。这两个测试程序不会学习；没有critic评估、探索噪声或专家接管。

正式成功标准沿用：20秒内任意时刻**真实末端距离小于1 cm**，无需8秒门槛，也不要求保持。训练的0.2秒保持奖励是优化目标，不冒充正式测试标准。

## 已完成的程序检查

- TD3终止/时间截断的Bellman目标、延迟actor更新和CUDA梯度更新通过检查。
- actor零输出、导航阶段禁止修正、100 Hz动作保持通过检查。
- 零修正与120版在相同目标/种子下的完整轨迹逐项完全一致，覆盖8.069秒导航完成及随后机械臂控制，直到11.277秒结束。
- 训练环境的导航前缀和切换后20个物理步，与原测试环境的观测、动作和真实位置逐项完全一致。
- 做过12个RL时间步的smoke，确认仿真、经验回放、8次GPU更新和检查点保存接通；smoke不参与自动正式模型选择。
- 独立并行测试完成一次20 ms的协议检查，进程错误为0；该极短测试不用于评价任务成功率。

上述检查只说明程序接通，不代表RL有效。正式结果应与120/144基线按相同目标、种子和噪声设置比较，关注新修复和新退化的目标，并保留训练失败的可能。
