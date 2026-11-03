# 从采集到演示的完整流程

所有相对路径以本仓库根目录为起点，`python` 指已激活的 ML 环境。先按 README 完成 `prepare_workspace.py` 和 `verify_release.py --runtime`。

本仓库有两条路线：

1. **现有模型复现**：不需要完整 NPZ，直接运行单点、批量测试和演示。
2. **训练复现/新采集**：需要原始数据。恢复作者数据时使用冻结清单；重新采集时文件名和测量噪声 realization 会变化，需要为新文件重建清单，不能直接套旧文件名。

以下将脚本路径存为 PowerShell 变量，避免反复输入：

```powershell
$task = "LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task"
```

## 1. 专家采集

### 采集策略

- 仿真步长 1 ms，初始位置和朝向固定，整圈多半径、多高度目标。
- 使用 `$task/configs/robotnik_arm.json`；测量噪声开关为 1，各配置参数值为 0.001。没有通过批量脚本关闭它。
- task3 支持额外力矩扰动，但本数据集**明确使用 `--noise-std 0`**；不是同时叠加两种噪声。
- 先导航至停车位，再由专家机械臂控制收敛；在 1 cm 内连续保持 1 s，并至少运行到 8 s 才正常结束。
- 保存控制前观测与相配专家动作，以及对齐的末端位置。额外保持段用来提供到达附近的维持动作样本。
- 采集规划 seed 不等于完全固定环境测量噪声；重新采集不保证逐帧一致。

单目标试运行（会保存新轨迹，GUI 绘图）：

```powershell
python "$task/robotnik_arm_task_3.py" --target 0.0 1.1 0.75 --noise-std 0 --hold-seconds 1 --output-dir RoboEnv/expert_manual_check
```

原环状设计为 931 点，包含后续不使用的近距离位置。保留原采集脚本，是为了记录模型实际的预训练数据历史：

```powershell
# 只打印计划，不运行
python "$task/collect_expert_ring.py" --dry-run

# 新实验用独立数据目录，避免和旧文件混在一起
python "$task/collect_expert_ring.py" --data-dir RoboEnv/recollected_ring
```

脚本打印新建的 `expert_collection_logs/ring_时间戳` 路径；后续 `--collection-config` 使用其中的 `collection_config.json`。已有同目标记录会跳过，失败记录也不会自动重采。

## 2. 数据划分

### 恢复原始数据时

直接使用准备工具恢复的三份固定清单，不重新随机划分。原始数据 ZIP 应解压到根目录，保留文件名和目录。可运行：

```powershell
python tools/verify_release.py --runtime --data
```

这检查当前 998 文件的存在、目标、形状和有限值；从最初的 931 预训练开始，还需要导出时的 `--include-pretraining-near`。

### 重新采集后

新文件名会变化，使用发布辅助工具重建清单。它复用原目标归属算法，只增加输入输出路径参数，不覆盖任何冻结模型目录。

```powershell
# 将下面采集日志路径替换成刚才程序打印的真实路径
$collectionConfig = "RoboEnv/expert_collection_logs/ring_实际时间戳/collection_config.json"
python tools/rebuild_splits.py full --data-dir RoboEnv/recollected_ring --collection-config $collectionConfig --output-dir RoboEnv/fresh_splits/full
python tools/rebuild_splits.py r08 --full-manifest RoboEnv/fresh_splits/full/data_split.json --output-dir RoboEnv/fresh_splits/r08
```

full 为 651/140/140，r08 为 558/116/124。基于整个目标分组；不把相邻时间帧随机分进不同集合。

若仅核对恢复的数据，下面的命令会在新目录重建一份 full/r08，而不覆盖原清单：

```powershell
python tools/rebuild_splits.py full --output-dir RoboEnv/rebuilt_splits/full
python tools/rebuild_splits.py r08 --full-manifest RoboEnv/rebuilt_splits/full/data_split.json --output-dir RoboEnv/rebuilt_splits/r08
```

## 3. 原 26 维模型预训练

这是当前机械臂权重的起点，最初输出 4 轮速 + 7 力矩。26 维特征由 23 维原观测与末端误差派生。网络 `26→256→256→128→11`、ReLU、Adam，初始学习率 1e-4，batch 4096，最多 300 轮，耐心 15。

训练损失包括归一化后的轮速误差、力矩误差，以及额外的停止阶段轮速误差；停止附近、低目标且接近目标的训练样本被提高采样概率。具体归约方式和超参数以脚本及运行配置为准。

```powershell
# 恢复原完整 931 数据后
python tools/train_bodyframe.py

# 或使用刚重采并重建的清单
python tools/train_bodyframe.py --split-path RoboEnv/fresh_splits/full/data_split.json --output-root RoboEnv/fresh_bodyframe --epochs 300
```

结果会打印新 `run_时间戳`。本仓库已有作者的最佳第 133 轮快照，所以仅继续当前分离式方法时可跳过这一步。

## 4. 单独训练底盘

从原轨迹取专家动作 `actions[:,0:4]` 作为标签，派生 8 维导航特征。新 MLP `8→64→64→4` 从随机初始化训练；机械臂不参与优化。

- 每 10 帧保留一帧，额外保留阶段切换前后帧。
- 四个阶段按逆频率平衡采样，每轮最多 250,000 帧。
- 损失：总体归一化轮速 MSE + 2 × done 阶段轮速 MSE。
- Adam 1e-3，batch 8192，最多 120 轮；验证 loss 选择最佳模型及早停，另有 ReduceLROnPlateau。

```powershell
# 使用发布的预训练手臂和 798 清单，从头训练新的底盘
python "$task/train_nav_bc_ring_r08.py"

# 使用新预训练结果时：把该路径替换成第3步打印的真实结果目录
$bodyRun = "RoboEnv/fresh_bodyframe/run_实际时间戳"
python "$task/train_nav_bc_ring_r08.py" --split-path RoboEnv/fresh_splits/r08/data_split.json --arm-run-dir $bodyRun
```

底盘运行配置记录它绑定的原机械臂、清单、代码指纹。不要手动拿其他 `best_model.pt` 覆盖进去。

## 5. 第一次仅机械臂微调

使用底盘配置绑定的原机械臂，加载原权重和归一化；底盘文件完全不更新。网络依旧有 11 个输出，但只用 `prediction[:,4:11]` 计算力矩 MSE，隐藏层也参与更新。

没有只截取“导航结束后”样本：完整轨迹包括导航时的手臂姿态维持、接近和到达保持。低高度 `z≤0.35` 与其他高度在采样期望上各占一半，每轮最多 500,000 帧。

Adam 1e-5，batch 4096，最多 30 轮，耐心 8。按 `0.5×低高度验证力矩MSE + 0.5×其他高度验证力矩MSE` 选择模型。验证不重采样；起点 epoch 0 作为回退快照。

```powershell
# 默认绑定发布的底盘运行
python "$task/finetune_arm_bc_ring_r08.py"

# 若第4步重新训练了底盘，使用其打印的真实目录
$navRun = "RoboEnv/bc_training_nav_ring_r08/run_新的实际时间戳"
python "$task/finetune_arm_bc_ring_r08.py" --nav-run-dir $navRun
```

作者对应结果为 `previous` 第 11 轮。到此可以用 `test_bc_split_nav_arm_finetuned.py` 或配套 batch 文件独立测试。

## 6. 补采连续低高度

如果恢复原数据，200 条补采已经存在，无需再次采集。若要新采：固定 50 个新 XY，每个 XY 取四个连续低高度，共 200 条；保存独立计划和数据目录。

```powershell
# 使用发布的底盘清单，补采缺少的目标
python "$task/collect_expert_ring_lowz_supplement.py" --dry-run
python "$task/collect_expert_ring_lowz_supplement.py"

# 全新流程使用自己的底盘清单和独立补采目录
python tools/collect_supplement.py --base-split "$navRun/data_split.json" --data-dir RoboEnv/recollected_lowz --dry-run
python tools/collect_supplement.py --base-split "$navRun/data_split.json" --data-dir RoboEnv/recollected_lowz
```

重采不会保证所有点都完成连续保持；合并工具会检查补采是否齐全且达到保持条件，缺失/失败时应先查看采集日志。不要悄悄移除失败点后沿用原分母。

## 7. 合并补采清单

恢复原数据时使用固定的合并清单即可。若要重新核对或者重建：

```powershell
# 原文件已恢复时；在新目录核对合并结果
python tools/rebuild_splits.py merged --output-dir RoboEnv/rebuilt_splits/merged

# 新采集流程：必须使用采集计划绑定的同一份 base-split
python tools/rebuild_splits.py merged --base-split "$navRun/data_split.json" --supplement-dir RoboEnv/recollected_lowz --output-dir RoboEnv/fresh_splits/merged
```

998 条分为 698/144/156。旧目标及顺序保留，新补采按 XY 组分配；模型分离不意味着数据集训练/验证归属可以独立重洗。

## 8. 当前补采版仅机械臂微调

加载第5步的机械臂继续训练，回放旧数据并加入新数据，底盘仍固定。学习率 1e-5，7 维力矩 MSE、50/50 高度组采样和验证指标都沿用第一次微调。注意分组阈值仍为 0.35 m，不是把所有新增 0.2～0.5 m 都算进低组。

```powershell
# 使用发布的模型和原始补采清单，新建一次微调运行
python "$task/finetune_arm_bc_ring_lowz_supplement.py"

# 使用自己新生成的模型/清单；替换 parent 路径
$parentArmRun = "RoboEnv/bc_training_arm_ring_r08_finetune/run_新的实际时间戳"
python "$task/finetune_arm_bc_ring_lowz_supplement.py" --nav-run-dir $navRun --parent-arm-run-dir $parentArmRun --split-path RoboEnv/fresh_splits/merged/data_split.json
```

作者结果最佳第 30 轮，达到最大轮数，没有触发早停。不能把只修改 `--epochs` 误认为从上次第30轮恢复：本脚本默认仍从 parent 开始一次新微调。

## 9. 独立测试与演示

训练完成后**另行运行**测试文件。没有把训练和仿真测试写进同一个入口：

```powershell
$armRun = "RoboEnv/bc_training_arm_ring_r08_lowz_supplement/run_自己的实际时间戳"
python "$task/batch_test_bc_split_nav_arm_lowz_supplement.py" --arm-run-dir $armRun --split val
python "$task/test_bc_split_nav_arm_lowz_supplement.py" --arm-run-dir $armRun --target 0.8 0.8 0.75 --duration 20 --sensor-noise off
python "$task/demo_robotnik_bc.py" --arm-run-dir $armRun --height-mode trained
```

批量默认 val；独立 test 用 `--split test`。只测试旧验证点用 `--subset base`，新增连续低层点用 `--subset supplement`。不要跨集合直接比较百分比，也不要把动作 MSE 当作成功率。

GUI 运行时两模型每步输出拼接后执行：状态更新 → 8D/26D 特征 → 两网络前向 → 反标准化 → ±3 / ±30 限幅 → 仿真一步。状态机 done 后是网络学习停止，并没有覆盖轮速为零。

## 常见问题

- **找不到文件**：是否执行准备工具？是否取得完整数据？`@WORKSPACE@` 出现在运行目录说明尚未迁移；`artifacts/` 内出现它是正常的。
- **哈希不一致**：不要关掉校验。核对模型目录、父模型、清单和代码是否匹配；迁移应使用工具重新生成运行副本。
- **只想演示**：无需重新训练或下载 2.75 GB 数据；用仓库已有权重。
- **连续高度失败但离散层成功**：这是目前已知泛化问题，不能通过放宽成功阈值或只挑成功点解释为解决。
- **重新训练效果不完全一致**：随机优化、平台差异和测量噪声可能影响结果；发布模型和固定目标/种子用于明确本次实验证据。
