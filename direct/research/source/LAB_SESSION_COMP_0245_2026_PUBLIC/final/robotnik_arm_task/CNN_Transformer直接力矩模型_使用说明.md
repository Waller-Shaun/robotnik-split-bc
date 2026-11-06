# CNN＋Transformer直接力矩模型

这是独立的新架构试验：机械臂模型从零开始监督训练，直接输出7个关节力矩，原导航模型继续输出4个轮速。它不叠加旧机械臂MLP，也没有新增PD、IK、专家接管、RL奖励或强制底盘停车。两个机器人配置均保留。

## 网络结构

```text
历史输入 (batch, 16, 26)，从150ms前到当前，间隔10ms
    ↓ 转置为 (batch,26,16)，沿时间方向卷积
因果Conv1d：26→64，kernel3，stride1，只左填充；GELU
因果Conv1d：64→128，kernel3，stride1，只左填充；GELU
    ↓ 转回 (batch,16,128)，LayerNorm＋可学习位置编码
Transformer Encoder：2层，4头，特征128，前馈256，dropout0.1
    ↓ 取最后一个时间点的128维表示
Linear128→64，GELU，Linear64→7
    ↓ 沿用原7维力矩输出归一化的反变换
当前7个机械臂力矩
```

共305,991个可训练参数。Transformer采用因果注意力，卷积只左侧填充；没有任何当前时间之后的输入。每个仿真物理步更新历史并重新预测力矩；10ms是历史采样间隔，不是把力矩保持10ms。物理仍为1kHz。

额外提供`--architecture cnn`对照：去掉注意力和位置编码，使用卷积的当前特征与全窗口平均特征拼接后回归7维力矩，共46,919参数。两者参数量不同，不能把结果写成严格等容量对照。

## 数据和划分

- 原698个训练目标、约7,306,407帧原始专家数据。
- 已有102条专家恢复片段、233,150帧；来源与120版恢复微调完全相同，不新增采集。
- 原144个val和156个test目标保持不变。val只用于选模型，test不进入训练。
- 输入是相同的底盘yaw坐标26维，标准化统计继承恢复120版，不重新用val拟合。
- 标签是当前帧的7维专家力矩；不读取专家内部参考角作为输入或新标签。

“16帧”组成一个窗口样本，不是16个独立新示范。所有窗口保留原轨迹边界。原轨迹起始历史不足时，重复该轨迹首帧，与在线推理缓冲一致。

恢复导出的片段没有专家接管前的完整历史，所以每条恢复片段的前150帧只作为窗口上下文，不作为训练端点；不会拼接另一条轨迹或伪造接管前历史。默认可用恢复端点为217,850个，原文件与全部233,150帧保留。

默认每轮有放回抽样500,000个端点：80%来自原数据（低高度z≤0.35与其他高度各半），20%来自已有恢复片段的有效端点。这个比例沿用恢复微调思路。

## 训练、速度和早停

```powershell
pixi run -e ml python "F:\UCL\COMP0245 Labs\LAB_SESSION_COMP_0245_2026_PUBLIC\final\robotnik_arm_task\train_arm_bc_cnn_transformer.py" --device cuda
```

默认：最多200个epoch，至少20轮，验证连续15轮未改善后早停；AdamW，学习率1e-4，weight decay1e-4，batch512，梯度范数限1。验证MSE平台期会触发ReduceLROnPlateau，学习率减半（patience5），最低1e-6。

这是从零训练，不是从120版或失败的RL版继续微调。第0轮显示的是随机网络的验证误差，不是基线表现；best从训练后的第1轮起选择。

GPU只存单帧数组，每批按索引生成16帧窗口，不提前复制成几千万个窗口。支持时，CUDA训练采用bf16 autocast，验证和正式推理保持float32。使用`--no-amp`可关闭混合精度。

**默认验证固定每10ms取一个端点，并保留每条轨迹首尾，所有144个验证目标仍保留。** 早停依据这些固定端点上的`0.5×低高度MSE＋0.5×其他高度MSE`。这是节约训练时间的抽样验证指标，不是旧实验的全帧验证MSE，二者不能直接比较数值高低。

需要完整逐帧验证时，从训练开始使用：

```powershell
pixi run -e ml python "F:\UCL\COMP0245 Labs\LAB_SESSION_COMP_0245_2026_PUBLIC\final\robotnik_arm_task\train_arm_bc_cnn_transformer.py" --device cuda --validation-stride 1
```

CNN对照训练使用相同数据、采样和默认设置，但输出到独立目录：

```powershell
pixi run -e ml python "F:\UCL\COMP0245 Labs\LAB_SESSION_COMP_0245_2026_PUBLIC\final\robotnik_arm_task\train_arm_bc_cnn_transformer.py" --device cuda --architecture cnn
```

默认结果目录：

```text
RoboEnv/bc_training_arm_ring_r08_cnn_transformer/run_时间戳/
  config.json                  模型/数据/验证设置和代码哈希
  data_split.json               继承清单
  normalization.npz             继承统计量
  recovery_manifest.json         原恢复示范来源清单
  initial_validation.json        随机初始网络的MSE
  initial_model.pt               随机初始网络
  history.csv                    每轮train/val/低高度/其他高度MSE、lr、秒数
  best_model.pt                  平衡验证MSE最低的已训练模型
  last_model.pt                  最后一轮模型、AdamW、scheduler和随机状态
  training_finished.json         最佳轮、最终轮和程序检查标记
```

CNN对照根目录为`bc_training_arm_ring_r08_cnn_history`。当前入口尚未实现恢复经验/优化状态的续训命令；不能用旧MLP/GRU/RL的续训程序加载本检查点。

## 独立测试

完整验证（默认配置噪声、144个目标、原随机种子）：

```powershell
pixi run -e ml python "F:\UCL\COMP0245 Labs\LAB_SESSION_COMP_0245_2026_PUBLIC\final\robotnik_arm_task\batch_test_bc_split_nav_arm_cnn_transformer_parallel.py" --split val --workers auto
```

单点观察：

```powershell
pixi run -e ml python "F:\UCL\COMP0245 Labs\LAB_SESSION_COMP_0245_2026_PUBLIC\final\robotnik_arm_task\test_bc_split_nav_arm_cnn_transformer.py" --target 1 -1.732051 1 --seed 1956328971 --duration 20
```

默认选择最新完成的非smoke CNN＋Transformer结果。测试CNN对照或指定某次训练时，附加`--arm-run-dir "完整run目录"`。

所有测试保持20秒内任意时刻真实PyBullet末端误差<1cm的标准，真实位置不作模型输入，无8秒门槛和保持要求。输入使用带噪观测构造的同一26维特征；支持`--sensor-noise off`做单独无噪声评估，不更改配置文件。原导航权重和底盘阶段输入规则沿用原实现，规则产生的专家轮速被丢弃。

由于每步需要处理历史，推理计算量会增加。本机单CPU线程的小规模测量约0.9ms/次（不含物理、FK、导航和其他开销），只能作为速度参考；仿真可能慢于实时运行，不能沿用MLP的整批耗时预期。

## 验收与结论边界

已做语法、输出维度、未来帧不可见、历史顺序、窗口与在线缓冲一致、轨迹边界、恢复前缀排除、CUDA bf16梯度与CNN对照前向检查。

另做过4条原train、4条原val、2条恢复片段的两轮小样本检查，以及20ms单目标独立并行测试接口检查。它们仅验证程序，不代表模型成功率；smoke目录不会被默认正式模型选择器选中。

本方案是时间卷积＋Transformer的监督力矩回归，不是完整ACT实现或新原创算法。它同时改变了架构、初始化和优化设置，未必优于已有120版。此前GRU效果有限，不能把“增加历史/注意力”直接当成提升保证；应以同目标、同噪声、同种子的闭环成功率及失败变化判断。

结构参考：[ConvTran](https://arxiv.org/abs/2305.16642)，[PyTorch TransformerEncoder](https://docs.pytorch.org/docs/2.11/generated/torch.nn.TransformerEncoder.html)。ConvTran原研究为时间序列分类，这里借用卷积特征编码思路并改为直接力矩回归，不声称复现其研究结果。
