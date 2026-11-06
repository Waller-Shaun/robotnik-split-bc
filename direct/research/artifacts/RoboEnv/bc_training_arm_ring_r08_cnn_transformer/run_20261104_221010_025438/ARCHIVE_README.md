# bc_training_arm_ring_r08_cnn_transformer / run_20261104_221010_025438

分类：历史正式训练运行／模型快照。

训练结束元数据见 [training_finished.json](training_finished.json)，以该快照为准。

| 检查点 | 用途 | SHA256 |
|---|---|---|
| [best_model.pt](best_model.pt) | 按该运行实际选模规则保存；不自动代表超过父模型 | `fdf2187d97ee0dd7426da3501a24eb324e2c06a0f15fb1213b689c1a8e19ebc6` |
| [initial_model.pt](initial_model.pt) | 训练前初始策略／零残差，不能作为新训练成果 | `5435dd5b19e8bca3831f9e92d557169dcb91d2f8ad9e3d4481e0f2545ffbdda6` |
| [last_model.pt](last_model.pt) | 最后训练状态，部分包含优化器，供续训；不等于最佳 | `e17d35c7e386e064bcfafc1e102dde3870984d2e2579c1c16247dd2ebfb3e121` |

本目录保持源文件字节不变。配置和检查点内部可能保留历史绝对路径；这是归档，不代表解压后能在任意目录直接运行。配套策略类、父模型、归一化与导航均需匹配，不能只加载某个同名best_model.pt。原始大体积rollout NPZ未随本分支全部提交，summary/CSV及子目录小型诊断记录已保留。
