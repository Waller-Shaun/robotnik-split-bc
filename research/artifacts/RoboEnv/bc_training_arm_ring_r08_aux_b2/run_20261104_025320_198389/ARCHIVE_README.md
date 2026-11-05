# bc_training_arm_ring_r08_aux_b2 / run_20261104_025320_198389

分类：历史正式训练运行／模型快照。

训练结束元数据见 [training_finished.json](training_finished.json)，以该快照为准。

| 检查点 | 用途 | SHA256 |
|---|---|---|
| [best_model.pt](best_model.pt) | 按该运行实际选模规则保存；不自动代表超过父模型 | `419e31c2382fcbc4f3e719aa71258e7da39929b44cd984c0be6be6bf4cd31440` |
| [last_model.pt](last_model.pt) | 最后训练状态，部分包含优化器，供续训；不等于最佳 | `dd49740b6060b8d1a434276b1e78516fc98af5895fe39770c3c3955fcef9ce32` |

本目录保持源文件字节不变。配置和检查点内部可能保留历史绝对路径；这是归档，不代表解压后能在任意目录直接运行。配套策略类、父模型、归一化与导航均需匹配，不能只加载某个同名best_model.pt。原始大体积rollout NPZ未随本分支全部提交，summary/CSV及子目录小型诊断记录已保留。
