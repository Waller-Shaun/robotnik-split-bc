# bc_training_arm_ring_r08_recovery / run_20261104_130656_584873

分类：历史正式训练运行／模型快照。

训练结束元数据见 [training_finished.json](training_finished.json)，以该快照为准。

| 检查点 | 用途 | SHA256 |
|---|---|---|
| [baseline_model.pt](baseline_model.pt) | 冻结父模型对照，不是新模型 | `9c1bb6fe5de8ca87ec401c223d799ae7c64124f7eb8c8926beeb3b52e3550e81` |
| [best_model.pt](best_model.pt) | 按该运行实际选模规则保存；不自动代表超过父模型 | `1792dfb3af79cdf592c6389415df2548c7392591981ff445dcc77e50ae9b65d6` |
| [last_model.pt](last_model.pt) | 最后训练状态，部分包含优化器，供续训；不等于最佳 | `222858fb94129f3cb9e46f7c55e102b362c2fc5727ddee4e2f3f8273144897ee` |

本目录保持源文件字节不变。配置和检查点内部可能保留历史绝对路径；这是归档，不代表解压后能在任意目录直接运行。配套策略类、父模型、归一化与导航均需匹配，不能只加载某个同名best_model.pt。原始大体积rollout NPZ未随本分支全部提交，summary/CSV及子目录小型诊断记录已保留。
