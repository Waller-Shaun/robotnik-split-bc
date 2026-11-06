# 模型与运行索引

覆盖本机34个历史运行；既包含本轮全部尝试，也保留早期基线用于追溯。正式运行、程序冒烟与助手执行的短续训分开标注。模型状态不依文件名猜测。

| 实验族 | 运行 | 分类 | 检查点 | 最佳轮／步 |
|---|---|---|---|---|
| bc_training | [run_20261023_135447_849619](artifacts/RoboEnv/bc_training/run_20261023_135447_849619/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt | 见原history/config |
| bc_training_23d | [run_20261028_012502_944390](artifacts/RoboEnv/bc_training_23d/run_20261028_012502_944390/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt | 见原history/config |
| bc_training_23d | [run_20261028_204007_921743](artifacts/RoboEnv/bc_training_23d/run_20261028_204007_921743/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt | 见原history/config |
| bc_training_23d_noise | [run_20261029_024749_959833](artifacts/RoboEnv/bc_training_23d_noise/run_20261029_024749_959833/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt | 见原history/config |
| bc_training_23d_noise_finetune | [run_20261030_100242_765759](artifacts/RoboEnv/bc_training_23d_noise_finetune/run_20261030_100242_765759/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt | 见原history/config |
| bc_training_23d_noise_resample | [run_20261029_170007_131617](artifacts/RoboEnv/bc_training_23d_noise_resample/run_20261029_170007_131617/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt | 见原history/config |
| bc_training_23d_noise_wide | [run_20261030_112637_239667](artifacts/RoboEnv/bc_training_23d_noise_wide/run_20261030_112637_239667/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt | 见原history/config |
| bc_training_23d_noise_wide_extended | [run_20261031_130737_274460](artifacts/RoboEnv/bc_training_23d_noise_wide_extended/run_20261031_130737_274460/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt | 见原history/config |
| bc_training_23d_noise_wide_finetune | [run_20261031_025012_026265](artifacts/RoboEnv/bc_training_23d_noise_wide_finetune/run_20261031_025012_026265/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt | 见原history/config |
| bc_training_26d_noise_ring | [run_20261102_032833_505981](artifacts/RoboEnv/bc_training_26d_noise_ring/run_20261102_032833_505981/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt | 见原history/config |
| bc_training_26d_noise_ring_bodyframe | [run_20261102_184221_743102](artifacts/RoboEnv/bc_training_26d_noise_ring_bodyframe/run_20261102_184221_743102/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt | 见原history/config |
| bc_training_26d_noise_wide_extended | [run_20261101_150934_422124](artifacts/RoboEnv/bc_training_26d_noise_wide_extended/run_20261101_150934_422124/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt | 见原history/config |
| bc_training_arm_ring_r08_aux_b2 | [run_20261104_024810_354398](artifacts/RoboEnv/bc_training_arm_ring_r08_aux_b2/run_20261104_024810_354398/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt, last_model.pt | 2 |
| bc_training_arm_ring_r08_aux_b2 | [run_20261104_025320_198389](artifacts/RoboEnv/bc_training_arm_ring_r08_aux_b2/run_20261104_025320_198389/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt, last_model.pt | 2 |
| bc_training_arm_ring_r08_aux_b2_control | [run_20261104_025706_206181](artifacts/RoboEnv/bc_training_arm_ring_r08_aux_b2_control/run_20261104_025706_206181/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt, last_model.pt | 185 |
| bc_training_arm_ring_r08_aux_velocity_b4 | [run_20261104_041750_567925](artifacts/RoboEnv/bc_training_arm_ring_r08_aux_velocity_b4/run_20261104_041750_567925/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt, last_model.pt | 31 |
| bc_training_arm_ring_r08_cnn_transformer | [run_20261104_221010_025438](artifacts/RoboEnv/bc_training_arm_ring_r08_cnn_transformer/run_20261104_221010_025438/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt, initial_model.pt, last_model.pt | 86 |
| bc_training_arm_ring_r08_finetune | [run_20261103_111655_849512](artifacts/RoboEnv/bc_training_arm_ring_r08_finetune/run_20261103_111655_849512/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt | 11 |
| bc_training_arm_ring_r08_gradgate_b3 | [run_20261104_034810_928084](artifacts/RoboEnv/bc_training_arm_ring_r08_gradgate_b3/run_20261104_034810_928084/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt, last_model.pt | 31 |
| bc_training_arm_ring_r08_gru_b1 | [run_20261103_203745_353439](artifacts/RoboEnv/bc_training_arm_ring_r08_gru_b1/run_20261103_203745_353439/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt | 30 |
| bc_training_arm_ring_r08_gru_b1 | [run_20261103_214030_030129](artifacts/RoboEnv/bc_training_arm_ring_r08_gru_b1/run_20261103_214030_030129/ARCHIVE_README.md) | 助手执行的续训探索（最佳55轮）；保留并标注，非新的用户完整评估 | best_model.pt, last_model.pt | 55 |
| bc_training_arm_ring_r08_gru_b1 | [run_20261103_215456_176247](artifacts/RoboEnv/bc_training_arm_ring_r08_gru_b1/run_20261103_215456_176247/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt, last_model.pt | 96 |
| bc_training_arm_ring_r08_gru_b1 | [run_20261103_220030_514697](artifacts/RoboEnv/bc_training_arm_ring_r08_gru_b1/run_20261103_220030_514697/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt, last_model.pt | 269 |
| bc_training_arm_ring_r08_lowz_supplement | [run_20261103_141232_983306](artifacts/RoboEnv/bc_training_arm_ring_r08_lowz_supplement/run_20261103_141232_983306/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt | 30 |
| bc_training_arm_ring_r08_recovery | [run_20261104_120308_587271](artifacts/RoboEnv/bc_training_arm_ring_r08_recovery/run_20261104_120308_587271/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt, last_model.pt | 0 |
| bc_training_arm_ring_r08_recovery | [run_20261104_130656_584873](artifacts/RoboEnv/bc_training_arm_ring_r08_recovery/run_20261104_130656_584873/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | baseline_model.pt, best_model.pt, last_model.pt | 100 |
| bc_training_arm_ring_r08_recovery | [run_20261104_152933_829681](artifacts/RoboEnv/bc_training_arm_ring_r08_recovery/run_20261104_152933_829681/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | baseline_model.pt, best_model.pt, last_model.pt | 138 |
| bc_training_arm_ring_r08_residual | [run_20261104_171630_415142](artifacts/RoboEnv/bc_training_arm_ring_r08_residual/run_20261104_171630_415142/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | baseline_model.pt, best_model.pt, last_model.pt | 94 |
| bc_training_arm_ring_r08_residual120 | [run_20261104_174723_237778](artifacts/RoboEnv/bc_training_arm_ring_r08_residual120/run_20261104_174723_237778/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | baseline_model.pt, best_model.pt, last_model.pt | 43 |
| bc_training_arm_ring_r08_rl_td3 | [run_20261104_195549_802858](artifacts/RoboEnv/bc_training_arm_ring_r08_rl_td3/run_20261104_195549_802858/ARCHIVE_README.md) | 程序冒烟检查，不计正式成绩 | best_model.pt, initial_model.pt, last_model.pt | 见原history/config |
| bc_training_arm_ring_r08_rl_td3 | [run_20261104_200609_974210](artifacts/RoboEnv/bc_training_arm_ring_r08_rl_td3/run_20261104_200609_974210/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt, initial_model.pt, last_model.pt | 见原history/config |
| bc_training_arm_ring_r08_rl_td3_conservative | [run_20261104_204551_439479](artifacts/RoboEnv/bc_training_arm_ring_r08_rl_td3_conservative/run_20261104_204551_439479/ARCHIVE_README.md) | 程序冒烟检查，不计正式成绩 | best_model.pt, initial_model.pt, last_model.pt | 0 |
| bc_training_arm_ring_r08_rl_td3_conservative | [run_20261104_205012_830783](artifacts/RoboEnv/bc_training_arm_ring_r08_rl_td3_conservative/run_20261104_205012_830783/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt, initial_model.pt, last_model.pt | 0 |
| bc_training_nav_ring_r08 | [run_20261103_022339_570160](artifacts/RoboEnv/bc_training_nav_ring_r08/run_20261103_022339_570160/ARCHIVE_README.md) | 历史正式训练运行／模型快照 | best_model.pt | 120 |
