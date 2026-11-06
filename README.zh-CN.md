# COMP0245 课设代码：从直接力矩模仿到结构化反馈与连续巡检

[English](README.md) | **简体中文**

**小组：** Mingzhou Li、Shuchang Zhang、Boyan Li，University College London。  
**整理日期：** 2026年11月6日。  
**任务：** 使用监督学习完成移动机械臂末端位置到达，并探索连续多目标巡检。

本仓库的 `main` 提供当前作业提交版本；原 `main` 的已发布基线保存在 [`baseline` 分支](https://github.com/Waller-Shaun/robotnik-split-bc/tree/baseline)。此前研究与巡检开发分支继续保留。这里包含报告涉及的源代码、主要可运行模型及实验结果记录；可选的历史检查点和恢复示范原始数组不在当前提交版中。

这次整理没有启动新训练或仿真。原实验代码、已保留的权重、归一化和记录按原字节复制；新增内容是顶层运行入口、使用说明、代码索引及提交完整性检查。

## 从这里开始

- [快速运行说明](QUICKSTART.md)：环境准备、单点演示、巡检、批量重跑、输出位置与独立训练入口。
- [报告与代码对应表](REPORT_CODE_MAP.md)：报告中的实验编号、源码、模型与结果位置。
- [来源与修改说明](SOURCES_AND_MODIFICATIONS.md)：课程和上游代码来源、小组实现及修改范围。
- [打包范围与复现边界](BUNDLE_SCOPE.md)：包含和省略的材料、历史路径限制及校验范围。

无需安装机器人仿真依赖即可检查提交文件和重数已有结果：

```powershell
python -X utf8 submission.py verify
python -X utf8 submission.py results
python -X utf8 submission.py --help
```

`verify` 检查文件哈希、Python语法与JSON，不导入实验模块，不需要Torch/PyBullet，也不会训练、仿真或联网。`results` 仅统计已保存的逐次结果，不是重新运行物理评估或重新审计完整原始轨迹。

## 三个主要运行方案

| 方案 | 主要结构 | 运行入口 |
|---|---|---|
| 固定PD结构化TCN，单目标 | TCN学习运动场与重力补偿，积分器维护参考角，固定100/30的PD生成手臂力矩；导航模型冻结 | `submission.py fixed-tcn` |
| 动态PD结构化TCN，单目标 | 冻结原运动／重力／导航网络，新增4590参数增益网络，预测7个P和7个D，并进行因果平滑 | `submission.py dynamic-pd` |
| D3恢复MLP，连续巡检 | 冻结学习到的导航与直接力矩模型，保留目标间真实物理状态；增加相对停车点、访问排序和任务监测 | `submission.py patrol` |

固定与动态PD均是明确披露的**结构化混合控制**，不是自由的端到端直接力矩回归。固定P=100、D=30沿用了生成示范数据的专家设置，并未证明最优。动态版的P范围80–120、D范围24–36以及滤波设置由人为设计；本次只训练增益网络。

巡检使用的是D3直接力矩模型，**不是**将高成功率TCN用于连续巡检后的成绩。任务是末端到达与保持，没有实现实体抓取或避障。

## 新电脑准备与最简演示

在仓库根目录执行。需要Python、Git和[Pixi](https://pixi.sh/)。Windows建议使用较短的克隆路径，因为研究归档中部分历史路径较长。

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
python -X utf8 structured/baseline/tools/setup_dependencies.py
pixi install --manifest-path structured/baseline/external/RoboEnv/pixi.toml -e ml --locked
pixi run --manifest-path structured/baseline/external/RoboEnv/pixi.toml -e ml --locked python -X utf8 structured/tools/verify_experiment.py --runtime
```

准备工具按指定提交下载RoboEnv及其仿真子模块，需要联网，但不训练或运行任务。不同版本的已有目录不会被静默替换。Linux/macOS的环境变量语法及完整步骤见[QUICKSTART.md](QUICKSTART.md)。

固定PD TCN，显示一个目标的仿真：

```powershell
pixi run --manifest-path structured/baseline/external/RoboEnv/pixi.toml -e ml --locked python -X utf8 submission.py fixed-tcn --target -0.9 1.2 1.1 --seed 137
```

动态PD，使用相同目标与种子：

```powershell
pixi run --manifest-path structured/baseline/external/RoboEnv/pixi.toml -e ml --locked python -X utf8 submission.py dynamic-pd --target -0.9 1.2 1.1 --seed 137
```

D3随机五目标巡检：

```powershell
pixi run --manifest-path structured/baseline/external/RoboEnv/pixi.toml -e ml --locked python -X utf8 submission.py patrol --num-targets 5 --target-seed 1266108476 --height-range 0.5 1.3
```

添加 `--headless` 可关闭窗口。巡检支持重复传入 `--target X Y Z` 指定目标，以及调整数量、半径、高度、访问顺序等参数，详见[巡检说明](direct/patrol/README.md)。顶层入口会在切换子程序工作目录前，将显式路径参数按调用者当前目录解析。

固定TCN、动态PD与巡检可共用上述安装环境，但各自仍加载绑定的模型、统计与配置。共享资源目录未被重新用于复现全部旧巡检物理轨迹，不能保证跨机器逐步一致。

## 已记录结果

| 研究 | 评估协议 | 结果 |
|---|---|---|
| P0已发布直接力矩基线 | 144个验证目标，无噪声／带噪声 | 111/144／106/144 |
| C0纯力矩继续训练 | 相同144个带噪验证目标 | 113/144 |
| D3专家恢复示范微调 | 相同带噪验证集；另有独立带噪测试集 | 120/144；132/156 |
| D3上的监督残差R2 | 相同带噪验证集 | 122/144；没有声称完成156点独立测试 |
| 固定TCN原主测试 | 500个新目标，每点3个噪声种子 | 1486/1500＝99.07%；无噪声493/500 |
| 固定PD，配对预留测试 | 另一批500个目标，每点3个噪声种子 | 1495/1500＝99.67% |
| 动态PD，配对预留测试 | 与上一行完全相同的目标和种子 | 1492/1500＝99.47%；配对差值−0.20个百分点，置信区间包含零 |
| 连续巡检 | 独立开发演示记录 | 保留成功、超时及未尝试目标，不汇总为独立任务成功率 |

单目标成功标准为：**20秒内任意时刻，PyBullet真实末端误差严格小于1cm**，不要求保持。带噪协议使用八项原生测量噪声，标准差各为0.001；它们不是外力扰动。

巡检以**观测误差小于5cm并连续保持1秒**切换目标，真实位置另行评分。这与单目标的1cm到达成功率不是同一指标。

完整系统的改善不能全部归因于TCN或PD，因为输入特征、监督方式和控制结构同时发生了变化。动态PD降低了离线力矩误差，但没有证明闭环成功率提升。RL和CNN-Transformer的失败／中断记录也保留，未将部分评估冒充完整基准结果。原156目标集、主测试500目标集和预留500目标集相互不同；1500次重复不等于1500个独立目标。

## 文件组织

```text
direct/
  LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task/
                           原基线采集、划分、训练与测试代码
  research/source/         GRU、辅助监督、恢复示范、残差、RL、Transformer等研究源码
  research/artifacts/      历史配置、归一化、训练过程和结果；历史PT检查点省略
  research/data/           恢复采集清单和诊断；原始恢复NPZ省略
  patrol/                  D3多目标巡检代码与记录
  artifacts/patrol120/     巡检所用主模型、统计与配置
  collaborator/            组员原固定TCN发布包
structured/
  baseline/                动态PD绑定的原固定TCN发布包
  dynamic_pd/              增益网络、数据处理、训练、控制与评估
  tools/                   便携校验、基准重跑与增益重训练
  runs/full/               最佳第17轮增益模型及训练记录
  outputs/                 已记录的开发与最终逐次结果
submission.py              统一显式运行入口
submission_tools/          静态完整性与结果检查
```

`direct/`和`structured/`内重复的基线文件为保留相对路径和原校验关系而存在，请勿自行合并、改名或混用归一化文件。

## 训练、测试与数据范围

训练和测试是分开的入口。演示或批量测试不会自动启动训练，也不会更新已有模型。使用预训练主模型不需要原完整数据集；重训练需自行恢复数据，并指定全新的输出目录。便携增益重训练使用 `submission.py retrain-gain`，详细命令见[QUICKSTART.md](QUICKSTART.md)。

当前保留主方案模型和小型示例，省略了61个历史研究PT检查点、102条恢复示范原始NPZ、完整1131条基础数据、完整原始评估轨迹、仿真资源与Python环境。配置、清单、源代码快照和逐次结果仍保留；精确省略范围见[OMITTED_OPTIONAL_ARTIFACTS.json](OMITTED_OPTIONAL_ARTIFACTS.json)。当前提交以自己的 `SUBMISSION_MANIFEST.json` 校验，不能用缺少可选文件的精简目录宣称原完整归档检查已通过。

历史研究代码保留原机器路径和父模型关系，并非每个旧实验都能在新电脑上一键重训。顶层支持的可移植演示、基准重跑和增益重训练已单独标明。不要直接执行历史一次性 `bootstrap.py`，也不要批量改写原校验哈希。

## 来源与贡献

课程任务和原专家控制器来自[课程公开仓库](https://github.com/VModugno/LAB_SESSION_COMP_0245_2026_PUBLIC/tree/main/final/robotnik_arm_task)。环境使用[RoboEnv](https://github.com/VModugno/RoboEnv)及其[simulation_and_control](https://github.com/VModugno/simulation_and_control)子模块；网络与优化器基于[PyTorch](https://pytorch.org/)。上游仿真引擎和机器人资源不属于本组从零实现的内容。

小组实现包括数据采集与划分、特征和分离策略、模型优化与恢复示范、结构化TCN、动态PD消融、连续巡检及评估工具。个人分工以最终报告的贡献声明为准。原方法、文献和代码来源说明均保留，详见[SOURCES_AND_MODIFICATIONS.md](SOURCES_AND_MODIFICATIONS.md)。代码准备使用了AI辅助，小组负责审阅、理解并按课程要求披露。

固定依赖版本：RoboEnv `7bb49439677c2ac111347b40e997025ce87bff50`；simulation_and_control `df5850526affcc33b25b15d3096c4d21b614bacd`。具体模型与文件哈希见原发布清单及顶层提交清单。
