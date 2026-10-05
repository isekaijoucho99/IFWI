# IFWI 实验使用说明

所有命令从仓库根目录执行，使用已安装兼容 torch / torchvision 和 `requirements.txt` 的 Python 环境。`experiments/requirements.txt` 引用同一套基础依赖，也可用 `python -m pip install -r experiments/requirements.txt` 安装。

公开前已在本地通过 148 项 CPU 小网格测试，测试程序仅保留本地。验证正演、梯度、更新和恢复行为不等于证明完整规模的反演精度；范围见 [发布检查](../docs/release-check-20261005.md)。

## 选择入口

| 用途 | 入口 / 控制配置 | 默认网络 / seed / 更新数 |
|---|---|---|
| 单次参数实验，可组合自定义 | 根目录 `experiment.py` | 4×128 / 3 / 4001 |
| 固定十组单变量参数矩阵 | `run_parameter_sweep.py`；`baseline.yaml` / `legacy_random_baseline.yaml` | 4×128 / 3 / 4001 |
| 方法消融 | `run_experiment.py`；`feature_baseline.yaml` | 4×256 / 42 / 4000 |

前两项直接使用原 `IFWI2D.train()`、`save_state()` 和 `predict()`；第三项使用独立训练循环。参数实验与方法框架的控制组、best 选择和 checkpoint 格式不同，请在各自协议内比较。

## 单次参数实验

`experiment.py` 无参数运行随机 baseline，支持 `-shots`、`-depth`、`-width`、`-omega`、十种 `--preset` 以及组合覆盖：

```bash
python experiment.py
python experiment.py -shots 25 -depth 6 -width 192 -omega 20
python experiment.py --preset shots_25 -width 192
python experiment.py -omega 15.5 --dry-run
```

完整默认参数、预设表、输出与续训示例见 [项目 README](../README.md)。自定义值和多个参数组合由该入口支持；下述批量矩阵仍限定为注册的单变量设计。

## 十组单变量参数矩阵

矩阵包括 baseline，以及炮数 25/49、隐藏层数 6/8、宽度 256/512、omega 10/20/50，其他设置保持 baseline 一致。正式协议固定 seed=3、4001 次更新、记录间隔 100；对应原 epoch 标签 0..4000，保存点为完成更新 1、101、…、4001。完整炮集与 1000 点时间记录一起输入，不支持 `--shot-batch-size`。

先查看配置，再按需启动整套训练：

```bash
python experiments/run_parameter_sweep.py --output-dir results/parameter_sweep --device cuda:0 --dry-run
python experiments/run_parameter_sweep.py --output-dir results/parameter_sweep --device cuda:0
```

没有 GPU 时可指定 `--device cpu`，完整规模计算较慢。增加炮数、深度或宽度可能显著增加显存；25/49 炮完整输入的需求尚未实测。短验证须显式加 `--preliminary`，例如：

```bash
python experiments/run_parameter_sweep.py --output-dir results/parameter_smoke --device cuda:0 --iterations 3 --log-interval 3 --preliminary
```

使用已经完成的原随机 baseline 是可选路径。仅在自己已有相应目录时，将下面的占位路径替换为实际位置；源码仓库没有附带这些训练结果：

```bash
python experiments/run_parameter_sweep.py --output-dir results/parameter_sweep_import --device cuda:0 --completed-baseline "YOUR_COMPLETED_BASELINE_DIRECTORY"
```

导入会核验原配置、初始模型、真值、核心源码与 4001 次更新预算，保留编号 checkpoint 及哈希，跳过 baseline 训练。原观测未归档时按相同采集几何重新生成。该选项接受符合原 legacy 运行结构的 baseline，并非任意训练目录。若只完成部分训练，可用 `--reuse-baseline` 提供原编号 checkpoint；它与 `--completed-baseline` 互斥。

每组运行记录在 `runs/` 和 `logs/`，冻结源码及配置保存在 `source/` 和 `configs/`。每组结束后更新 `summary.csv`、`report.md`、`parameter_effects.png`；失败状态保留，已完成 baseline 导入失败时停止，其余训练失败会记录后继续其他注册项。汇总时可单独运行：

```bash
python experiments/summarize_parameter_sweep.py --suite results/parameter_sweep
```

原训练输出使用 `loss.csv` 和 `checkpoints/MarmousiI_random-checkpoint-N.pth`。best 按记录点更新前 loss 选择，保存更新后权重；末步与 best 分开保存，末步空间指标不包含额外波形 MSE 评估。协议定义见 [baseline_protocol.py](baseline_protocol.py)，发布对照和可移植限制见 [发布检查](../docs/release-check-20261005.md)。

## 方法消融

先运行一个短检查，以下两个命令分别创建独立运行目录：

```bash
python experiments/run_experiment.py --config experiments/configs/feature_baseline.yaml --device cuda:0 --iterations 3 --log-interval 3 --output-dir results/feature_smoke
python experiments/run_experiment.py --config experiments/configs/depth_weighted_loss.yaml --device cuda:0 --iterations 100 --log-interval 10 --output-dir results/feature_stability
```

方法框架默认 seed=42，主干为四层 256 宽 SIREN，omega=30；使用相同 94×288 网格、13 炮、1000 时间点和 dt=1.9 ms。未传 `--iterations` 时采用各 YAML 的预算，常规配置为 4000 次更新，时空探测配置为 100。可用 `--seed` 指定方法对照的种子。默认无有效裁剪，`training.clip_grad` 设置正有限数才开启。短跑不会改变配置中的 LR 调度总周期或时空频带阶段。

| 配置 | 实验内容 |
|---|---|
| `feature_baseline` | 方法控制组：波形 MSE + Adam |
| `depth_weighted_loss` | 物理速度场梯度的深度预条件；数据 MSE 本身不加权 |
| `attention` | 基于固定物理深度坐标的输出注意力 |
| `attention_residual`、`attention_film` | 有界残差输出调制 / 隐藏通道 FiLM；末层零初始化 |
| `adaptive_lr`、`layerwise_conservative` | 网络层分组学习率；不等于地下不同深度的学习率 |
| `prior_only` | 深部水平平滑与速度范围先验，默认关闭单调性约束 |
| `prior_normalized_tv`、`prior_normalized_charbonnier` | 用 1000 m/s 尺度归一化的先验及横向差异核 |
| `data_huber` | Huber 波形目标；另外报告普通 data MSE |
| `combined_best` | 注意力 + 梯度预条件 + 分层学习率候选组合，不含先验；名字不代表已证实最优 |
| `st_multiscale`、`st_time`、`st_space`、`st_joint` | 分阶段频带、时间 / 空间权重的探测对照 |

`gradient_preconditioner` 沿速度场的 nz 维修改正演速度副本的反传梯度，平均权重归一化为 1；不修改正演值，也不重复加权独立先验梯度。`layerwise_lr` 按网络早/晚层分组。时空方案不含显式断层注意力，空间响应权重也不是物理照明。

十组方法矩阵默认运行 100 次更新、每 10 步评估、seed=42，串行执行；其 `baseline` 名称映射到 `feature_baseline.yaml`：

```bash
python experiments/run_ablation_suite.py --output-dir results/ablation100 --dry-run
python experiments/run_ablation_suite.py --output-dir results/ablation100 --iterations 100 --log-interval 10 --seeds 42
python experiments/run_ablation_suite.py --output-dir results/spatiotemporal100 --configs baseline st_multiscale st_time st_space st_joint --iterations 100 --log-interval 10 --seeds 42
```

`--configs` 指定子集，`--seeds 3 42 123` 指定多个种子。默认十组不含 `adaptive_lr`、`combined_best` 或时空配置；前两项可用 `run_experiment.py` 单独运行。批次必须使用新的空输出目录，首个失败时停止，运行期间请勿修改受哈希核验的源码与配置。

## 方法框架输出与恢复

| 文件 | 含义 |
|---|---|
| `config.json` / `config.yaml`、`environment.json` | 实际配置、依赖版本与源码哈希 |
| `loss_history.csv` | 每步更新前目标；记录点另算更新后目标与数据 loss，梯度范数为裁剪前值 |
| `last.pth` | 可恢复状态：模型、Adam、LR 调度、随机状态、完成更新数、best 与协议信息 |
| `best.pth` | 评估用快照，不是恢复入口 |
| `metrics.json` / `v_pred.npy` | 已评估记录点中选择的 best |
| `final_metrics.json` / `last_velocity.npy` | 固定末步评估 / 速度，适合跨目标比较 |
| `training_summary.json`、`status.json` | 更新数、参数量、耗时、GPU 峰值显存与状态 |
| `result.png` | 同色标真值、预测及对称残差图，轴单位 km；由 `evaluation.save_plots` 控制 |

best 仅在记录点和最后更新评估，不是所有更新中的全局最优。常规方法按更新后训练目标选择，时空协议按原始 data MSE 选择；均不使用速度真值挑模型。`data_mse` 始终是普通波形 MSE，`data_objective` 是实际优化的数据目标；不同先验或损失组的 total loss 不能直接比较。

将下面路径替换为自己的 `last.pth`：

```bash
python experiments/run_experiment.py --config experiments/configs/feature_baseline.yaml --device cuda:0 --iterations 100 --resume results/feature_smoke/baseline_RUN/last.pth --output-dir results/feature_resumed
```

最终总更新数必须大于已完成数。默认检查网络、目标、物理参数、seed、观测和源码；可增加最终预算、改变记录间隔，但不能修改 LR 调度周期。恢复写新目录，旧的不完整状态 checkpoint 会拒绝，跨硬件 / 库版本不保证逐位一致。

## 方法结果比较

```bash
python experiments/compare_results.py --results-dir results/ablation100/runs
```

仅加载 `completed` 运行，检查观测、真值、采样、预算、主干和源码是否符合共同对照协议；比较目录应只存一个相同预算的批次。输出 `comparison/runs.csv`、`by_seed.csv`、`summary.csv`，保留各 run_id，先汇总同 seed 重复运行，再统计跨 seed 均值、标准差和数量。带 `final_` 的指标对应固定末步，其余对应 best，请勿混用。

常量真值的 SSIM、零真值梯度的梯度保真度会记为 null。梯度保真度可能为负，不证明物理分辨率。恢复运行的计时仅统计本次调用，速度比较需保持相同起点与预算。

这些配置用于检验方法假设。短验证、单一种子或更低训练 loss 均不足以证明精度提升；正式结论需相同协议和预算的多种子实验。
