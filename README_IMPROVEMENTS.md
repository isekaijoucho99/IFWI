# IFWI 方法消融框架

`experiments/` 提供注意力、梯度预条件、网络层学习率、先验、数据损失与时空权重的实验对照。它使用独立训练循环，原作者核心模块保留。新增配置是待检验的假设，运行成功或更低训练 loss 不代表已证明反演精度提升。

方法控制组为 **`feature_baseline.yaml`：4×256、seed=42、omega=30、4000 次更新**。根目录 `experiment.py` 与 `baseline.yaml` / `legacy_random_baseline.yaml` 则使用 **4×128、seed=3、4001 次更新** 的原训练协议。两套 baseline 不应互换。

## 开始

从仓库根目录执行，先安装兼容的 torch / torchvision，再安装依赖：

```bash
python -m pip install -r experiments/requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python experiments/run_experiment.py --config experiments/configs/feature_baseline.yaml --device cuda:0 --iterations 3 --log-interval 3 --output-dir results/feature_smoke
```

CPU 测试使用真实小网格正演、反向传播、更新和状态恢复。完整网格短跑仍需要正演与反演的显存；无 GPU 可指定 `--device cpu`。

## 对照配置

| 配置 | 改变的内容 |
|---|---|
| `feature_baseline.yaml` | 方法控制组：普通波形 MSE + Adam |
| `depth_weighted_loss.yaml` | 速度场梯度的深度预条件；MSE 本身不加权 |
| `attention.yaml` | 固定物理深度坐标的输出注意力 |
| `attention_residual.yaml` | 有界残差输出调制，末层零初始化 |
| `attention_film.yaml` | 输出线性层之前的隐藏通道 FiLM，末层零初始化 |
| `adaptive_lr.yaml`、`layerwise_conservative.yaml` | 网络层分组学习率，不是不同地下深度的学习率 |
| `prior_only.yaml` | 深部水平平滑与速度范围先验，默认关闭单调性约束 |
| `prior_normalized_tv.yaml`、`prior_normalized_charbonnier.yaml` | 归一化先验与不同横向差异核 |
| `data_huber.yaml` | Huber 波形目标，同时报告普通波形 MSE |
| `combined_best.yaml` | 注意力、梯度预条件与网络层学习率组合；名称不代表已证实最优 |
| `st_multiscale.yaml`、`st_time.yaml`、`st_space.yaml`、`st_joint.yaml` | 分阶段频带与时间 / 空间权重对照 |

注意力参数量单独报告，初始模型保存。残差与 FiLM 变体初始前向和主干梯度与方法 baseline 一致，但存在额外可训练参数。时空配置默认预算为 100 次更新，不含显式断层注意力；空间响应权重不等于物理照明。

## 批量与评估

```bash
python experiments/run_ablation_suite.py --output-dir results/ablation100 --dry-run
python experiments/run_ablation_suite.py --output-dir results/ablation100 --iterations 100 --log-interval 10 --seeds 42
python experiments/compare_results.py --results-dir results/ablation100/runs
```

默认十组包含方法 baseline、深度预条件、三种注意力、三种先验、Huber 与保守分层学习率。`--configs` 可选择子集或时空配置，`--seeds 3 42 123` 可运行多个种子。每个批次使用新的空目录，单 GPU 串行执行，失败保留日志并停止。

`metrics.json` 是已评估记录点中的 best，`final_metrics.json` 是固定末步，两者独立报告 `data_mse` 与 `data_objective`。常规配置按更新后训练目标选择 best，时空协议按原始波形 MSE 选择；不使用速度真值挑模型。不同目标的 total loss 不适合跨组直接比较，比较方法优先看相同预算的末步波形与模型指标。

安装、完整命令、输出格式、恢复和比较约束见 [experiments/README.md](experiments/README.md)。原版参数实验见 [项目 README](README.md)，公开发布代码对照与限制见 [发布检查](docs/release-check-20261005.md)。源码仓库不包含历史完整训练结果。
