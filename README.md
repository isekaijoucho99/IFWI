# IFWI — Implicit Full Waveform Inversion

基于 SIREN 隐式神经表示与有限差分波动方程的二维地震全波形反演代码。主要入口 `experiment.py` 用于随机初始化的 Marmousi 参数实验；另保留预训练、含噪观测、Dropout 不确定性、Overthrust 和传统 FWI 入口，以及独立的方法消融框架。

## 安装与检查

在仓库根目录使用 Python 3.10 环境。先安装与设备、驱动匹配且彼此兼容的 **torch / torchvision**，再安装仓库依赖；GPU 构建请使用对应的 PyTorch 安装命令。

```bash
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python experiment.py --help
python experiment.py --list-presets
python experiment.py --dry-run
```

第一条是通用安装示例，不固定 CUDA 构建。已有可用的 torch / torchvision 时可跳过。`experiments/requirements.txt` 引用同一套基础依赖；测试安装与运行：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

测试使用小网格验证真实正演、梯度、更新、参数校验、checkpoint 恢复与结果汇总；不运行完整训练。也可用 `python test_framework.py` 运行 unittest 测试入口。本地虚拟环境与 CUDA 版本不随仓库分发。

## 单次参数实验

以下命令彼此独立，每条只运行一次实验。无参数时启动 baseline 训练，无需修改 YAML：

```bash
python experiment.py
python experiment.py -shots 25
python experiment.py -depth 6
python experiment.py -width 192
python experiment.py -omega 15.5
python experiment.py -shots 25 -depth 6 -width 256 -omega 20
python experiment.py --preset width_512
python experiment.py --preset shots_25 -width 192 -omega 20
python experiment.py -shots 37 --dry-run
python experiment.py -width 192 --epochs 3
```

`-shots` / `--shots`、`-depth` / `--depth`、`-width` / `--width`、`-omega` / `--omega` 等价，允许同时修改多个参数。配置顺序是 baseline → `--preset` → 显式参数覆盖。研究某个参数的独立影响时保持其他设置一致；同时改变多个参数适合探索组合，结果不能直接归因于其中一项。

| 预设 | 相对 baseline 的变化 |
|---|---|
| `baseline` | 无 |
| `shots_25`、`shots_49` | 炮数 25、49 |
| `depth_6`、`depth_8` | 隐藏层数 6、8 |
| `width_256`、`width_512` | 每个隐藏层宽度 256、512 |
| `omega_10`、`omega_20`、`omega_50` | omega 10、20、50 |

这十个预设也允许继续覆盖。自定义炮数须为 2..241 的整数，深度与宽度须为正整数，omega 须为正有限数。

## 默认 baseline

`experiment.py` 调用原 `IFWI2D.train()`、`save_state()` 和 `predict()`，沿用原日志与编号 checkpoint 流程。原作者核心模块和参考入口的哈希会在训练前核验。

| 参数 | 默认值 / 行为 |
|---|---|
| 初始化 / 随机种子 | 随机初始化，seed=3 |
| 网络 | SIREN，输入 2 维、4 个隐藏层且每层 128、输出 1 维；线性输出层、含 bias |
| omega | 30，同时用于正弦激活与原网络权重初始化 |
| 速度归一化 | mean=3、std=1，单位 km/s；导出速度单位 m/s |
| 优化器 | Adam，lr=1e-4，betas=(0.9, 0.999)，eps=1e-8，weight_decay=0 |
| 训练预算 | 4001 次更新，原 epoch 标签为 0..4000 |
| 记录 / 保存间隔 | 100；完成更新 1、101、…、4001 时保存，共 41 份 checkpoint |
| best 选择 | 在记录点比较更新前 loss，保存该次更新后的权重 |
| loss | 原波形 MSE；alpha=0，TV 统计仍计算但不进入目标 |
| 梯度裁剪 | 保留原调用；参数生成器耗尽使原裁剪无实际效果 |
| 输入 | 所有炮、完整时间记录；不分炮累积梯度 |
| 数据 | `data/vel_marmousi_376x1151.csv`，默认 CSV header 读取、4 倍下采样，94×288，float32 |
| 空间 / 时间采样 | dx=dz=15 m，dt=0.0019 s，nt=1000 |
| 子波 | 8 Hz Ricker |
| 正演 | 有限差分阶数 2，PML 15，自由表面 |
| 震源 | 13 炮，源深网格索引 1；x 索引 20、40、…、260 |
| 接收器 | 每炮 288 个，接收深网格索引 2，覆盖全部 x 网格点 |
| 附加功能 | 无噪声、Dropout、预训练、先验、注意力、LR 调度或加速后端 |
| 设备 | 自动选择 `cuda:0`，不可用时选择 `cpu`；可用 `--device` 覆盖 |
| 输出 | `results/single_experiments/` 下的唯一运行目录；可用 `--output-dir` 修改父目录 |

自定义炮数保持 x 索引 20..260 的孔径，用均匀取点后取整生成位置；不一定包含原 13 个炮点。实际源与接收器位置保存于 `acquisition.json`。增加炮数、深度或宽度会提高计算和显存需求，25/49 炮的完整输入显存需求尚未实测。

`--epochs` 表示最终总更新数。`--log-interval` 同时控制 best 选择与 checkpoint 保存，最后一步也会保存；不是仅改变终端打印。偏离 4001 次更新或间隔 100 时，`run_metadata.json` 标记 `preliminary=true`。短跑只检查运行情况，不能代替完整预算的精度对照。

## 输出与续训

逐轮进度显示在终端并写入 `progress.log`，默认生成 `result.png`。每次运行建立新目录，不覆盖已有结果。

| 文件 | 内容 |
|---|---|
| `config.json`、`baseline_config.json` | 实际配置与原入口格式的配置 |
| `original_source_hashes.json`、`run_metadata.json` | 原代码身份、训练协议与 preliminary 标记 |
| `loss.csv` | 完成更新数及原 total loss、data loss、regularization statistic；loss 为更新前值 |
| `checkpoints/MarmousiI_random-checkpoint-N.pth` | 原编号 checkpoint，N 为已完成更新数 |
| `initial_velocity.npy`、`last_velocity.npy`、`best_velocity.npy` | 初始、末步、原 loss-best 速度 |
| `true_velocity.npy`、`observed.npy`、`acquisition.json` | 真值、合成观测与采集几何 |
| `metrics.json`、`final_metrics.json` | best 模型误差与固定末步空间指标；末步未额外计算波形 MSE |
| `training_summary.json`、`status.json` | 更新数、best 时刻、耗时、显存与运行状态 |

`v_true.npy` / `v_pred.npy` 分别是供汇总工具使用的真值 / best 别名。原 best 的更新前 loss 与更新后权重对应关系保留，不能将记录的 loss 当成保存模型的重新评估值。

续训需使用实际生成的编号 PTH，并保留同目录的 `config.json`。例如将下面的 `baseline_RUN` 换成自己的运行目录：

```bash
python experiment.py --epochs 4001 --resume results/single_experiments/baseline_RUN/checkpoints/MarmousiI_random-checkpoint-101.pth
```

最终预算须大于 checkpoint 已完成更新数。自定义实验需重复其 `-shots/-depth/-width/-omega` 设置；恢复会检查配置。续训写新目录，跨设备或依赖版本不承诺逐位一致。源码仓库不包含历史反演结果或 checkpoint 序列。

## 批量参数与方法对照

注册的十组单变量参数矩阵可用 `experiments/run_parameter_sweep.py` 运行；使用现成 baseline 是可选的，路径由用户提供，详见 [实验使用说明](experiments/README.md)。

方法消融采用另一套控制组：

| 用途 | 控制配置 | 网络 / seed / 总更新数 |
|---|---|---|
| 原训练协议的参数实验 | `experiment.py`、`baseline.yaml`、`legacy_random_baseline.yaml` | 4×128 / 3 / 4001 |
| 注意力、损失、梯度预条件等方法对照 | `feature_baseline.yaml` | 4×256 / 42 / 4000 |

两套入口的训练循环、best 选择与 checkpoint 格式不同，请在各自协议内比较。方法配置、批量运行、指标与恢复说明见 [experiments/README.md](experiments/README.md) 和 [方法框架概览](README_IMPROVEMENTS.md)。新增方法是待验证的实验假设，仓库不承诺固定精度提升。

## 保留的 legacy 入口

`main.py` 提供固定网络的多实验入口，`fwi.py` 提供独立传统 FWI 对照，`pretrain_marmousi.py` 生成平滑模型预训练权重。下面每条命令会单独开始训练：

```bash
python main.py --experiment random --epochs 4001 --plots
python main.py --experiment pretrain --epochs 4001 --plots
python main.py --experiment noisy --noise 2 --epochs 4001 --plots
python main.py --experiment uncertainty --dropout 0.2 --samples 1000 --epochs 4001 --plots
python main.py --experiment overthrust --epochs 4001 --plots
python fwi.py --experiment fwi_smooth --epochs 4000 --plots
python fwi.py --experiment fwi_noisy --noise 2 --epochs 4000 --plots
python pretrain_marmousi.py --epochs 5000 --output-dir outputs/pretrain_new
python main.py --experiment pretrain --epochs 4001 --pretrained outputs/pretrain_new/ifwi_pretrain_marmousi.pth
```

这些入口可用 `--device` 选择设备，用 `--help` 查看其余参数。`main.py` 默认不绘图；`--clip-grad` 可显式启用有效裁剪，`--accelerate` 限无噪声、无 Dropout 的 random/pretrain。其输出默认位于 `outputs/`。随仓库提供的 `weights/ifwi_pretrain_marmousi.pth` 是小型平滑模型预训练权重。

**Legacy 限制：** `main.py` 和 `fwi.py` 的 `fwi_random` 模式仍读取源码中指定的仓库外历史随机 baseline 初始模型及真值，CLI 没有路径覆盖参数。仅克隆本仓库不能直接运行该模式；上面的 `fwi_smooth` / `fwi_noisy` 使用仓库内数据。

## 来源与限制

原作者模块为 `ifwi_modules.py`、`rnn_fd.py`、`generator.py`、`plot_functions.py`，文件头署名与参考信息保留。补充入口及 `experiments/` 不应全部署名为原论文作者的发布实现；具体来源与引用见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。来源材料没有可确认的上游 LICENSE，本仓库不擅自授予新许可。

本版本 Marmousi 震源深度 15 m、接收深度 30 m；采样与其他论文版本可能不同。legacy `regularization` 的原 auto-TV 逻辑及原 IFWI 非有限波场处理保留；它们不能作为有效 TV 正则化或数值稳定性的证据。结果受训练预算、硬件、依赖与随机性影响，本仓库不代表论文全部实验的精确复现。历史记录保存在 `docs/`，其中的本机环境、旧路径与旧验证结论仅属于各自记录。
