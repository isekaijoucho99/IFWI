# Residual IFWI：1000 epoch baseline

本仓库另提供固定平滑初速度加 SIREN 残差的独立入口 [`residual_ifwi.py`](../residual_ifwi.py)。今后的 Residual IFWI 对照以 [`residual_ifwi_baseline1000.yaml`](../experiments/configs/residual_ifwi_baseline1000.yaml) 的 **1000 次 Adam 更新后模型**为 baseline。1000 是用户指定的比较预算，没有按真值误差选择最优检查点，也不表示已经收敛。原版 [`experiment.py`](../experiment.py) 和原作者核心源码保持原样，原随机初始化 baseline 仍可独立运行。

## 初速度、残差与单位

```text
v_pred(x,z;θ) = v_init(x,z) + 1000 × std_kmps × IRN(x,z;θ)
```

物理速度、固定背景及导出的数组单位均为 m/s。`std_kmps=1`，网络输出 1 对应 1000 m/s 增量；残差分支不加 `mean_kmps=3`。坐标顺序为 `(x,z)`，单位 km。背景是脱离梯度的固定缓冲区，只有原 SIREN 参数参加优化。新建残差实验的末层权重和偏置清零，因此初始预测严格等于背景；续训恢复已有残差。

数据来自 `data/vel_marmousi_376x1151.csv`，按 m/s 读取，沿两个轴 `[::4]` 采样为 94×288、15 m 反演网格。`csv_header: legacy` 保留原 Pandas `header=0` 行为：376×1151 数值 CSV 的首行被当作表头，读取后为 375×1151，再下采样。改为 `none` 会改变数值和深度采样起点，应作为另一项数据协议。

先读取并转换单位、下采样、可选裁剪、转 float32，随后生成背景：

```python
v_init = scipy.ndimage.gaussian_filter(
    sampled_truth, sigma=15.0, mode="reflect", truncate=4.0
).astype(np.float32)
```

sigma=15 是反演网格点数，两轴均对应标准差 225 m；平滑发生在采样之后。已有 `vel_marmousi_smooth400_376x1151.csv` 只用于来源审计，其未知生成配方不定义此背景。真值用于合成脱离梯度的观测、生成指定背景和离线评价，不作为网络训练标签；背景包含合成真值的低频信息。

## 固定实验协议

| 项目 | baseline1000 |
| --- | --- |
| seed / 网络 | 3；原 SIREN `[2,128,128,128,128,1]`，omega=30，线性输出，无 Dropout |
| 炮点 | 49 炮，x 索引 20,25,…,260；源深索引 1，接收深索引 2，每炮 288 个接收器 |
| 每次更新 | 从 49 炮均匀无放回抽 8 炮，下次独立重抽；两个 4 炮微批次累积梯度后执行一次 Adam |
| 时间记录 | dt=0.0019 s，nt=2632，末采样时刻 4.9989 s；nt×dt=5.0008 s，约 5 s；完整记录，无时间分段 |
| 正演 | 原二阶 FD，8 Hz Ricker，PML 15，自由表面，反射参数 1e-6 |
| 目标 / 优化器 | 抽中 8 炮、全部时间和接收器的普通波形 MSE；Adam lr=1e-4，固定学习率 |
| 预算 / 保存 / 评价 | 1000 epoch；每 epoch 恰好一次 Adam 更新；每 50 次更新保存并评价全部 49 炮，末步也评价 |

微批次按本次总抽样炮数 8 归一化，保持同一参数值累积梯度。其梯度期望对应 49 炮平均 MSE，Adam 轨迹仍会受抽样影响。1000 次更新训练共 8000 炮次，观测生成及评价另计。

逐步 `sampled_loss.csv` / `sampled_training_history.json` 的 loss 和空间指标对应**更新前**模型；`loss.csv` / `full_evaluation_history.json` 的全炮评价及 `metrics.json.final` 对应**更新后**模型。额外的 `common_1p9s_data_mse` 为 49 炮前 1000 点，`baseline_13shots_1p9s_data_mse` 为原 13 炮位置前 1000 点；两者仅用于评价。

同为 seed=3 不会消除实验设计差异。相对原随机绝对速度 baseline，固定背景、末层零初始化和残差参数化同时改变；此协议另改成 49 炮、随机抽炮、约 5 s 记录及 1000 次更新。独立 runner 还去掉未进入目标的 TV 坐标导数统计，遇非有限值报错，不掩盖为零，并显式评价末步模型。任何改善都不能仅归因于“加入初速度”或残差表达式。

## 可移植运行与续训

先按 [README 安装说明](../README.md#安装与检查) 准备 Python 及适配本机设备的 PyTorch，在克隆仓库根目录运行以下命令。源码不依赖记录实验时的本机绝对路径。

检查配置不运行正演：

```bash
python residual_ifwi.py --config experiments/configs/residual_ifwi_baseline1000.yaml --dry-run
```

从零完成 1000 次更新：

```bash
python residual_ifwi.py --config experiments/configs/residual_ifwi_baseline1000.yaml --device cuda:0 --output-dir outputs/residual_ifwi_baseline1000
```

无 CUDA 可用时改为 `--device cpu`。每次在指定父目录中新建唯一运行子目录，实际目录由程序打印。`--epochs` 是累计总更新数。将下面 `RUN_ID` 换成自己的运行目录；若该检查点已完成 800 次更新，恢复到 1000 只新增 200 次更新：

```bash
python residual_ifwi.py --config experiments/configs/residual_ifwi_baseline1000.yaml --epochs 1000 --device cuda:0 --resume outputs/residual_ifwi_baseline1000/RUN_ID/checkpoint.pt --output-dir outputs/residual_ifwi_baseline1000_resumed
```

恢复保留模型、固定背景、Adam、完成更新数、历史和 Python/NumPy/Torch/CUDA 随机状态；中断时保存的 `interrupted.pt` 也可作为恢复来源。除累计预算和保存/评价间隔外，配置须匹配；数据/背景来源和原核心源码身份也须匹配。来源记录包含实际数据路径，移动工作目录可能使严格来源校验拒绝恢复。随机抽样由 NumPy MT19937 状态继续。小规模真实 FD 验证已覆盖连续运行与断点恢复的参数和随机序列一致性；跨设备、依赖版本或浮点执行差异不承诺逐位一致。

**历史 800→1000 记录的配置名仍是 `residual_ifwi_49shots_5s_800`。**配置名也参与恢复校验；恢复该历史检查点应使用它原来的配置并覆盖总预算：

```bash
python residual_ifwi.py --config experiments/configs/residual_ifwi_49shots_5s_800.yaml --epochs 1000 --device cuda:0 --resume outputs/HISTORICAL_800_RUN/checkpoint.pt --output-dir outputs/residual_ifwi_historical_resumed
```

新 baseline1000 配置相对 800 配置只改 `experiment_name` 与 `training.epochs`。两者计算协议一致；结果记录保留实际历史配置名，不将改名后的配置伪装成当时的运行文件。

## 小规模检查

以下使用可自动生成炮点的 stage1 配置，运行 CPU 小网格、两炮、100 点、两次更新，检查数据、正演、梯度、保存和绘图流程：

```bash
python residual_ifwi.py --config experiments/configs/residual_ifwi_stage1.yaml --device cpu --crop 24 64 --shots 2 --nt 100 --epochs 2 --checkpoint-interval 1 --output-dir outputs/residual_ifwi_smoke
```

它不使用 baseline1000 的显式 49 炮点，也不验证完整预算精度；裁剪后再平滑的背景与全网格背景的裁片不同。已有验证代码位于 [`verification/residual_ifwi`](../verification/residual_ifwi)：

```bash
python -m pip install pytest
python -m pytest verification/residual_ifwi -q
```

本次发布检查通过该目录全部 165 项测试（53.19 s），保留原版空参数迭代器梯度裁剪的一条既有警告。

已有分阶段续训输出可每 100 次更新生成论文样式对照图；将 `CONTROLLER_RUN` 换成包含 `stages/` 的实际目录：

```bash
python scripts/plot_residual_every100.py --controller-root outputs/CONTROLLER_RUN --output-dir outputs/residual_ifwi_every100
```

## 已记录的 1000 次结果

实测链为 0→800→1000：从新建残差模型训练 800 次，再恢复模型、Adam 和随机状态到累计 1000 次。采用该阶段保存的 `final_velocity.npy` / `checkpoint.pt` 及对应 `metrics.json.final`，并非从后续 2000 次训练的更新前日志截取。

| 指标 | 固定平滑初速度（0 次） | 第 800 次模型 / 续训起点 | 第 1000 次更新后 |
| --- | ---: | ---: | ---: |
| RMSE（m/s） | 518.752793 | 454.054981 | 438.387596 |
| MAE（m/s） | 348.328739 | 261.397024 | 248.866346 |
| SSIM | 0.381094 | 0.589978 | 0.609624 |
| 49 炮完整记录 MSE | 0.0149615651 | 0.0001663704 | 0.0005433399 |
| 49 炮前 1000 点 MSE | 0.0354954688 | 0.0002535354 | 0.0009028941 |
| 原 13 炮前 1000 点 MSE | 0.0347798544 | 0.0002618709 | 0.0007871615 |

1000 次阶段 `metrics.json.initial` 表示 800 次恢复起点，`fixed_background` 才是固定初速度。SSIM 使用真值范围 4000 m/s。空间误差在此阶段改善，完整波形 MSE 高于第 800 次值；1000 预算不构成各指标最优或收敛证据。同一后续续训链在 1900 次全炮评价记录 RMSE 391.995230 m/s，2000 次末步 RMSE 402.192688 m/s、SSIM 0.645306，进一步说明 1000 是指定比较点。

记录环境为 Python 3.10.18、PyTorch `2.9.0.dev20250904+cu128`、NumPy 2.2.6、SciPy 1.15.3、CUDA 12.8、NVIDIA GeForce RTX 5060 Ti，torch CPU 线程数 1。0→800 训练耗时 27333.56 s，800→1000 训练耗时 7435.46 s，合计约 9 h 39 min；各段准备分别 186.15 s、188.63 s，训练耗时含周期全炮评价，均不含后续绘图。峰值 CUDA 分配为 9.2909 GiB。这是实测环境与分段耗时，未另行测量一次启动的 fresh 1000 次耗时或验证其逐位相同结果。

便于追溯的小型记录见 [`residual_ifwi_baseline1000.json`](experiments/residual_ifwi_baseline1000.json)，包括实际配置、数值、断点链、数据/原核心/残差源 SHA256 及保存产物哈希。历史 CPU 审计通过，重算了保存数组指标与检查点一致性；该审计未重新计算完整 FD 波形，不证明物理正确性或收敛。仓库不附带本次大体积观测、权重、历史输出或本机虚拟环境，复现实验会自行生成这些文件。
