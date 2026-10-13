# LR-IFWI：在 Residual IFWI 上加入低秩残差

对应论文：Chen, Wu, Li & Luo, *Full-waveform inversion with velocity model low-rank implicit neural representation*, IEEE TGRS 63 (2025), 5918816（下称 LR-IFWI）。论文未公开代码。

本实现只替换残差项，固定背景、正演、目标函数与 baseline1000 协议不变：

```text
v(z,x) = v_init(z,x) + 1000 · std_kmps · output_gain · [U C Vᵀ](z,x)
U = Ψ_z(coord_scale · z_km) ∈ R^{nz×rank_z}，V = Ψ_x(coord_scale · x_km) ∈ R^{nx×rank_x}，C ∈ R^{rank_z×rank_x}
```

尚未在 GPU 上训练，`output_gain` 尚未按全尺寸标定。

## 与论文的对应关系

| 论文 | 实现 |
| --- | --- |
| Eq.(4)(7) Ψ(x₁,x₂) = Ψ_ω₁(x₁) C Ψ_ω₂(x₂)ᵀ = U C Vᵀ | `experiments/low_rank_modules.py::LowRankResidual` |
| Eq.(5) 两支 MLP，各输入一维坐标 | 两个原版 `IRN`，`neuron = [1, *hidden, rank]`，sine、ω₀ = 30、线性输出 |
| §II-E 3 层、每层 128；r₁ = 50、r₂ = 100（94×288 Marmousi，n₁ = 94 为深度） | `hidden: [128,128,128]`、`rank_z: 50`、`rank_x: 100` |
| Stage 1：网络先拟合 m_init | 残差版本省略（见下文“论文模式”的更正）；论文模式 `initialization.pretrain` 实现 |
| 无显式正则 | 同样不加 |

参数量 90 910（z 支 39 730、x 支 46 180、C 5 000），baseline 2D SIREN 为 50 049。C 的有效秩不超过 min(rank_z, rank_x) = 50；rank_x = 100 只放宽 V 的表达，不提高秩上限。

## 论文模式（按原文步骤）

此前残差版本省略 Stage 1 的理由（"固定背景已承担初始模型"）不成立：固定背景只让起始**速度**相同，不让起始**网络状态**相同。论文中 U、V 先拟合平滑初始模型，反演开始时其列已是平滑基；残差版本的 U、V 停在随机 SIREN 初始化、C = 0，早期更新是梯度在随机高频特征上的投影。13 炮残差 LR（gain 5.50）复现了 IFWI13 的失败：RMSE 518.75 → 约 701 m/s，第 623 次更新局部速度超过 CFL 上限（5582.4 m/s）后回滚停止。

论文模式按 §II-E、§III-A/B 实现，配置 `lrpaper13_lr_ifwi.yaml`（LR-IFWI）与 `lrpaper13_ifwi.yaml`（论文对照用的 IFWI），二者只差 `experiment_name` 与模型部分（有测试保证）：

| 论文 | 实现 |
| --- | --- |
| m_pred = ξ·m̂ + μ，μ = 3、ξ = 1 km/s（§II-E） | `parameterization: absolute`，低秩时 m̂ = output_gain·U C Vᵀ，output_gain = 1 |
| 坐标 x₁ = 1…n₁、x₂ = 1…n₂（Eq. 6） | `low_rank.coordinates: index`，`coord_scale: 1` |
| C 随机初始化 | `low_rank.core_init: random`，分布论文未给，取 U(±1/√rank_x)；在两个因子网络之后抽取，因子权重与零核时逐位相同 |
| Stage 1：最小化 ‖Ψ − m_init‖²_F | `initialization.pretrain`：标准化输出上的 MSE、Adam；论文未给步数和学习率，取原 IFWI 预训练脚本 `pretrain_marmousi.py` 的 5000 次、1e-4、保留 loss 最低的状态 |
| Stage 2：Adam lr 1e-4、平滑初始 1000 epoch、无显式正则 | 预训练后新建 Adam；其余同 runner |
| IFWI 对照：sine、深度 4、128、Adam 1e-4（§III-A） | `lrpaper13_ifwi.yaml`：绝对 2D SIREN [2,128,128,128,128,1]，同样预训练 |
| 13 炮、炮距 300 m、震源深 30 m、检波器在地表、8 Hz、1.9 ms、1000 采样 | 炮距、频率、采样同论文；深度沿用原 IFWI 代码的震源 15 m（索引 1）、检波器 30 m（索引 2），见下文 |

预训练只用初始模型，真值只写入日志（`pretraining.json` 的 `logged_only_rmse_vs_truth_mps`）。输出 `pretraining_history.csv`、`pretraining.json`；`run_start_velocity.npy` 为预训练后的起点，与 `initial_velocity.npy`（高斯初始模型）不再逐位相同。续训从检查点恢复权重，不重复预训练。CPU 全尺寸预训练检查：LR 5000 次后与初始模型 RMSE 0.8 m/s、IFWI 0.5 m/s。

论文未给出、由我们决定的：高斯 σ（沿用 15 格，与所有 residual 运行一致）、预训练步数与学习率、C 的分布。整数坐标配合 ω₀ = 30 时首层相位在相邻格点间相差可达 30 rad，网格点上的拟合不受影响（上面的预训练结果），但 Lemma 1 的平滑常数很大，论文所述平滑先验在此是否成立需看训练结果；`coordinates: km` 可作消融。

采集深度：IFWI 与 LR-IFWI 论文正文均写震源 30 m、检波器在地表（FR-IFWI 写震源 15 m、检波器在地表），但 IFWI 作者公开代码用震源 15 m、检波器 30 m（[原代码说明](original_notes.md)），LR-IFWI 又称 IFWI 按其源码运行。论文实际所用几何无法确认，故 lrpaper13 沿用代码几何，与此前全部 13 炮运行一致（数据、采集、正演、训练各节与 `paper13_ifwi.yaml` 相同，有测试），可直接对照 IFWI13 / FR13。运行器也支持检波器放在第 0 行（自由表面下第 0 行不被置零，有测试），需要时可单独做几何对照。

```bash
python residual_ifwi.py --config experiments/configs/lrpaper13_ifwi.yaml --device cuda:0 --output-dir outputs/lrpaper13_ifwi
python residual_ifwi.py --config experiments/configs/lrpaper13_lr_ifwi.yaml --device cuda:0 --output-dir outputs/lrpaper13_lr_ifwi
```

论文模式不需要 `low_rank_calibration`（无额外增益）。

## 与论文不同或论文未说明的地方（残差版本）

1. **C 零初始化（论文为随机）**。第 0 步速度严格等于 v_init。第一次更新只有 C 有梯度（∂L/∂U = G V Cᵀ = 0），之后 U、V 才开始更新，对应 2D residual SIREN 末层清零时只有末层先有梯度。不能把 U 或 V 置零，否则梯度恒为 0。
2. **坐标**。论文写整数索引 1…n；配合 ω₀ = 30 会严重混叠，这里沿用 runner 的 km 坐标并乘 `coord_scale`。原版 IRN 首层权重为 U(±1/输入维)：2D 网络每轴 ±1/2，一维因子为 ±1。`coord_scale: 0.5` 使每个方向的首层频率范围与 baseline 相同。
3. **`output_gain`（我们加的）**。SIREN 线性输出层初值很小，因子元素约 0.05，C = 0 时首步速度变化远小于 baseline（94×288、2 炮、400 采样点的 CPU 检查：gain = 1 时首步 RMS 1.14 m/s，baseline 5.69 m/s）。用 `experiments/low_rank_calibration.py` 标定，见下文。
4. **只支持 residual 参数化**。`model.low_rank` 设置时 `model.neurons` 必须为 `null`，避免留下一个不起作用的网络配置。
5. **与 FR、可训练 ω 可同时开启**，不互斥：`model.fourier` 重参数化两个因子网络的隐藏层（需 `hidden` 至少 2 层），`model.trainable_omega` 包装两个因子网络的所有激活层。C 与因子输出层保持普通参数。

## 残差的低秩程度（94×288 网格，离线 SVD）

截断到秩 k 后的相对 Frobenius 误差：

| 矩阵 | k = 10 | k = 20 | k = 50 | 99 % 能量所需秩 |
| --- | ---: | ---: | ---: | ---: |
| 真值 | 0.077 | 0.052 | 0.020 | 6 |
| 平滑背景 | ≈0 | ≈0 | ≈0 | 1 |
| 真值 − 背景 | 0.440 | 0.300 | 0.119 | 56 |

论文所说的低秩主要来自低频背景，已由固定背景承担；残差上低秩约束主要起正则作用。按背景 RMSE 519 m/s 折算，秩 20 的截断下限约 156 m/s、秩 50 约 62 m/s，均远低于 baseline1000 的 438 m/s，秩不是当前瓶颈，因此值得做秩扫描。

## 首步速度变化的结构

同一 FD 梯度下（94×288、2 炮、400 采样点），baseline 与低秩模型第一步 Adam 更新都约 99 % 是常数平移（2D SIREN 秩 1 能量 0.992；低秩 1.000）：Adam 首步约为 lr·sign(g)，而 sine 特征含大比例常数分量。因子输出层偏置在初始化时占 U、V 能量的约 85–90 %，但将其置零只把常数占比从 99 % 降到 93 %，首步仍近似秩 1，因此保留原版初始化。`output_gain` 标定匹配的是首步整体步长，不是空间结构。

## 运行

先在 GPU 上标定 `output_gain`（只算一次全部 49 炮的 v_init 处梯度，约等于 6 次训练更新的正演，不训练）：

```bash
python -m experiments.low_rank_calibration --config experiments/configs/residual_lr_ifwi_baseline1000.yaml --reference-config experiments/configs/residual_ifwi_baseline1000.yaml --device cuda:0 --output outputs/low_rank_calibration.json
```

把输出的 `recommended_output_gain` 写入 `residual_lr_ifwi_baseline1000.yaml` 后训练：

```bash
python residual_ifwi.py --config experiments/configs/residual_lr_ifwi_baseline1000.yaml --device cuda:0 --output-dir outputs/residual_lr_ifwi_baseline1000
```

续训、`--crop/--shots/--nt` 小规模检查与 baseline 相同，见 [Residual IFWI 文档](residual_ifwi.md)。`--shots` 需配合自动生成炮点的配置（如 stage1），baseline1000 系配置写死了 49 个炮点。

## 13 炮协议（paper13）

49 炮每步随机 8 炮、约 5 s 记录的协议下 FR 后期停在约 430 m/s；13 炮、1.9 s、每步全炮的 paper13 协议下 FR + 可训练 ω 1000 步达 375 m/s，因此后续低秩实验以 paper13 为基础。两份配置相对各自基准只改 `experiment_name` 与模型部分（有测试保证，并满足标定脚本对其余设置完全一致的要求）：

| 配置 | 残差网络 | 标定参考 | 对照 |
| --- | --- | --- | --- |
| `paper13_lr_fr_ifwi.yaml` | 低秩 U C Vᵀ，FR 与可训练 ω 作用于两个因子网络 | `paper13_fr_ifwi.yaml` | FR13（375.3 m/s） |
| `paper13_lr_ifwi.yaml` | 低秩 U C Vᵀ，固定 ω、无 FR | `paper13_ifwi.yaml` | IFWI13（升到 749.0 m/s，未越过早期陷阱） |

先在 GPU 上分别标定，各只算一次全 13 炮梯度：

```bash
python -m experiments.low_rank_calibration --config experiments/configs/paper13_lr_fr_ifwi.yaml --reference-config experiments/configs/paper13_fr_ifwi.yaml --device cuda:0 --output outputs/low_rank_calibration_paper13_fr.json
python -m experiments.low_rank_calibration --config experiments/configs/paper13_lr_ifwi.yaml --reference-config experiments/configs/paper13_ifwi.yaml --device cuda:0 --output outputs/low_rank_calibration_paper13.json
```

把各自的 `recommended_output_gain` 写入对应配置后训练：

```bash
python residual_ifwi.py --config experiments/configs/paper13_lr_fr_ifwi.yaml --device cuda:0 --output-dir outputs/paper13_lr_fr_ifwi
python residual_ifwi.py --config experiments/configs/paper13_lr_ifwi.yaml --device cuda:0 --output-dir outputs/paper13_lr_ifwi
```

两份配置要分别标定、不能共用一个增益：CPU 小网格检查（24×288、2 炮、120 采样点）中，FR + ω 二维参考的首步 RMS 为 1.23 m/s，普通二维参考为 5.80 m/s，推荐增益分别约 1.2 与 4.9。这些只是流程检查，全尺寸数值须在 GPU 上重测。标定只匹配第一步步长；FR 层之后的有效步长放大（二维时中间层约 5.5–7.1 倍）不在其中。

## 诊断

奇异值谱、有效秩与检查点间相对变化（对应论文 Fig. 3 与 Fig. 23），只读已保存的速度数组，也可直接用于已有 baseline1000 输出：

```bash
python -m experiments.residual_lowrank_report outputs/BASELINE,outputs/BASELINE_RESUMED outputs/LR --labels baseline lr50 --output-dir outputs/lowrank_report
```

输出 `lowrank_summary.csv/png`、`lowrank_spectra.json` 与各次运行的 `lowrank_spectra_<label>.png`。r90/r99 为达到 90 %/99 % 能量所需的最小秩，熵有效秩为 exp(−Σ pᵢ log pᵢ)，pᵢ = sᵢ/Σs。相对变化 ‖v_b − v_a‖²/‖v_a‖² 按检查点窗口计算（论文按 epoch），只在检查点间隔相同的运行之间比较。

`experiments/residual_step_report.py` 支持低秩检查点，逐层报告 z 支、x 支与 C 的位移；层结构不同的运行（2D SIREN 与低秩）需分开报告。区域误差与分频带误差脚本只读速度数组，用法不变。

## 测试

```bash
python -m pytest verification/residual_ifwi/test_low_rank.py -q
```

覆盖：第 0 步等于背景、输出等于 gain·U C Vᵀ 且秩 ≤ rank_z、各因子只见一个坐标轴、首次梯度只到 C、与 FR/可训练 ω 组合、配置校验、真实小网格 FD 训练与断点续训逐位一致、续训拒绝改动的低秩配置、标定后首步 RMS 与参考一致、两个离线报告。
