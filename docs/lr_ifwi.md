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
| Stage 1：网络先拟合 m_init | 不需要：固定背景已承担初始模型 |
| 无显式正则 | 同样不加 |

参数量 90 910（z 支 39 730、x 支 46 180、C 5 000），baseline 2D SIREN 为 50 049。C 的有效秩不超过 min(rank_z, rank_x) = 50；rank_x = 100 只放宽 V 的表达，不提高秩上限。

## 与论文不同或论文未说明的地方

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
