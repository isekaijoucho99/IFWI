# FR-IFWI：在 Residual IFWI 上加入 Fourier 重参数化

对应论文：Kang, Chen, Yang, Li & Wu, *Implicit full waveform inversion with adaptive Fourier frequency bases learning*, GJI 244 (2025), ggaf404（下称 FR-IFWI）。论文未公开代码；Fourier 基的具体构造沿用其方法来源 Shi et al. (CVPR 2024) 的公开实现 [FR-INR](https://github.com/LabShuHangGU/FR-INR)（`modules.py::sin_fr_layer`、`2d_image_fitting.py`）。

fr_inr 主配置已在 baseline1000 协议上训练 1000 次（RMSE 429.6 m/s、SSIM 0.573，baseline 为 438.4 / 0.610）；可训练 ω、修正后的 pinv 与 paper13 配置尚未训练。

## 与论文的对应关系

| 论文 | 实现 |
| --- | --- |
| Eq.(5) W⁽ⁿ⁾ = Λ⁽ⁿ⁾B⁽ⁿ⁾，Λ 可训练、B 固定 | `experiments/fourier_modules.py::FRLinear`，`weight = lamb @ bases`，`bases` 为不可训练 buffer |
| Eq.(6) b_ij = cos(ω_i z_j + φ_i)，φ ∈ {0, 2π/P, …, 2π(P−1)/P}，M ≥ d_{n−1} | `fourier_bases`：与 FR-INR `init_bases` 逐行一致（低频 (i+1)/L、高频 1…H、P 个相位、z_j 为 [−T_max/2, T_max/2] 上 d_{n−1} 个点、T_max = 2π/最低频，整体乘 α）；有逐元素对照测试 |
| Eq.(7) y = σ(ΛBy + b)，σ = sin(ω₀·) | 原 `IRN.forward` 不改，仅把隐藏层 `Linear` 换成 `FRLinear`，仍是 sin(30·(ΛBy + b)) |
| Fig.1 速度 = 背景 + FR-INR 扰动 | 现有 residual 参数化 `v = v_init + 1000·IRN`，末层零初始化 |
| §3.2：各层激活频率可训练；Fig.16 测 ω = 25/30/35 | 可选 `model.trainable_omega`：每个带激活的层一个可训练 ω，初值 `omega_0`，单独 Adam 学习率（见下文）；未设置时固定 ω₀ = 30 |

超参数取 FR-INR 2D 实验中 sine 网络的值：H = 128、L = 128、P = 32（M = 8192 ≥ 128）、α = 0.01（源码注释：relu 用 0.05，sin 用 0.01）。FR-IFWI 论文未给出这些数值。

## 与论文字面不同或论文未说明的地方

1. **只重参数化隐藏层（128→128，共 3 层）**。论文 Eq.(5) 写 n = 1, 2, …，Eq.(14) 的示例也重参数化了 2 维输入层。但按 FR-INR 的基构造，输入维为 2 时 z_j = ±T_max/2，每个基在这两点取值相同（数值差约 1e-11），B 的秩为 1，x 与 z 无法区分（`test_two_input_first_layer_would_be_rank_one`）。FR-INR 本身也只对隐藏层做 FR，首层与输出层保持普通 Linear。输出层保持普通层，也保证 residual 末层零初始化后严格从 v_init 出发。
2. **Λ 初始化**。论文未说明。`lambda_init: fr_inr`（主配置）沿用 FR-INR：第 i 列 U(±√(6/M)/‖b_i‖/ω₀)，偏置为 0。`lambda_init: pinv`（消融配置，我加的）令 Λ = W₀·pinv(B)（截断 rtol 1e-6），ΛB 等于同 seed 原 IRN 权重在 B 行空间上的投影：中间各列不变，首尾两列取平均（见第 5 条），偏置不变。两种初始化在 residual 下第 0 步速度都严格等于背景。
   - **2026-10-11 修正**：此前 pinv 未截断，128/128/32 下 float32 舍入使近零奇异值被放大，Λ 达 ~5e6、ΛB 相对误差约 17 倍，之前的 pinv 配置不可用；已修复并加测试。
3. **训练协议**完全沿用 baseline1000（49 炮、每次随机 8 炮、约 5 s 记录、seed 3、Adam lr 1e-4、1000 次更新），不是论文的 13 炮/1.9 s 设置（论文协议另见下文 paper13）；论文未给出优化器、学习率或调度。两份 FR 配置相对 baseline1000 只改 `experiment_name` 与新增 `model.fourier`（有测试保证）。
4. **参数量**：每个 FR 隐藏层 Λ 为 128×8192，三层约 3.1M 参数（baseline 约 5 万）。B 由配置确定，不写入检查点；Λ 与 Adam 状态使每个检查点约 40 MB，FD 正演仍占绝大部分显存与时间。
5. **首尾输入列绑定**。FR-INR 的所有频率都是最低频率的整数倍，而 z_j 的首尾点相差一个最低频周期，所以每个基在首尾两点取值相同，B 的第一列与最后一列相等，秩 ≤ d−1：每个 FR 层无法区分前一层第 1 个与最后 1 个神经元（128/128/32 有效秩 127；64/64/4 为 126）。high/low 在 128 点上也会混叠到同一组离散频率。基保持与 FR-INR 一致；秩检查以 float64 构造、相对容差 1e-6 计算有效秩，只允许这一维亏损（此前用 float32 buffer 求秩，舍入噪声掩盖了它）。

## 可训练 ω（`model.trainable_omega`）

论文 §3.2 写明各层激活频率可训练，但未给出参数化、初值或学习率。实现见 `experiments/trainable_omega.py`：原 `IRN` 有哈希校验不改，把每个带激活的层包成 `layer(x)·ω_l/ω₀`，IRN 外层再乘 ω₀，即 sin(ω_l·(Wx+b))；输出层无激活，不包。ω_l 初值 ω₀，第 0 步与固定 ω 网络逐位相同。ω 单独一个 Adam 参数组：若与网络共用 lr 1e-4，1000 步最多变化约 0.1，等于不训练。配置取 `learning_rate: 0.01`（1000 步最多约 ±10，覆盖 Fig.16 的 25–35），是我们的选择，需扫描。逐步 ω 记在 `history` 的 `omega_after` 和 `progress.log`。

```yaml
model:
  trainable_omega:
    learning_rate: 0.01
```

## 论文协议对照（paper13）

`paper13_ifwi.yaml` / `paper13_fr_ifwi.yaml` 按论文 §3.1：13 炮、炮距 300 m、源深 15 m、每个网格点一个接收器、8 Hz、dt 1.9 ms、1000 个采样点、每次更新用全部炮，1000 次更新（Table 1 平滑初始模型）。二者只差 `fourier` + `trainable_omega`（有测试保证），IFWI 侧固定 ω。论文未给出的项沿用 stage1：Adam lr 1e-4、高斯 σ = 15 格、4×128 网络、接收器深度索引 2。因此绝对数值仍不能直接对照论文 Table 1（论文 IFWI 平滑初始 1000 步速度 MSE 0.0299 (km/s)²，约 173 m/s RMSE），应比较两者的相对差。

## 运行（需在 GPU 机器上）

```bash
python -m pytest verification/residual_ifwi/test_fourier.py -q
python residual_ifwi.py --config experiments/configs/residual_fr_ifwi_baseline1000.yaml --dry-run
python residual_ifwi.py --config experiments/configs/residual_fr_ifwi_baseline1000.yaml --device cuda:0 --output-dir outputs/residual_fr_ifwi
python residual_ifwi.py --config experiments/configs/residual_fr_ifwi_pinv_baseline1000.yaml --device cuda:0 --output-dir outputs/residual_fr_ifwi_pinv
python residual_ifwi.py --config experiments/configs/residual_fr_ifwi_omega_baseline1000.yaml --device cuda:0 --output-dir outputs/residual_fr_ifwi_omega
python residual_ifwi.py --config experiments/configs/paper13_ifwi.yaml --device cuda:0 --output-dir outputs/paper13_ifwi
python residual_ifwi.py --config experiments/configs/paper13_fr_ifwi.yaml --device cuda:0 --output-dir outputs/paper13_fr_ifwi
```

建议先各跑 300 次更新（`--epochs 300`），对照 baseline 每 50 次的全炮评价曲线，再决定是否跑满 1000。续训、`--crop/--shots/--nt` 小规模检查与 baseline 用法相同，见 [Residual IFWI 文档](residual_ifwi.md)。

## 分频带误差诊断

背景已含真值约 0.8 c/km 以下的长波长，整体 RMSE 的改善几乎都来自中高波数。`experiments/spectral_metrics.py` 用正交 2D DCT-II 计算径向波数 k（cycles/km）各频带的速度 RMSE，各频带平方和严格等于总 RMSE²，并给出相对背景的比值（< 1 表示该频带改善）。默认频带：0–1、1–3、3–8、8–16、>16 c/km（15 m 网格 Nyquist 为 33.3 c/km）。

离线脚本只读取已保存的速度数组，不跑正演，可直接用于已有 baseline1000 输出。续训的一组用逗号连接其目录（先早后晚）：

```bash
python -m experiments.residual_band_report outputs/residual_ifwi_baseline1000/RUN_ID,outputs/residual_ifwi_baseline1000/RESUMED_RUN_ID outputs/residual_fr_ifwi/RUN_ID --labels baseline fr --output-dir outputs/band_report
```

输出 `band_errors.csv/json`、`band_ratio.png`，以及区域误差 `region_errors.csv/json`（`experiments/region_metrics.py`）：深度带 0–500、500–1000、>1000 m，和真值 ≥ 5000 m/s 的掩膜及其补集；每个区域给 RMSE 和占总平方误差的比例，终端打印背景与各组终点对照。下采样 Marmousi 中 ≥ 5000 m/s 恰为两条 5500 m/s 薄层（955 格、3.5%、深约 1.1–1.2 km，厚 4–5 格，远薄于 8 Hz 分辨率）；仅高斯背景的误差平方和中它们就占 42%，深于 1 km 的部分占 63%。

## 有效步长诊断

检验"FR 层在相同 Adam lr 下步长更大"（按符号步估计 128/128/32 约 4.7 倍、64/64/4 约 0.3 倍）。`experiments/residual_step_report.py` 读取 `checkpoints/update_XXXX.pt`，按保存的配置重建模型（FR 基确定性生成），合并 W = ΛB，计算相邻检查点之间每层预激活权重 A = ω·W 的位移 ‖ΔA‖_F（输出层无 ω），并在相同窗口上与第一组相除。这是净位移不是路径长度，来回振荡会使其偏小；各组同样口径。只读文件、不跑正演：

```bash
python -m experiments.residual_step_report outputs/residual_ifwi_baseline1000/RUN_ID,outputs/residual_ifwi_baseline1000/RESUMED_RUN_ID outputs/residual_fr_ifwi/RUN_ID outputs/residual_fr_ifwi_omega/RUN_ID --labels baseline fr128 fr128_omega --output-dir outputs/step_report
```

输出 `step_displacement.csv/json/png`，终端打印各层相对 baseline 的中位比值及 200 步以内窗口的隐藏层比值。只支持本仓库运行器写出的检查点；FR64/64/4 来自另一实现、配置键不同，不能直接读取。

## 未验证项

- 本机 venv 缺 pandas/PyYAML/torchvision，`test_fourier.py`、`test_trainable_omega.py`、`test_experiment.py` 只在测试专用替身模块下运行通过，需在完整环境复跑。
- FR-IFWI 论文的提速结论（Table 4，500 次约等于 IFWI 1000 次）是在 13 炮、1.9 s、线性初始模型上得到的；在 49 炮随机抽炮的 baseline1000 协议下是否成立未知。
- 在 Adam 下，FR 的效果很大程度上等于改变有效步长：按符号步估计，128/128/32 的 ΔW 约为普通层的 4.7 倍，64/64/4 约 0.3 倍；比较 FR 与 baseline 时应同时扫描学习率。
- α 在 Adam 下主要起有效步长作用（ΔW = ΔΛ·B 随 α 缩放），可能需要和学习率一起扫描。
