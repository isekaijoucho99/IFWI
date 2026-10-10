# FR-IFWI：在 Residual IFWI 上加入 Fourier 重参数化

对应论文：Kang, Chen, Yang, Li & Wu, *Implicit full waveform inversion with adaptive Fourier frequency bases learning*, GJI 244 (2025), ggaf404（下称 FR-IFWI）。论文未公开代码；Fourier 基的具体构造沿用其方法来源 Shi et al. (CVPR 2024) 的公开实现 [FR-INR](https://github.com/LabShuHangGU/FR-INR)（`modules.py::sin_fr_layer`、`2d_image_fitting.py`）。

**本分支的代码没有运行过任何训练**，下表各项均为按论文与 FR-INR 源码逐项核对的设计，FR-IFWI 在本协议上的效果尚未验证。

## 与论文的对应关系

| 论文 | 实现 |
| --- | --- |
| Eq.(5) W⁽ⁿ⁾ = Λ⁽ⁿ⁾B⁽ⁿ⁾，Λ 可训练、B 固定 | `experiments/fourier_modules.py::FRLinear`，`weight = lamb @ bases`，`bases` 为不可训练 buffer |
| Eq.(6) b_ij = cos(ω_i z_j + φ_i)，φ ∈ {0, 2π/P, …, 2π(P−1)/P}，M ≥ d_{n−1} | `fourier_bases`：与 FR-INR `init_bases` 逐行一致（低频 (i+1)/L、高频 1…H、P 个相位、z_j 为 [−T_max/2, T_max/2] 上 d_{n−1} 个点、T_max = 2π/最低频，整体乘 α）；有逐元素对照测试 |
| Eq.(7) y = σ(ΛBy + b)，σ = sin(ω₀·) | 原 `IRN.forward` 不改，仅把隐藏层 `Linear` 换成 `FRLinear`，仍是 sin(30·(ΛBy + b)) |
| Fig.1 速度 = 背景 + FR-INR 扰动 | 现有 residual 参数化 `v = v_init + 1000·IRN`，末层零初始化 |
| ω₀ = 30（Fig.16 另测 25/35） | 沿用 baseline `omega_0: 30.0` |

超参数取 FR-INR 2D 实验中 sine 网络的值：H = 128、L = 128、P = 32（M = 8192 ≥ 128）、α = 0.01（源码注释：relu 用 0.05，sin 用 0.01）。FR-IFWI 论文未给出这些数值。

## 与论文字面不同或论文未说明的地方

1. **只重参数化隐藏层（128→128，共 3 层）**。论文 Eq.(5) 写 n = 1, 2, …，Eq.(14) 的示例也重参数化了 2 维输入层。但按 FR-INR 的基构造，输入维为 2 时 z_j = ±T_max/2，每个基在这两点取值相同（数值差约 1e-11），B 的秩为 1，x 与 z 无法区分（`test_two_input_first_layer_would_be_rank_one`）。FR-INR 本身也只对隐藏层做 FR，首层与输出层保持普通 Linear。输出层保持普通层，也保证 residual 末层零初始化后严格从 v_init 出发。
2. **Λ 初始化**。论文未说明。`lambda_init: fr_inr`（主配置）沿用 FR-INR：第 i 列 U(±√(6/M)/‖b_i‖/ω₀)，偏置为 0。`lambda_init: pinv`（消融配置，我加的）令 Λ = W₀·pinv(B)，使 ΛB 在浮点误差内等于同 seed 原 IRN 的权重、偏置不变，此时与 baseline 的唯一差别是训练动力学（重参数化等价于对 W 的梯度做 BᵀB 预条件）。两种初始化在 residual 下第 0 步速度都严格等于背景。
3. **训练协议**完全沿用 baseline1000（49 炮、每次随机 8 炮、约 5 s 记录、seed 3、Adam lr 1e-4、1000 次更新），不是论文的 13 炮/1.9 s 设置；论文的学习率未给出。两份 FR 配置相对 baseline1000 只改 `experiment_name` 与新增 `model.fourier`（有测试保证）。
4. **参数量**：每个 FR 隐藏层 Λ 为 128×8192，三层约 3.1M 参数（baseline 约 5 万）。B 由配置确定，不写入检查点；Λ 与 Adam 状态使每个检查点约 40 MB，FD 正演仍占绝大部分显存与时间。

## 运行（需在 GPU 机器上）

```bash
python -m pytest verification/residual_ifwi/test_fourier.py -q
python residual_ifwi.py --config experiments/configs/residual_fr_ifwi_baseline1000.yaml --dry-run
python residual_ifwi.py --config experiments/configs/residual_fr_ifwi_baseline1000.yaml --device cuda:0 --output-dir outputs/residual_fr_ifwi
python residual_ifwi.py --config experiments/configs/residual_fr_ifwi_pinv_baseline1000.yaml --device cuda:0 --output-dir outputs/residual_fr_ifwi_pinv
```

建议先各跑 300 次更新（`--epochs 300`），对照 baseline 每 50 次的全炮评价曲线，再决定是否跑满 1000。续训、`--crop/--shots/--nt` 小规模检查与 baseline 用法相同，见 [Residual IFWI 文档](residual_ifwi.md)。

## 分频带误差诊断

背景已含真值约 0.8 c/km 以下的长波长，整体 RMSE 的改善几乎都来自中高波数。`experiments/spectral_metrics.py` 用正交 2D DCT-II 计算径向波数 k（cycles/km）各频带的速度 RMSE，各频带平方和严格等于总 RMSE²，并给出相对背景的比值（< 1 表示该频带改善）。默认频带：0–1、1–3、3–8、8–16、>16 c/km（15 m 网格 Nyquist 为 33.3 c/km）。

离线脚本只读取已保存的速度数组，不跑正演，可直接用于已有 baseline1000 输出：

```bash
python -m experiments.residual_band_report outputs/residual_ifwi_baseline1000/RUN_ID outputs/residual_fr_ifwi/RUN_ID --labels baseline fr --output-dir outputs/band_report
```

输出 `band_errors.csv/json` 与 `band_ratio.png`。

## 未验证项

- 本机未运行测试和任何训练；`test_fourier.py` 需在有 PyTorch 的环境中运行。
- FR-IFWI 论文的提速结论（Table 4，500 次约等于 IFWI 1000 次）是在 13 炮、无背景的 IFWI 上得到的；在有高斯背景的 residual 协议、Adam 下是否成立未知。
- α 在 Adam 下主要起有效步长作用（ΔW = ΔΛ·B 随 α 缩放），可能需要和学习率一起扫描。
