# G²IFWI 复现（分支 `g2ifwi-repro`）

对应论文：*G²IFWI: Gradient-Coupled Diffusion Priors for Regularizing Implicit Full Waveform Inversion*。
新代码全部在 `g2ifwi/`，旧的 Marmousi/SIREN 脚本（根目录 `main.py`、`ifwi_modules.py`、`rnn_fd.py` 等）未改动，仅作历史参考。

## 与论文的对应关系

| 论文 | 文件 |
|---|---|
| 多分辨率 hash 编码 Eq.(2)-(6)，Table 1 | `g2ifwi/hashgrid.py` |
| v = v_init + MLP(hash(r))，SIREN ω=30，2×128 | `g2ifwi/inr.py` |
| 声波正演 + 伴随梯度 Eq.(8)-(10) | `g2ifwi/forward.py`（梯度检查点，等价于离散伴随梯度，有有限差分单元测试） |
| 程序化速度模型数据集（1D 背景+水层、弹性形变、折叠、侵入体）Fig.2 | `g2ifwi/velocity_gen.py` |
| DDPM、U-Net、Table 2、RED、井引导 Eq.(11)-(35) | `g2ifwi/diffusion.py` |
| 先验训练 | `g2ifwi/train_prior.py` |
| Algorithm 1、Overthrust 实验、噪声/初始模型敏感性、RMSE/SSIM/PSNR | `g2ifwi/invert.py` |

核心耦合在 `invert.py::epoch`：`g_total = g_data + λ·(v − Sampling(v))`，再用 `v.backward(g_total)` 回传到 hash 表和 MLP。

## 在另一台电脑（建议 CUDA GPU）上运行

```bash
pip install torch scikit-image scipy matplotlib numpy pytest
python -m pytest -q tests                      # 先跑单元测试（含梯度检验，几秒）

# 1) 训练扩散先验（Table 2 配置，A100 级别；约 250k step）
python -m g2ifwi.train_prior --preset paper --steps 250000 --batch 16 --out weights/prior_paper.pt
#    显存/时间不够时用小模型：
python -m g2ifwi.train_prior --preset small --steps 30000 --out weights/prior_small.pt

# 2) Overthrust：IFWI 与 G²IFWI 共享前 100 epoch，之后分叉（Fig.3/4）
python -m g2ifwi.invert --prior weights/prior_paper.pt --out outputs/overthrust

# 3) 噪声鲁棒性（Sec 5.2）、较差初始模型（Sec 5.3）
python -m g2ifwi.invert --prior weights/prior_paper.pt --snr 0  --out outputs/snr0
python -m g2ifwi.invert --prior weights/prior_paper.pt --sigma 50 --alpha 0.025 --out outputs/smooth50
python -m g2ifwi.invert --prior weights/prior_paper.pt --init 1d --out outputs/init1d
```

输出：`metrics.json`（RMSE/SSIM/PSNR）、`models.png`、`curves.png`、`history.json`、`models.npz`。
显存不够时调小 `--shot-batch`（默认 8）或增大 `--chunk` 的反面（减小 `--chunk`）；快速试跑可用 `--stride 2`（空间抽稀）、`--shots`、`--nt`。

## 需要你留意的偏差和未验证项

1. **λ 用 α 自动标定（默认）**。论文 λ=7×10⁻⁴ 依赖其梯度的量纲，在本实现里不可直接迁移；论文 5.1 也说明最优对应 α=‖λg_ddpm‖/‖g_data‖≈2.5%。因此默认 `--alpha 0.025`，每次注入按当前梯度范数换算 λ；也可用 `--lam` 固定。
2. **先验的噪声水平 `--t-start/--n-steps` 论文没给具体值**。Overthrust 写的是"最低噪声水平一步反向"，我默认 `t=1, 1 步`；BP 为 10 步。这个值对结果可能影响很大，请在你机器上扫一下（例如 t-start∈{1,5,10,20}）。
3. **归一化**：每个模型 min-max 到 [-1,1]（训练与推理一致），因此 Overthrust 的 6000 m/s 超出训练范围 [1500,4500] 也能输入。论文只说"normalized"。
4. **井引导符号**：论文 Eq.(30) 为 `μ + γ g_well`，字面上会远离井数据；实现里用 `μ − γ g_well`（下降方向）。`red_denoise(well=(W,M), gamma=…)` 已实现并有形状测试，但没有 Viking 数据，未做实验。
5. **网格间距**默认 25 m（论文），仓库里的 `overthrust2d.npz` 标注 20 m，可用 `--dz 20`。正演与观测数据同算子生成（inverse crime，论文同样合成数据）。边界为 Cerjan 海绵 + 顶部自由面，不是旧代码的 PML。
6. hash 表初始化 1e-4、网络输出缩放 1000 m/s、末层零初始化（保证从 v_init 出发）是我的选择，论文未给。我在 4×抽稀的小问题上只跑了 80 epoch：流程能跑通、数据误差下降，但模型 RMSE 尚未下降，hash 初始化在 1e-4～1e-1 之间差别不大——**论文那种 200 epoch 内收敛是否复现，还需要在全尺寸上验证**。
7. **未实现**：BP2004 与 Viking 实验（仓库没有数据）、λ 扫描图、5.3 的高通滤波缺低频实验（`AcousticModeler.loss_and_grad` 已留 `transform` 钩子）。
8. 扩散先验我没有训练（本机内存不足），所以 G²IFWI 的真实效果、论文指标（RMSE 423.73→310.12 等）都**没有验证**。

## Marmousi（论文没有这个实验）

```bash
python -m g2ifwi.invert --model marmousi --prior weights/prior_paper.pt --out outputs/marmousi
```

`--model marmousi` 使用旧代码的采集设置：`data/vel_marmousi_376x1151.csv` 抽稀为 94×288，dz=15 m，dt=1.9 ms×1000 步，8 Hz Ricker，每 20 格一炮（共 13 炮），检波器每格一个、深度索引 2。
其余（hash/SIREN、lr=1e-4、200 epoch、第 100 epoch 启用先验、α=2.5%、初始模型为 σ=15 的高斯平滑）**直接沿用论文 Overthrust 的设置，不是论文为 Marmousi 给的**；Marmousi 含水层与陡构造，扩散先验的 `--t-start` 和 `--alpha` 很可能需要重新调。
先验是在 1500–4500 m/s 的合成层状模型上训练的，Marmousi 上限约 5500 m/s，推理时按单模型 min-max 归一化处理。

## 配置核对：哪些来自论文，哪些是我自选

- 论文给出并已照做：hash 4 级/64→256/1 特征/T=2¹⁸；SIREN ω=30、2 层×128；Adam lr=1e-4；200 epoch；先验第 100 epoch 后每 epoch 一次；U-Net base 128、mult [1,2,4,8,16]、3 个残差块、4 头、1000 步、Adam lr 1e-6、batch 16、100 epoch×40000 crop（`--preset paper --steps 250000`）；8 Hz Ricker、2 ms、6 s；Overthrust 初始模型为 σ=15 的高斯平滑。
- 论文没给、我自选：线性 β 调度（1e-4→0.02）、attention 所在分辨率、海绵边界、hash 素数与初始化、输出缩放、末层零初始化、`t-start`、α 自动标定。
- 与论文字面不同：炮点沿全宽均匀 40 炮（论文写"10 m 间隔"与 10 km 宽度矛盾）；先验训练集为无限在线采样而非固定 40000 个 crop。
