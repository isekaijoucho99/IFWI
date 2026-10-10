# Residual FR-IFWI：实现与本地训练

本分支基于 `isekaijoucho99/IFWI` 的 `9b0bdf47999e5bb51baacfb90dddc7c3224ae3f9`，在已有残差 IFWI 中加入 Fourier 权重重参数化。原始 `ifwi_modules.py`、`rnn_fd.py`、`generator.py`、`plot_functions.py` 和 baseline1000 配置未修改。不要将这里的 CPU 验证当成完整 GPU 反演效果或收敛性验证。

## 论文对应关系与边界

核心实现见 `experiments/fr_siren.py`。它实现 Kang 等人的 FR-IFWI 公式 (5)–(7)：

\[
W^{(l)}=\Lambda^{(l)}B^{(l)},\qquad
B_{ij}=\alpha\cos(\omega_i z_j+\phi_i),\qquad
y^{(l)}=\sin\!\left(\omega_0(W^{(l)}y^{(l-1)}+b^{(l)})\right).
\]

这里 `FourierLinear` 只做线性映射，Sine 由外层 IRN 施加一次。`B` 注册为固定 buffer，可训练参数是 `coeff`（即 Λ）和 bias。它不是坐标位置编码、速度场 FFT 或地震数据频带滤波。

GJI 论文没有完整列出基数量、采样区间、系数初始化等可执行细节。本实现的这些部分依据其引用的 Shi 等人 CVPR 2024 工作及公开 FR-INR `sin_fr_layer` 实现：

- 低频集合为 `1/L, 2/L, ..., 1`，高频集合为 `1, 2, ..., H`，保留频率 1 的重复。
- 每个频率配 `P` 个相位 `2πp/P`。在最低基频的一个完整周期内均匀取点，包含两端点。
- 系数第 i 列独立初始化为 `U(-a_i,a_i)`，其中 `a_i=sqrt(6/M)/(||B_i||₂ × omega_0)`。FR 隐藏层 bias 初始化为零。
- 保留原 SIREN 首层、输出层，只替换中间 `128→128` 的三层。最后线性层仍清零，确保新建残差模型严格从固定背景开始。

默认 `L=64,H=64,P=4,alpha=0.01` 是本项目已讨论的本地实验设置，不是声称找到了 GJI 作者未公开的默认超参数。原 SIREN 隐藏层与 FR 隐藏层的初始化方式也不同，因此主实验检验的是这套 FR 参数化与初始化方案，而不是在完全相同隐藏权重上的纯坐标变换。

GJI §3.2 另提到各层激活频率可训练。本分支提供独立的 `residual_ifwi_fr1000_learnable_omega.yaml`，每个带激活层增加一个可训练 ω（含首层），初值 30。不对它做门控、调度或正数重参数化。首轮 `residual_ifwi_fr1000.yaml` 固定 ω=30，便于先观察 Fourier 本身的贡献。两种配置均不代表作者所有实验的逐参数完整复现。

参考来源：

1. Kang et al. *Implicit full waveform inversion with adaptive Fourier frequency bases learning*. GJI. https://doi.org/10.1093/gji/ggaf404
2. Shi et al. *Improved Implicit Neural Representation with Fourier Reparameterized Training*. CVPR 2024, pp. 25985–25994. https://openaccess.thecvf.com/content/CVPR2024/html/Shi_Improved_Implicit_Neural_Representation_with_Fourier_Reparameterized_Training_CVPR_2024_paper.html
3. 官方 FR-INR 参考实现：https://github.com/CVL-UESTC/FR-INR/blob/main/modules.py 。本文件按公式独立组织实现，没有引入其图像数据集、LPIPS 或训练框架依赖。

## 固定的对照协议

`experiments/configs/residual_ifwi_fr1000.yaml` 只在 baseline1000 上改变实验名称并增加网络设置：

```yaml
model:
  neurons: [2, 128, 128, 128, 128, 1]
  omega_0: 30.0
  mean_kmps: 3.0
  std_kmps: 1.0
  architecture: fr_siren
  fourier:
    low_freq_num: 64
    high_freq_num: 64
    phi_num: 4
    alpha: 0.01
    target_layers: hidden_only
    learnable_omega: false
```

仍采用 94×288 Marmousi、15 m 网格、采样后 Gaussian σ=15、49 炮、每步无放回抽 8 炮、4 炮微批次、dt=0.0019 s、nt=2632、8 Hz Ricker、原二阶 FD/PML/自由表面、普通波形 MSE、Adam lr=1e-4 和 1000 次参数更新。每 50 次更新进行全炮评价并保存。两次微批次 backward 后只执行一次 Adam 更新，不做时间分段，不加 TV、Attention、扩散、裁剪、学习率调度或速度截断。

物理速度输出仍是 `fixed_init_mps + 1000 * std_kmps * raw`，残差分支不加 `mean_kmps`。NumPy 抽炮随机状态不被 Fourier 初始化消耗。相同种子的各组仍应核对日志里的 `shot_indices`。

默认可训练参数为 197,505，原 4×128 SIREN 为 50,049。可学习 ω 版本为 197,509。后续参数量匹配对照可另用普通 4×256 SIREN，但不要把它混入首轮。

## 一个不能忽略的基矩阵细节

默认每层 `B.shape=(512,128)`，并不代表有 512 个独立方向。在保留参考实现的端点采样和相位重复后，以浮点容差计算得到数值秩 126。代码会记录 SVD、数值秩、容差、有效条件数和基矩阵 SHA256，不会擅自加 DC 基、改成正交 DCT 或删除重复相位。

因此不能声称该 B 满列秩，或任意 128×128 权重都能被精确重参数化。`condition_number=null` 表示不满足满列秩，不是 JSON 缺失。遇到近零范数的退化基行则直接拒绝初始化，防止系数除以近零量。自定义小宽度或频率范围时先跑下面的无正演检查。

## 在本地运行

使用你已经能运行 IFWI 的 Python/CUDA 环境，不必为这个模块重装 GPU PyTorch。若尚未安装测试依赖，只需 `python -m pip install pytest`。

在仓库根目录执行，无正演配置和初始化检查：

```bash
python residual_ifwi.py --config experiments/configs/residual_ifwi_fr1000.yaml --dry-run
python scripts/check_fr_ifwi.py --output outputs/fr_preflight.json
```

运行 CPU 自动检查。此命令仅在测试进程中隐藏 CUDA、使用单线程并禁用 oneDNN：

```bash
python verification/fr_ifwi/run_cpu_checks.py
```

理由是原测试包含不同炮批量正演结果的逐位相等断言，而 PyTorch 2.10 默认 oneDNN 的浮点累加顺序可能不同。生产训练代码的后端开关没有改变。新增 Fourier 测试也应在默认 CPU 后端运行：

```bash
python -m pytest verification/fr_ifwi/test_fourier.py -q
```

独立的小网格真实 FD 验证（24×64、5 炮中抽 3 炮、100 个时间点、3 次更新）：

```bash
python residual_ifwi.py --config experiments/configs/residual_ifwi_fr_smoke.yaml --device cpu --output-dir outputs/fr_smoke
```

它只验证运行链路，裁剪后平滑的背景不等于完整背景的同位置裁片，不能比较其 RMSE 与正式 Marmousi 结果。

正式 GPU 训练：

```bash
python -u residual_ifwi.py --config experiments/configs/residual_ifwi_fr1000.yaml --device cuda:0 --output-dir outputs/residual_ifwi_fr1000
```

也可以先把正式配置运行到 50 次更新，检查显存和数值稳定性：

```bash
python -u residual_ifwi.py --config experiments/configs/residual_ifwi_fr1000.yaml --epochs 50 --device cuda:0 --output-dir outputs/residual_ifwi_fr_probe
```

随后使用程序打印的实际运行目录恢复到累计 1000 次，不能直接使用原 SIREN 的 checkpoint。下面 `RUN_ID` 必须替换成实际目录名：

```bash
python -u residual_ifwi.py --config experiments/configs/residual_ifwi_fr1000.yaml --epochs 1000 --device cuda:0 --resume outputs/residual_ifwi_fr_probe/RUN_ID/checkpoint.pt --output-dir outputs/residual_ifwi_fr1000_resumed
```

恢复会检查架构、Fourier 设置、数据来源、原核心文件及 FR 核心/适配器 SHA256。不要在同一次续训链中编辑这两份实现文件。PyTorch、设备或计算后端变化不保证逐位一致。若发生 CFL/非有限值错误，原训练器会回滚该次更新并保存失败信息；本实现不靠隐藏裁剪来继续训练。

独立可学习激活频率实验使用 `experiments/configs/residual_ifwi_fr1000_learnable_omega.yaml`。不要拿它恢复固定 ω 的 checkpoint，也不要将两组结果混为一组。

## 输出及评价

沿用现有输出 `config.json`、`initial_velocity.npy`、`final_velocity.npy`、`metrics.json`、`sampled_loss.csv`、`full_evaluation_history.json`、`checkpoint.pt` 以及每 50 次的快照。新增 `fourier_diagnostics.json` 记录启动/恢复时的基矩阵检查，`experiment_source_hashes.json` 包含新模块。固定 B、系数、可选 ω、Adam 和随机数状态都进入 checkpoint。

用固定 500、800、1000 次**更新后**全炮评价比较，不要把抽炮的更新前 loss 与全炮更新后 loss 混用。主指标仍是全模型 RMSE/MAE/SSIM、全炮波形 MSE、用时和显存。深部误差与低通误差可从保存的速度数组离线计算。参数梯度范数属于不同参数空间，不能直接把 FR 更大的梯度范数解释为更强的物理更新。

历史 `verification/residual_ifwi/audit_stochastic_run.py` 是硬编码原 49 炮 SIREN 协议的专用审计，不是 FR 审计。对应的 `run_residual_ifwi_to_convergence.py` 也冻结该专用审计及文件列表，本轮不支持用它控制 FR。请使用上面的 `residual_ifwi.py --resume`。现有绘图输出和数值数组仍可使用。

1000 次是指定预算，不是最优或收敛证明。平滑初速度含合成真值的低波数信息，也不是独立现场先验。当前交付只验证实现与兼容性，不承诺 GPU 完整反演精度、稳定性、显存或加速倍数。
