# 多尺度频率延拓：先拟合低频数据，再拟合全频带

本文件记录 2026-10-15 对 Muon 四组结果和 FR-IFWI 频谱诊断的复核，以及据此加入的改进：训练目标的多尺度频率延拓（Bunks et al., 1995）。2026-10-16 的 GPU 结果与随后加入的学习率安排见[第 6 节](#6-gpu-结果2026-10-16与学习率安排)：多尺度把 IFWI13 从 749.0 降到 397.0 m/s，但 FR13 从 375.3 变差到 404.0 m/s。

## 1. 复核结论

### 1.1 Muon（13 炮、seed 3、1000 次更新）

| 组 | 速度 RMSE（m/s） | SSIM | 波形 MSE | 数据残差能量 ‖r‖²/‖d‖² |
| --- | ---: | ---: | ---: | ---: |
| IFWI + Adam | 749.0 | 0.185 | 9.29e-3 | 2.0e-1 |
| IFWI + Muon | 519.9 | 0.325 | 7.35e-3 | 1.6e-1 |
| FR-IFWI + Adam | **375.3** | **0.662** | **2.31e-5** | **5.0e-4** |
| FR-IFWI + Muon | 470.9 | 0.469 | 4.16e-3 | 9.1e-2 |

- IFWI13 在 Muon 下从 749 回到 520（第 450 次左右 RMSE 达 675，之后持续下降，1000 次时仍在下降），与"普通 SIREN 隐藏层更新秩很低"一致，但 1000 次内没有低于初始模型。
- FR13 在 Muon 下明显变差。数据残差是 Adam 的 180 倍，说明它在数据空间也没有收敛，不是走到了数据约束的极限。按波数分带，FR + Muon 在 2.15–10 c/km 的误差（252、224 m/s）远大于 FR + Adam（155、164）；k ≥ 10 c/km 的误差为 278（Adam 264，初始 260），可以理解为额外加入了约 √(278² − 260²) ≈ 98 m/s 与真值无关的细尺度成分，对应图上的颗粒感。正交化把更新的所有奇异方向拉到同一量级（包括数据约束很弱的方向）；没有学习率衰减时，Muon 的更新幅度不随梯度减小，loss 来回振荡。这两点都在论文 §5 的局限之内。
- 结论：FR + Muon 不再继续。若要再试 Muon，只在 IFWI13 上加学习率衰减。

### 1.2 对 FR 频谱诊断的三点补充

用四组终态速度（`final_models` 结果包）和真值按同一频带阈值（0.833、2.147 c/km，偶延拓 FFT）复算：

**(a)"深层低频变差（R = 1.385）"主要来自薄层与背景的误差抵消，不是深层背景速度被反演坏了。** 下采样 Marmousi 在 1.0–1.2 km 深处有两条 5500 m/s 的薄层（4–5 格厚）。高斯初始模型在薄层内偏慢、在周围偏快，两者在低频带互相抵消。FR13 修正了周围，于是薄层的负误差在低频带显露出来。把误差拆成"薄层（v_true ≥ 5000 m/s）内"与"其余"两部分，分别取深层（≥ 915 m）低频带 RMSE：

| | 深层低频合计 | 仅薄层部分 | 仅其余部分 | 深层平均误差 | 其中薄层以外 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 初始模型 | 115.5 | 218.0 | 135.7 | −102.3 | +86.1 |
| FR13 + Adam | 160.0 | 151.0 | 43.6 | −131.8 | −2.6 |

两部分各自都下降了（其余部分 R ≈ 0.32），合计上升只是因为抵消消失了。深层薄层以外的平均误差从 +86 降到 −2.6 m/s。

**(b) FR13 终态 MSE 有一半落在 8 Hz 数据分辨不了的尺度上。** 按径向波数分解 FR13 终态误差（m/s）：

| 波数（c/km） | < 0.83 | 0.83–2.15 | 2.15–6 | 6–10 | ≥ 10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 初始模型 | 98.5 | 177.3 | 317.3 | 244.1 | 260.1 |
| FR13 + Adam | 101.1 | 101.0 | 154.5 | 163.7 | 264.2 |

k ≥ 10 c/km（波长 < 100 m）占终态 MSE 的 49.6%，与初始模型相同（初始模型在此带没有内容，这部分误差就是真值本身的细尺度结构）。作为参照，把真值低通到 k < 10 c/km 后与真值本身的 RMSE 就有 260 m/s（k < 8 时为 301，k < 13 时为 214）。因此"高频误差占 85%"大部分是数据分辨率的下限，换网络或优化器无法消除。真正还能改进的是 k < 10 c/km 的部分：FR13 已从 449 降到 267 m/s。

**(c) FR13 终态已经拟合了数据。** 在终态速度上做一次 13 炮正演：残差能量只占观测能量的 5.0 × 10⁻⁴，其中 83% 在 1.0 s 之后（深部反射）。此时物理梯度 −g 与剩余模型误差的余弦只有 +0.025。剩余误差大多是这些数据几乎看不见的成分，继续优化同一个目标收益有限；能改变结果的是反演路径（先恢复哪些尺度）或先验。

### 1.3 早期误差上升的原因：全频带梯度方向错误

你的诊断显示 0→100 次更新时三个频带的误差都上升，0→50 次的低频更新与 −e 的余弦为 −0.46。在 σ = 15 的初始模型上，对不同目标函数各做一次 13 炮 FD 正演与反传，计算 −g 与 v_true − v_init 的余弦（`gradient_report`；深度分区与你的报告相同：最深节点的 [0, 30 %)、[30 %, 65 %)、[65 %, 100 %]）：

| 目标函数 | 全网格 | 0–1 c/km | 1–3 c/km | 3–8 c/km | 浅层 | 中层 | 深层 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全频带（现有训练） | −0.059 | **−0.575** | −0.170 | −0.093 | −0.120 | −0.333 | −0.271 |
| 低通 ≤ 6 Hz | +0.011 | +0.490 | −0.149 | +0.018 | +0.097 | +0.001 | −0.212 |
| 低通 ≤ 4 Hz | +0.022 | +0.586 | +0.063 | +0.008 | +0.098 | +0.063 | −0.067 |
| 低通 ≤ 3 Hz | +0.014 | **+0.583** | +0.067 | +0.001 | +0.060 | +0.024 | **+0.034** |

全频带梯度在长波长上指向远离真值的方向，这与高频成分的周波跳跃一致：只保留 ≤ 3 Hz 的数据后，长波长余弦变为 +0.58，各深度分区都转为正值。另试了只给晚到时间加权、增强深部（残差乘 t 或 t²，不在代码中）：0–1 c/km 的余弦仍为 −0.55 和 −0.48，方向没有被修正。这一结果支持多尺度频率延拓，不支持深部加权。余弦只说明第一步的方向，能否转化为更好的终态要由训练确认。

IFWI13（Adam）的终点仍停在这种状态：其全频带梯度在 0–1 c/km 的余弦仍为 −0.56，0–4 Hz 的数据残差仍有该频带能量的 17%（FR13 终点为 0.06%），也就是低频数据始终没有被拟合，符合周波跳跃造成的局部极小。

## 2. 实现

| 文件 | 内容 |
| --- | --- |
| `experiments/frequency_continuation.py` | 配置校验；按更新序号返回当前截止频率；零相位低通；各阶段能量归一化系数 |
| `experiments/residual_ifwi_experiment.py` | 可选 `training.frequency_schedule`；低通阶段用该阶段目标反传，`data_mse` 仍记录同一次正演的全频带波形 MSE |
| `experiments/gradient_report.py` | 任意保存状态（初始模型、`.npy`、运行目录下全部检查点）的物理速度梯度：余弦（分波数带、分深度）、梯度能量的深度分布、数据残差按时间与频带分解 |

```yaml
training:
  frequency_schedule:
    - {max_hz: 3.0, updates: 100}   # 第 1–100 次
    - {max_hz: 5.0, updates: 100}   # 第 101–200 次
    - {max_hz: null}                # 其余更新：全频带
```

- **滤波**：对预测与观测记录做同一零相位低通，幅度响应 1/(1 + (f/f_c)¹²)，即 6 阶 Butterworth 正反向滤波，f_c 处 −6 dB，不产生时移。时间轴补零到 2 nt，避免循环卷积回绕。正演、子波和观测都不改；FD 对子波是线性的，滤波记录与滤波子波等价。
- **能量归一化**：低通阶段的 MSE 乘以 E_全/E_低通（全部观测记录滤波前后的能量比，每阶段一个常数）。≤ 3 Hz 的记录能量不到全频带的 1%，不归一化时梯度会骤降约两个数量级，Adam 的 eps 开始起作用，换阶段时梯度又骤升。归一化后目标等于"该频带的相对残差 × 全频带数据能量"，各阶段量级可比。
- **优化器状态跨阶段保留**，不重置。重置 Adam 会让每个阶段的前几步都变成满幅度的符号步；对 FR 来说，这正是早期大步长的来源。
- **约束**：截止频率严格递增且低于时间 Nyquist；最后一个阶段必须是全频带（`max_hz: null`）。续训时更新序号决定阶段，可逐位续训；换了 schedule 的续训会被拒绝。未设置时与原代码逐位相同：在提交 6a5390a 与本改动之间，用 paper13_ifwi、paper13_fr_ifwi、paper13_fr_ifwi_muon、paper13_lr_fr_ifwi、residual_ifwi_baseline1000 五份配置的小规模运行核对了最终速度、逐步历史和 comparison_notes。
- **日志**：`history` 中低通阶段的条目多出 `max_hz` 与 `objective`，`frequency_schedule.json` 记录各阶段起止与归一化系数，`progress.log` 标出阶段切换。

配置（相对基础配置只多了 `training.frequency_schedule`，有测试保证）：

- `paper13_fr_ifwi_multiscale.yaml`：FR13 + 多尺度，对照 FR13（375.3 m/s）。
- `paper13_ifwi_multiscale.yaml`：IFWI13 + 多尺度，对照 IFWI13（749.0 m/s），用于检验机制：13 炮下普通 IFWI 的失败有多少来自周波跳跃（第 3 节的 CPU 结果显示还有参数化的原因）。

3 Hz 起步依据上表；5 Hz 作为过渡。阶段长度 100/100/800 依据第 3 节：100 次后 3 Hz 目标已下降 94–98%（不用真值的判据），此时 5 Hz 与全频带梯度也已指向真值。这只是一次 CPU 检查后的选择，没有调参；最初写的是 200/200/600。

旧的原版 IFWI 参数实验框架里已有一个频带延拓（`experiments/improved_modules/spatiotemporal.py`，`st_multiscale.yaml`：100 次更新内 5 → 8 Hz → 全频带，响应 exp(−½(f/f_c)⁸)，无能量归一化），但它没有接入残差训练器，也没有在 13 炮协议上跑过。按上表，σ = 15 时 5–6 Hz 的梯度在 1–3 c/km 和深部仍然方向错误，所以这里从 3 Hz 起步。

## 3. CPU 前 100 次更新（第 1 阶段，≤ 3 Hz）

在本机 CPU 上用多尺度配置各跑了 100 次更新（全部处于 ≤ 3 Hz 阶段，与 100/100/800 的第一阶段相同），与 GPU 上已有基线运行的同一时刻比较。频带与比值 R 沿用你报告的阈值（0.833、2.147 c/km，相对高斯背景；R < 1 为改善）。CPU 与 GPU 的浮点细节不同，这里只看趋势：

| 运行 | 更新次数 | RMSE（m/s） | 低频 R | 中频 R | 高频 R | 全频带波形 MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FR13（基线，GPU） | 50 | 614.2 | 1.586 | 1.232 | 1.157 | 0.0148 |
| FR13 + 多尺度（CPU） | 50 | **519.0** | **0.793** | 1.064 | 0.999 | 0.0293 |
| FR13（基线，GPU） | 100 | 620.3 | 1.702 | 1.195 | 1.169 | 0.0132 |
| FR13 + 多尺度（CPU） | 100 | **520.7** | **0.846** | 1.038 | 1.005 | 0.0270 |
| IFWI13（基线，GPU） | 50 | 553.4 | 1.507 | 1.017 | 1.051 | 0.0196 |
| IFWI13 + 多尺度（CPU） | 50 | 523.5 | 1.553 | 0.918 | 0.992 | 0.0205 |
| IFWI13（基线，GPU） | 100 | 594.0 | 1.698 | 1.154 | 1.114 | 0.0155 |
| IFWI13 + 多尺度（CPU） | 100 | **510.3** | 1.325 | 0.874 | 0.981 | 0.0159 |

- **FR13**：早期的大幅上升消失了（100 次时 521 vs 620）。低频误差下降（R 0.79–0.85，基线为 1.59–1.70），中、高频基本不动。全频带波形 MSE 下降得比基线慢，这是预期的：此阶段只拟合 ≤ 3 Hz。
- **IFWI13**：RMSE 低于初始模型（510），中、高频改善，但低频仍比背景差（50 次 R 1.55，100 次 1.33）。按深度看，普通 SIREN 把浅部需要的减速几乎原样加到了所有深度（100 次时浅层平均 −114、深层 −83 m/s），而 FR 的修正主要在浅部（−70 m/s），深部按另一方向变化（+88 m/s）。这与普通 SIREN 末层输入特征的稳定秩只有约 1.6（FR 约 16）一致：它能独立调整的空间模式太少，数据约束只在浅部时，深部被一并带动。当时据此推断"多尺度未必能单独救回 IFWI13"；**GPU 结果否定了这一推断**（第 6 节：1000 次后 397.0 m/s），前 100 次的低频偏差在后续阶段被纠正。
- **第 100 次时的梯度方向**（`gradient_report`）：两个运行在 5 Hz 和全频带目标下，0–1 c/km 余弦都已为正（FR：+0.12 / +0.15；IFWI：+0.12 / +0.05），说明已离开第 0 步那种方向错误的区域。FR 在 3 Hz 目标下 0–1 c/km 余弦已变为 −0.15，同时低频 R 从 50 次的 0.79 回升到 0.85。此时 0–4 Hz 的数据残差只剩该频带能量的 1.4%，继续留在 3 Hz 收益不大，因此阶段长度改为 100/100/800。

这些只是前 100 次的趋势；全频带阶段能否保住低频改善，并在 1000 次内低于 375 m/s，要看 GPU 运行。

## 4. GPU 实验与命令

1. `paper13_fr_ifwi_multiscale` 对 FR13（375.3）：唯一变量是训练目标的频率延拓。
2. `paper13_ifwi_multiscale` 对 IFWI13（749.0）：机制对照。
3. 用 `gradient_report` 在已有 FR13、IFWI13 运行的全部检查点上计算物理梯度（你报告第 7 节第 3 条所需的 g_v），并对多尺度运行做同样的检查。
4. 多尺度有收益时，再做阶段长度、起始频率的敏感性和多种子。

```bash
python residual_ifwi.py --config experiments/configs/paper13_fr_ifwi_multiscale.yaml --device cuda:0 --output-dir outputs/paper13_fr_ifwi_multiscale
python residual_ifwi.py --config experiments/configs/paper13_ifwi_multiscale.yaml --device cuda:0 --output-dir outputs/paper13_ifwi_multiscale
python -m experiments.gradient_report --states initial outputs/paper13_fr_ifwi/RUN --config experiments/configs/paper13_fr_ifwi.yaml --max-hz 3 5 full --device cuda:0 --output-dir outputs/gradient_report/fr13
```

`gradient_report` 每个状态、每个目标各做一次全炮正演与反传（13 炮在 GPU 上每次约 1–2 s，在 CPU 上约 20 s），输出 `gradient_report.csv`（状态 × 目标的余弦与梯度能量分布）和 `residual_report.csv`（各状态的数据残差按记录四等分时段、按 f₀ 倍数频带分解）。省略 `--config` 时使用运行目录自带的 `config.json`。

评价沿用速度 RMSE / SSIM 和全频带波形 MSE（多尺度运行的 `data_mse` 与其他运行同一口径），并用分频带误差把"可分辨部分（k < 10 c/km）"与分辨率下限分开看。

## 5. 测试

```bash
python -m pytest verification/residual_ifwi/test_frequency_continuation.py verification/residual_ifwi/test_lr_schedule.py -q
```

覆盖：低通零相位、通带保持、阻带抑制、可微，补零后无回绕；阶段映射与配置校验；能量归一化使零预测的低通目标等于全频带 MSE；多尺度运行按阶段记录 `max_hz`/`objective`，`data_mse` 与全频带运行逐位相同，跨阶段续训逐位一致，换 schedule 续训被拒绝；`gradient_report` 的目标值与训练记录一致；两份配置相对基础配置只多 schedule。

`test_lr_schedule.py` 覆盖：学习率因子（每阶段 warmup、余弦衰减、衰减结束后保持）；配置校验；运行器按因子设置每个参数组（网络与 ω）、`history` 记录 `lr_factor`、跨阶段续训逐位一致、改动 schedule 的续训被拒绝；未设置时不记录、不缩放；四份配置相对基础配置只改学习率安排与 `report_estimates`（短阶段配置另改阶段长度）。

## 6. GPU 结果（2026-10-16）与学习率安排

### 6.1 结果（13 炮、seed 3、1000 次更新，各自对照全频带基线）

| 运行 | 速度 RMSE（m/s） | SSIM | 全频带波形 MSE |
| --- | ---: | ---: | ---: |
| FR13（全频带基线） | **375.3** | **0.662** | 2.31e-5 |
| FR13 + 多尺度 | 404.0 | 0.588 | 9.36e-5 |
| IFWI13（全频带基线） | 749.0 | 0.185 | 9.29e-3 |
| IFWI13 + 多尺度 | **397.0** | **0.650** | 8.26e-5 |

IFWI13 被救回，曲线平滑，支持 1.3 节"全频带梯度方向错误（周波跳跃）"的判断；第 3 节"多尺度未必能单独救回 IFWI13"的推断是错的。FR13 反而变差，原因如下。

### 6.2 FR13 多尺度为什么变差

**1. 一次晚期尖峰。** 第 671–700 次波形 MSE 涨到约 430 倍，RMSE 从 400.7 升到 478.7，到第 1000 次只回到 404.0（尖峰前 401.5）。按"超过前 20 个状态最小值 5 倍"检出的尖峰：

| 运行 | 尖峰起点：MSE 倍数 |
| --- | --- |
| FR13 基线 | 391：5.3；520：14；702：25；921：30 |
| FR13 + 多尺度 | 333：5.2；427：7.9；538：11；**671：427**；702：17；886：9.3 |
| IFWI13 + 多尺度 | 645：8.6；872：15 |

尖峰周期出现、间隔和幅度逐次增大。这是固定学习率下 Adam 接近极小值时的典型现象（Cohen et al., 2022, *Adaptive Gradient Methods at the Edge of Stability*）：Adam 每步的参数位移约等于 lr，不随梯度变小而变小，会周期性越过尖锐方向的稳定步长；β₂ = 0.999 的二阶矩估计在平稳期缓慢变化，使下一次越界来得更晚、更猛。按尖峰前的下降速度（第 600→680 次约 −0.04 m/s/次）外推，没有这次尖峰时第 1000 次约 390 m/s，即尖峰约占与基线差距的一半。

**2. 尖峰之前已经落后。**

| 更新次数 | 300 | 400 | 500 | 600 | 680 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FR13 基线 RMSE | 466.2 | 425.7 | 408.8 | 398.6 | 391.1 |
| FR13 + 多尺度 RMSE | 429.2 | 416.6 | 409.7 | 404.5 | 401.5 |

前 450 次多尺度领先，之后下降变慢；SSIM 从第 400 次起一直低于基线（400 次 0.580 vs 0.595，600 次 0.594 vs 0.632），图上是细颗粒纹理。

**3. 阶段切换冲击。** 第 201 次切到全频带时两组 MSE 都立即跳升（FR RMSE 480 → 约 500，IFWI 476 → 约 488），约 50 次后恢复：全频带目标更尖锐、梯度更大，而 Adam 状态还是按 5 Hz 阶段调好的。

**4. 低通阶段的细尺度注入。** σ = 15 初始模型处的 3 Hz 梯度，97% 的能量在震源所在的第 1 行（15 m 深），91% 在 k > 8 c/km（全频带梯度为 78% 与 74%），这是震源印记。网络和 Adam 把它变成低通数据看不见、之后也纠正不了的细尺度结构，FR 受影响更大：CPU 上 3 Hz 阶段 100 次后，速度改变量在 k ≥ 4 c/km 的 RMS，FR 为 121 m/s，IFWI 为 64 m/s（k < 2.15 c/km 部分分别为 181、156）。GPU 快照中 FR 第 100 次已有斑块状纹理，IFWI 平滑；FR 的 3/5 Hz 目标曲线有尖峰，IFWI 平滑下降。FR 拟合 3 Hz 数据很快：CPU 上低频 R 在第 50 次最好（0.79），第 100 次回升到 0.85，此时 3 Hz 梯度在 0–1 c/km 的余弦已是 −0.15。

### 6.3 试过、未采用

| 做法 | CPU 结果 | 结论 |
| --- | --- | --- |
| 低通阶段所有学习率 ×0.2（另有 20 次 warmup） | 第 47 次 3 Hz 目标 0.0044（未改 0.0008）；固定梯度下更新的高/低波数 RMS 比 0.56（未改 0.66） | 只是整体变慢，更新的频谱几乎不变 |
| 低通阶段梯度高斯平滑，σ = 1500/(4·f) = 125/75 m（另有 20 次 warmup） | 第 100 次 RMSE 545.2（未平滑 520.7），低频 R 1.36（0.85），3 Hz 目标 0.0020（0.0002）；k ≥ 10 c/km 注入 +1 m/s（+4） | 抑制了细尺度注入，但把震源行的大幅值摊到深处，3 Hz 拟合停滞 |
| 屏蔽前 3 行（震源、检波器）梯度（固定梯度近似） | 更新的高/低波数 RMS 比 0.71（未屏蔽 0.66） | 细尺度主要来自网络 + Adam 的响应，不只来自震源行 |

这三种做法没有留在代码中。

### 6.4 改进方向与代码

验证分工：本机只跑单元测试（每项数秒），训练对比全部在 GPU 上做。下表按证据强弱排序，每一项都是配置开关，未设置时与原代码逐位相同（已在提交 f59d519 与本改动之间，用五份配置的小规模运行核对最终速度、逐步历史和 comparison_notes）。

| # | 方向 | 针对的现象（证据） | 代码 / 配置键 |
| --- | --- | --- | --- |
| 1 | 学习率安排：每个频率阶段开头 warmup，全频带阶段余弦衰减 | 尖峰周期出现且越来越大；FR 多尺度第 671 次一次损失约 75 m/s；第 201 次切换冲击 | `experiments/lr_schedule.py`，`training.lr_schedule` |
| 2 | FR 的低通阶段缩短（50/50 次） | FR 拟合 3 Hz 很快，第 50 次低频最好，之后注入细尺度；CPU 上短阶段在第 96 次到达 GPU 原多尺度第 200 次的状态（RMSE 481 vs 480，波形 MSE 0.0116 vs 0.0109），全频带阶段多出 100 次 | 配置：阶段长度 |
| 3 | 普通 SIREN + 多尺度这条路 | IFWI13 + 多尺度已到 397.0 m/s，与 FR 基线差 22 m/s，且曲线平滑 | 配置 |
| 4 | Adam β₂ = 0.99 | 尖峰来自二阶矩估计跟不上步长变化；β₂ 小则反应快 | `training.optimizer: {name: adam, betas: [0.9, 0.99]}` |
| 5 | FR 基的尺度 α：0.01 → 0.005 | Adam 下 FR 隐藏层步长约为普通 SIREN 的 6 倍，且与 α 成正比；初始权重与 α 无关（已核对逐位相同） | 配置：`model.fourier.alpha` |
| 6 | 晚到时间增益 (t/t_end)^p（从第 301 次起） | FR13 终点残差 83% 在 1.0 s 之后，晚到部分的相对残差是早到的 20–30 倍，而普通 MSE 由早到强能量主导 | `experiments/training_options.py`，`training.time_gain` |
| 7 | EMA 与最低数据残差状态（不改训练，只多输出） | 晚期尖峰和细尺度振荡；原 IFWI 代码也按 loss 选模型 | `training.report_estimates`，输出 `final_velocity_ema.npy`、`lowest_misfit_velocity.npy` 与 `metrics.json` 的 `estimates` |

没有采用：第 6.3 节的三种做法；全变差（TV）正则——真值的 TV（0.214 km/s/格）高于 FR13 终态（0.141），TV 惩罚只会让模型更平滑，与真值差得更远，无噪声合成数据下没有依据；改 ω 学习率——49 炮上 FR 与 FR + ω 只差 0.3 m/s。

`report_estimates` 中，EMA 从 `ema_start` 起对每次更新后的速度做指数平均；最低残差状态在每次更新前的全炮评价与最终评价中取最小（随机抽炮时只用全炮评价）。两者都不用真值，主指标仍是最后一次迭代（`final`）。

### 6.5 GPU 实验

所有新配置都带 `report_estimates`（EMA 衰减 0.99，从第 501 次起），每次运行同时给出最后迭代、EMA 与最低残差三个结果。

第一轮（四次运行）：

| 配置 | 相对基础配置的改动 | 对照 | 回答的问题 |
| --- | --- | --- | --- |
| `paper13_fr_ifwi_lrsched` | FR13 + 学习率安排（warmup 20，第 201–1000 次衰减到 0.1） | FR13（375.3） | 学习率安排对全频带 FR 的作用 |
| `paper13_fr_ifwi_multiscale_lrsched` | FR13 多尺度 + 同一学习率安排 | FR13 多尺度（404.0）；与上一行比较多尺度的作用 | 学习率安排能否消除晚期尖峰与切换冲击 |
| `paper13_fr_ifwi_multiscale_short_lrsched` | 低通阶段 50/50 次 + 学习率安排（第 101–1000 次衰减） | 上一行 | 缩短低通阶段的作用 |
| `paper13_ifwi_multiscale_lrsched` | IFWI13 多尺度 + 学习率安排 | IFWI13 多尺度（397.0） | 普通 SIREN 路线能走多远 |

第二轮（各只改一项，基于 `paper13_fr_ifwi_lrsched`；若第一轮最好的是别的配置，把同一项加到它上面）：

| 配置 | 改动 |
| --- | --- |
| `paper13_fr_ifwi_lrsched_beta2` | Adam β = (0.9, 0.99) |
| `paper13_fr_ifwi_lrsched_alpha005` | FR α = 0.005 |
| `paper13_fr_ifwi_lrsched_tgain` | 第 301 次起残差乘 t/t_end |

单种子下几 m/s 的差别不足以下结论；选出的组合再做 2–3 个种子。

```bash
python residual_ifwi.py --config experiments/configs/paper13_fr_ifwi_lrsched.yaml --device cuda:0 --output-dir outputs/round1/fr_lrsched
python residual_ifwi.py --config experiments/configs/paper13_fr_ifwi_multiscale_lrsched.yaml --device cuda:0 --output-dir outputs/round1/fr_ms_lrsched
python residual_ifwi.py --config experiments/configs/paper13_fr_ifwi_multiscale_short_lrsched.yaml --device cuda:0 --output-dir outputs/round1/fr_ms_short_lrsched
python residual_ifwi.py --config experiments/configs/paper13_ifwi_multiscale_lrsched.yaml --device cuda:0 --output-dir outputs/round1/ifwi_ms_lrsched
python residual_ifwi.py --config experiments/configs/paper13_fr_ifwi_lrsched_beta2.yaml --device cuda:0 --output-dir outputs/round2/fr_lrsched_beta2
python residual_ifwi.py --config experiments/configs/paper13_fr_ifwi_lrsched_alpha005.yaml --device cuda:0 --output-dir outputs/round2/fr_lrsched_alpha005
python residual_ifwi.py --config experiments/configs/paper13_fr_ifwi_lrsched_tgain.yaml --device cuda:0 --output-dir outputs/round2/fr_lrsched_tgain
```

回传时除终态指标和曲线外，请附上 `metrics.json`（含 `estimates`）和 `history.json`：后者有每次更新的 `lr_factor`、`objective`、`gradient_norm`、`omega_after`，可以确认尖峰是否消失、发生在哪一层面。

### 6.6 测试

```bash
python -m pytest verification/residual_ifwi/test_lr_schedule.py verification/residual_ifwi/test_training_options.py -q
```

覆盖：学习率因子（每阶段 warmup、余弦衰减）及其在每个参数组上的应用、续训逐位一致；Adam betas/eps 的校验与构造；时间增益的权重、能量归一化、与频率阶段组合、开始前与普通运行逐位相同、续训一致；EMA 与最低残差状态的数值（与逐次快照手算一致）、不改变训练、续训恢复；七份新配置相对基础配置只改各自那一项。
