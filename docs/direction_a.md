# 方向 A · Adam 的历史动量如何改变 IFWI 的实际速度更新

给定同一个物理速度模型，改变网络输出尺度后，Adam 会沿什么方向移动速度场？加入历史动量能把实际更新预测得更准吗？若一次更新可能使波形失配上升，检查和重算的成本是否值得？本实验用仓库的时域波动方程和残差网络逐步测量这三个问题。

## 复现

使用项目依赖及支持 `torch.func.functional_call` 的 PyTorch 2.x。以下命令在仓库根目录执行。

```bash
python -m experiments.direction_a
python -m pytest -q verification/direction_a
```

默认在两个模型上分别取居中的 16 × 24 窗口，使用 2 炮、110 个时间样点和 18 Hz 的 Ricker 子波。每个模型建立一个共享检查点，分别运行 SIREN 和 FR、输出尺度 1 和 4、7 种优化策略，以及两种学习率设置。每条轨迹最多更新 5 次。

缩小范围可以单独看某种现象。例如在较大学习率下检查动量重启和回溯：

```bash
python -m experiments.direction_a --datasets marmousi --representations fr --scales 4 --methods adam adam_armijo adam_guarded_no_remainder adam_guarded --lr-modes native --lr .008 --steps 12
```

全网格入口使用 Marmousi 每 4 点抽样后的完整网格或原始 Overthrust 网格。以下命令给出一组较长时间窗的运行设置。可通过 `--shot-batch` 降低每次反传的显存，各炮次组按全部炮数归一化。

```bash
python -m experiments.direction_a --full-grid --device cuda:0 --shots 13 --shot-batch 1 --nt 1000 --frequency 8 --steps 200 --width 128 --representations fr --scales 1 --methods adam adam_clipped adam_armijo adam_guarded --lr-modes matched --save-steps 1 20 100 200
```

## 公平起点与正演

- Marmousi 使用仓库配套的真值和 smooth400 初模，按已有 CSV 约定跳过首行并沿两个轴每 4 点采样，间距 15 m。Overthrust 使用 NPZ 中的 `vp`、`vi`，从 km/s 换算为 m/s，间距 20 m。时间步长分别为 1.5 ms 和 1.8 ms。模拟使用二阶差分、自由表面和侧底 15 格吸收层。
- 共享检查点默认由残差 FR 从零输出头预跑一步 Adam 得到。可以用 `--warmup-steps 0` 回到背景模型，或指定更晚的共同起点。复制时将 FR 隐藏层折叠为等价稠密权重得到 SIREN。对每个输出尺度 $s$，将网络输出系数乘 $s$，将最后一层权重和偏置同除以 $s$。因此已有非零扰动也会保留。程序检查 double 网络速度差异及实际送入 float32 正演的速度是否逐点相同。
- 各条轨迹从共同函数出发，优化器状态重新置零。两种表示的物理输出和隐藏宽度相同，FR 的系数个数和参数几何不同；报告记录了参数量。网络与更新诊断采用 float64，原有波动正演和伴随采用 float32。
- 每次更新重新计算当前模型的全炮波形均方差及物理梯度 $g=\nabla_v\Phi$。地质真值只用于合成观测和事后评估速度 RMSE。控制器能看到观测数据、当前速度和优化器状态。PML 参考速度由网格与时间步长确定。

## 单步分解

设 $h_t=J_t^\top g_t$，$D_t=(\sqrt{\hat v_t}+\epsilon)^{-1}$ 为 Adam 偏差校正二阶矩给出的逐参数因子。用 $m_{t-1}$ 表示上一步已接受的一阶矩，$t$ 从 1 开始。一阶矩未重启时，程序计算

$$
\Delta\theta_{\rm current}=-\eta D_t\frac{(1-\beta_1)h_t}{1-\beta_1^t},
\qquad
\Delta\theta_{\rm history}=-\eta D_t\frac{\beta_1m_{t-1}}{1-\beta_1^t}.
$$

历史矩已经累积了各个以往模型上的 $J_s^\top g_s$。用当前 Jacobian 对两部分做 JVP，就能得到它们对本次速度更新的局部预测，无需存储全部 Jacobian。梯度裁剪组将 $h_t$ 换成实际送入优化器的裁剪梯度；重启组重新计算一阶矩的偏差校正年龄，保留二阶矩及其年龄。

对每个接受步，实际更新为 $\Delta v_{\rm actual}=f(\theta+\alpha\Delta\theta)-f(\theta)$，其中 $\alpha$ 是本次步长系数。输出报告比较以下量。

| 记录 | 含义 |
|---|---|
| `linear`、`current`、`history` | 完整参数步及其当前项和历史项在同一个更新前 Jacobian 下的投影，均计入实际步长系数。 |
| `instant` | 保持该步 $D_t$，仅用当前梯度给出的 $-\alpha\eta J_tD_t h_t$。可用它检验“历史信息是否提高对这一步的预测”。各自独立训练的无动量 Adam 另作轨迹对照。 |
| `sgd_kernel` | 同一学习率下的 $-\alpha\eta J_tJ_t^\top g_t$。报告也计算预测方向与实际方向的余弦、匹配 RMS 后的相对误差和最佳非负标量拟合误差。标量拟合使用已观察到的更新，只作事后比较。 |
| `linear_relative_error` | $\|\Delta v_{\rm actual}-J_t\alpha\Delta\theta\|/\|\Delta v_{\rm actual}\|$，测量有限步非线性。 |
| `angle_to_descent_deg`、`history_gradient_dot` | 实际速度变化与 $-g_t$ 的夹角，以及 $g_t^\top\Delta v_{\rm history}$。后者为正表示历史分量单独看会一阶增大失配。 |
| `band_rms_mps` | 实际更新、线性预测、历史项、非线性余项在模型空间的径向 DCT 波数带 RMS，单位 m/s，波数单位 cycles/km。子波频率以 Hz 另行记录。 |
| `misfit_change`、`misfit_taylor_remainder` | 真实正演失配变化及其与 $g_t^\top\Delta v_{\rm actual}$ 的差，可用来区分波形目标曲率和网络映射的非线性。 |

## 策略和成本

`sgd`、`adam_no_momentum`（$\beta_1=0$）、`adam` 直接沿各自更新。`adam_clipped` 提供普通参数梯度裁剪对照。`native` 学习率在各输出尺度间保持不变，SGD 与 Adam 的默认值分别为 0.003 与 0.001。`matched` 用第一次已计算的物理梯度校准各自学习率，使初始线性预测步 RMS 同为 5 m/s，随后固定学习率。该校准只需网络 JVP。

`adam_armijo` 提供参数方向上的普通 Armijo 回溯。`adam_guarded` 先检查 $g_t^\top J_t\Delta\theta$；含历史的候选步没有线性下降时，试算重启一阶矩。随后约束实际速度步 RMS、相对网络线性化误差和 $g_t^\top\Delta v_{\rm actual}$。通过后，用额外的全炮正演检查 Armijo 条件，当前和试探失配都采用同一前向求值模式。拒绝时按二分回溯，只有接受后才提交网络参数和动量状态。`adam_guarded_no_remainder` 保留相同重启、步幅和 Armijo 规则，只关闭线性化余项门槛。所有阈值可通过 `--help` 查看。

每次运行在 `outputs/direction_a/` 下建立独立结果目录：

- `summary.csv` 每条轨迹一行，列出第二步起历史预测的胜出次数、误差中位数、失配上升次数及首次达到共同质量阈值时的成本。
- `report.json` 记录输入设置、初始函数误差、每次候选步的诊断、回溯、接受情况及成本。数据 MSE 和速度 RMSE 的默认目标分别为共同检查点初值的 80% 和 99%。阈值只用于记录首次到达的更新，运行预算内未达到时对应字段为 `null`。
- 默认保留第 1 步及最后一个预算步被接受时的 `step_*.npz`，包含更新前后速度、物理梯度、真实更新和五张预测图。`--save-steps` 可指定其他步，传空列表可关闭数组输出。
- `forward_shots` 和 `adjoint_shots` 以单炮为单位计在线成本，覆盖失败尝试以及回溯正演。`report_forward_shots` 单列事后评估用的正演；若控制器已经评估了接受步，则复用该值。`total_*` 加上每条轨迹分摊一次的共同准备、诊断和事后评分成本；时间计量在 CUDA 上包含同步。不同轨迹共享准备工作的实际执行，因此不要直接累加各行的 `total_*` 作为整次扫描的实耗。

上述 JVP 分解是局部一阶预测。相同阈值下的炮数与耗时，以及裁剪、学习率匹配和回溯对照，决定了控制带来的收益。理论背景见 [Chen 等的 wave-based NTK 工作](https://arxiv.org/abs/2603.22362) 和 [Kingma 与 Ba 的 Adam 原文](https://arxiv.org/abs/1412.6980)。

## CPU 小网格实测

默认配置在两种介质上共运行 112 条轨迹，完成 560 步。下表只统计其中的标准 Adam 组。每种介质有 8 条标准 Adam 轨迹，跳过一阶矩尚无历史的首步后，剩下 32 步。误差为预测更新与实际更新之差的范数除以实际更新范数。

| 介质 | 含历史的 JVP 相对误差中位数 | 只用当前梯度且保留同一二阶矩 | 含历史预测误差更低的步数 |
|---|---:|---:|---:|
| Marmousi | 1.42% | 5.27% | 27 / 32 |
| Overthrust | 1.51% | 6.31% | 28 / 32 |

在 FR、尺度 4、原生学习率 0.008 的 12 步对照中，普通 Adam 在两种介质上分别有 3 步和 2 步使失配上升。完整控制策略在两种介质上各接受 12 步，失配逐步下降。在 Marmousi 上，普通 Adam 首步就把失配降到初值的 80%，控制组用到第 8 步，总成本分别为 10 正演 + 4 伴随炮次和 38 正演 + 18 伴随炮次。在 Overthrust 上普通 Adam 同样首步达标，控制组在预算内未达标。关闭余项门槛的控制组取得相同的达标结果。本次数据支持历史项对单步预测的作用，尚未显示余项门槛带来额外收益。

另外验证了 Marmousi 94 × 288 和 Overthrust 94 × 401 的全网格入口。每种介质分别用普通 Adam 和完整控制策略运行 2 步，采用 1 炮、110 个时间样点。这项检查验证了全网格的梯度和参数更新，长期反演质量仍需另行测量。
