# 方向 B · 面向 IFWI 更新方向的正演与伴随误差控制

这个小网格实验回答两个问题。给定残差 SIREN 的候选更新方向，当前求解精度是否足以判定它是下降方向？计入每次额外求解和检查后，达到同一数据失配需要多少时间和计算量？

## 运行

在仓库根目录按主 README 安装依赖，使用支持 `torch.func.functional_call` 的 PyTorch 2.x。两个实验都可以在 CPU 上运行。

```bash
python -m experiments.direction_b
python -m experiments.direction_b --mode benchmark
python -m pytest -q verification/direction_b
```

第一条命令在两种介质上各做 4 次实际网络参数更新，在更新前、更新 2 次后、更新 4 次后分别审计。它在每个检查点固定同一个候选方向，对固定的正演、伴随容差取笛卡尔积，并与三种自适应策略比较。可用下面的命令改变测量规模。

```bash
python -m experiments.direction_b --datasets marmousi --shape 20 32 --shots 3 --frequencies 5 8 10 --checkpoints 0 3 6 --sweep-tols .03 .003 .0003 .00003 --output outputs/direction_b/marmousi.json
```

第二条命令让四种策略从相同的初始速度、权重和观测出发，分别生成自己的更新轨迹。默认预算是 8 次更新，质量阈值为初始数据失配的 80%。可用 `--max-updates`、`--target-ratio` 和 `--fixed-tol` 调整。高精度参考求解只用于合成观测和事后评分。

## 方程与上界的适用范围

在以米计的等距网格上，令 $m=v^{-2}$、$c=(2\pi f h)^2$。程序使用无量纲五点离散算子

$$
A=L-c\,\mathrm{diag}(m)-i\,\mathrm{diag}(\gamma),\qquad
\Phi=\tfrac12\sum_{f,s}\|R u_{fs}-d_{fs}\|^2.
$$

$L$ 为负拉普拉斯矩阵，网格外取零。每道源是单位点源，$R$ 选取互异的接收点。默认网格为 $16\times24$，使用两炮、5 Hz 和 8 Hz。Marmousi 的真值与 smooth400 初模采用仓库原有 CSV 读法，跳过首行、双轴每 4 点采样，使用 15 m 网格。Overthrust 使用 NPZ 的 `vp`、`vi` 和 20 m 网格，把 km/s 换算为 m/s。两种介质均居中裁剪，网络采用局部坐标和仓库的残差 SIREN，输出层零初始化。

设 $d_b$ 是节点至最近网格边缘的网格步数。吸收系数为 $\gamma=\gamma_0+a\max(1-d_b/3,0)^2$。默认 `--damping .04`、`--sponge .12`。对任意单位向量 $x$，$\|Ax\|\ge|\mathrm{Im}(x^*Ax)|\ge\gamma_0$。因此正演和伴随都可使用解析上界 $K=1/\gamma_0$，即使边缘吸收让算子非正规。设 `--sponge 0` 可运行均匀阻尼对照。改变阻尼也会改变正演物理，跨方法比较需固定它。推广到 PML 或零阻尼方程需重新建立逆算子范数上界。

## 如何判定一个实际更新方向

候选方向 $p$ 取近似参数梯度的单位负方向。PyTorch 计算 $\delta m=J_{m,\theta}p$，包含 $dm/dv=-2/v^3$。对每个频率和炮，GMRES 分别求解 $Au=q$ 与 $A^*\lambda=R^*(Ru-d)$，随后显式计算

$$
r_f=q-A\tilde u,\qquad
r_a=R^*(R\tilde u-d)-A^*\tilde\lambda,\qquad S=-c\,\mathrm{diag}(\delta m).
$$

近似方向导数为 $\tilde D=-\mathrm{Re}(\tilde\lambda^*S\tilde u)$。记 $E_u=K\|r_f\|$，$E_\lambda=K(\|r_a\|+E_u)$。后者的第二项来自正演误差对伴随右端的影响。将真波场写成近似波场加误差，分别对三个误差内积使用 Cauchy–Schwarz 不等式，可得方向相关上界

$$
B=E_\lambda\|S\tilde u\|+E_u\|S^*\tilde\lambda\|+
E_\lambda E_u\|S\|.
$$

程序同时记录附件中的全局范数界。只有 $\sum\tilde D+\sum B<0$ 才允许双残差策略沿这个 $p$ 更新。先求所有炮频的有符号总导数，可以处理不同贡献之间的抵消。这个判据针对当前离散目标的一阶导数，使用双精度数值计算。实际步长将线性化的最大速度改变量缩放至 `--step-mps`（默认 25 m/s），有限步的真实失配另行评分。

| 方法 | 正演与伴随精度如何决定 |
|---|---|
| `fixed` | 分别使用给定容差。基准轨迹每个炮频只做一次正演和一次伴随。 |
| `sloppiness` | 每个炮频独立将正演容差减半，温启动重算，直到相邻两次失配变化不超过新失配的 $\eta$ 倍，或到达下限。伴随采用最终正演容差。 |
| `dual` | 对冻结的 $p$ 比较上界中由正演和伴随残差产生的两部分，每轮将贡献较大一侧的容差收紧十倍。正演变化时同步更新伴随右端。 |
| `dual_corrected` | 每次检查先做一次复数 Jacobi 残差校正，按正演场、伴随右端、伴随场的顺序进行，再用校正后的场重算方向导数与误差界。 |

`dual_corrected` 用对角逆分别作用在正演残差和更新后的伴随残差上，得到场校正 $z_f,z_a$。方向导数的改变量为 $-\mathrm{Re}(\tilde\lambda^*S z_f+z_a^*S\tilde u+z_a^*S z_f)$。同一精度下的校正前后导数都写入审计报告，可以直接检验校正是否减小实际误差。

`sloppiness` 对应 van Leeuwen 和 Herrmann 第 3.2 节的失配稳定准则。这里采用已知单位点源、固定炮集与 GMRES。比较针对求解精度规则。神经预条件器的训练和反演适配属于后续求解器实验。[Controlled Sloppiness 原文](https://slim.gatech.edu/Publications/Public/Journals/SIAMJournalOnScientificComputing/2014/vanLeeuwen2014SISC3Dfds/vanLeeuwen2014SISC3Dfds.pdf) · [Goren 与 Treister 的频域 FWI](https://arxiv.org/html/2405.17696v1)

## 阅读输出

默认结果写入 `outputs/direction_b/audit.json` 和 `outputs/direction_b/benchmark.json`。两者均记录完整运行参数。

- 在审计报告中，`comparisons` 包含每一对容差的方向导数、LU 参考导数、符号错误、参数梯度夹角、方向界、全局界及实际误差。`bound_to_error` 接近 1 表示界较紧，实际误差为零时该比值记为 `null`。检查固定方向时，基线也计算诊断量。
- `solve_matvecs`、`check_matvecs` 和 `correction_matvecs` 分别统计迭代求解、显式残差检查和校正使用的矩阵向量乘。`online_matvecs`、`online_seconds` 含生成候选方向的成本。两种双残差策略复用候选方向已有的波场。求解时间、检查时间、校正时间和高精度评分时间单列。
- 在基准报告中，`first_crossing` 给出各方法首次达到阈值时的更新数、累计在线耗时和矩阵向量乘次数。返回 `null` 表示本次预算内没有达到阈值。应在同一设备、同一介质和同一质量阈值下比较这些数。基线轨迹自行生成方向，免除方向认证开销。
- 双残差策略若耗尽精度预算或判定方向为上升方向，将记录拒绝原因并停止。已认证方向仍可能因有限步过大而增加失配。该实验只检验小网格、合成观测下的离散目标，实测加速需看同等质量下的时间和 PDE 工作量。
- `case_checks` 按频率、炮次顺序记录实际使用的正演与伴随残差范数。GMRES 未收敛的尝试会记录已消耗的求解工作和诊断信息。该次尝试无法生成可用于比较的梯度。

## 默认算例的结果

CPU、Python 3.12、SciPy 1.18.1、PyTorch 2.14.1 下，两个模型共得到 6 个检查点、72 组容差或策略比较。72 个方向导数误差均落在计算的界内。两种双残差策略在全部检查点均给出下降判定，LU 参考解验证了这些方向的符号。

下表记录首次将真实数据失配降至初值 80% 时的累计矩阵向量乘次数，包括产生候选方向和检查残差的开销。

| 介质 | 固定容差 0.003 | 失配自适应 | 双残差 | 双残差加校正 |
|---|---:|---:|---:|---:|
| Marmousi | 718 | 1878 | 1603 | 1667 |
| Overthrust | 984 | 1670 | 2252 | 2348 |

在这两个算例上，固定容差达到相同质量的 PDE 工作量最低。带校正的双残差策略在 Marmousi 上花费约 2.32 倍的矩阵向量乘，在 Overthrust 上约为 2.39 倍。每次运行的墙钟时间可从基准报告中的 `first_crossing.online_seconds` 读取。
