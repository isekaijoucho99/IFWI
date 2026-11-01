# 原方法与 IFWI 离散适配的数学边界

## 文献与复现范围

Hawkins, Khalid, Schlottbom, Smetana, *Model Order Reduction for Seismic
Applications*, SIAM J. Sci. Comput. 47(5), B1045–B1076 (2025),
[10.1137/24M1667737](https://doi.org/10.1137/24M1667737)。
作者代码：[Zenodo 15033159](https://doi.org/10.5281/zenodo.15033159)，
[GitHub v_2](https://github.com/mhamzakhalid/Model_order_reduction_for_Seismic_Applications/tree/v_2)。

原方法不是神经网络代理：对二维弹性有限元方程做 Laplace 变换，在感兴趣的有限
频带采样解，以能量内积 POD / POD-greedy 构造空间，投影质量与刚度矩阵，最后用
Weeks 方法恢复接收记录。低频限制是取得小维数的重要条件。

本次实际运行作者 Figure_8_bc_Plot.m 的固定材料 POD_s + Weeks 实验及四个额外
完整 FEM 频率查询；没有声称复现需要多日计算的所有多材料 POD-greedy 图。
Python Weeks、Ricker-Laplace 和 Galerkin 核心也与真实 MATLAB 导出逐项比较。
作者核心文件未修改，封装脚本在私有副本中运行。

## 已发表证明适用于什么

| 性质 | 原文范围 | 本项目适配的状态 |
|---|---|---|
| Laplace 变分问题的稳定性 | Proposition 3.1；Re(s)>0、给定频带、材料与边界假设 | 带离散 PML 的非对称算子不自动满足这些假设 |
| 指数逼近 / n-width 衰减 | Proposition 4.1；固定材料、感兴趣频带 | 不等于任意 Marmousi 速度轨迹存在统一小维空间 |
| 后验误差控制及时间域估计 | Proposition 4.2；残差与有效稳定常数，以及逆变换误差控制 | 本实现未取得大规模 PML 的认证稳定常数 |
| 反演梯度精度 | 原文不保证 IFWI 的速度梯度或网络梯度 | 必须直接比较原 FDTD 梯度，已设置准入门槛 |
| 长时间非线性反演收敛 | 不由固定模型的频域逼近定理推出 | 短试更新也不能证明 1000 轮收敛 |

## 严格保留原离散方程

本项目原求解器是声学二阶 FDTD + PML，不是文中的弹性 FEM。
因此没有直接用连续 Helmholtz 方程代替它。对原离散递推消去 PML 辅助状态，
设 U(z)=sum_n u_n z^(-n)，得到压力 Schur 补：

```
L(z,v) = ((z-1)^2/z) M(v) + dt (z-1) M_delta(v) - L_dt
         + sum_l z/(z-alpha_l) P_l
L(z,v) U(z) = diag(1+dt*delta) M(v) E F(z)
Y(z) = z C U(z)
M(v) = diag(v_padded^-2)
M_delta(v) = diag(delta * v_padded^-2)
P_(d,l) = dt^3 (diag(b_d) D_d + diag(c_d)) diag(beta_d mask_l) D_d
```

这里 c_d 必须采用原 conv2d 计算的边缘值；v_padded 是原始复制填充；源项在原程序
中于阻尼除法之后注入，因而 RHS 中的速度和阻尼系数不能省略；Y 前的 z 来自
原程序记录更新后的压力。完整空间下这是代数消元，不是另一套 PDE 近似。
CPU 全基测试比较了双精度离散算子、原 FDTD 的压力、记录和速度导数。

固定实基 Q 后，使用 L_r=Q^T L Q、b_r=Q^T b、C_r=CQ。Q 与预投影量均是
register_buffer，没有训练参数。在线重新装配 Q^T M(v) Q 与 Q^T M_delta(v) Q，
对小型复矩阵求解并做逆 FFT；PyTorch 自动微分保留质量、源项及复制填充的导数。
基冻结时计算的是冻结基 ROM 的精确导数，不包含重建 Q(v) 的导数。

在线投影是两次 O(N r²) 的大网格投影，不在每个频率重做。小型频域求解约为
O(n_frequency r³ + n_frequency r² n_shot)，记录恢复约为 FFT 成本。
因此不能仅从全网格自由度 / r 的比值推算实际加速比。

## 联合正演与伴随空间

只用正演快照可能把波形拟合很好，却遗漏残差反传的伴随场。因此额外构造
L(z)^(-H) C^T conj(z) FFT[(prediction_background-observed) exp(+eta*n)]
的实、虚快照，与正演快照联合 POD。输入仅为背景模型和已有观测，不使用测试
真速度的波场。两个快照组按总能量相等加权；这是明确选择的启发式，并非论文中
已证明最优的目标导向 POD。省略的 FFT 端点标量不改变单个快照跨度，但会影响
POD 权重，因此不能称其为严格的平衡截断。

第二种联合构造先保留指定维数的正演 POD 空间，再对其能量正交补中的伴随快照
做 POD。它保留已有正演空间的跨度；这只是一条子空间包含性质，不证明非对称
Galerkin 解误差必然单调下降。该性质有独立测试，两个构造的实际结果均保留。

背景校准正演是每个新炮集的准备成本，应单列；若以后在反演途中刷新基，刷新
属于在线成本。现有原实验用真模型最大速度固定 PML；两个求解器共用该参数，
这一基线设定不应被描述为完全不依赖真模型的部署方案。

## 哪些误差界可以写，哪些尚未认证

对每个可逆离散频率算子，若 beta <= sigma_min(L) 且 beta>0，则

```
||u-u_r|| <= ||b-L u_r|| / beta.
```

这是条件成立时的标准残差界。仅报告残差但不给出可信 beta，不能称为认证误差界。
对整个频率块系统 A，需统一 beta <= sigma_min(A)=min_k sigma_min(L_k)。
令 R 包含记录相位、逆变换、截取及 MSE 归一化，J=1/2 ||Ru-d||²；R 和 d
同时吸收原 MSE 所需尺度。半谱 irfft 是实线性重建，以下使用实化频谱空间及其
实 Hilbert 伴随；H 是该伴随的简记，不把 irfft 误当作任意半谱上的复线性矩阵。令

```
A u=b,             A^H lambda=R^H(Ru-d)
g_j=Re lambda^H (b_j-A_j u)
e_u <= ||r_primal||/beta
e_lambda <= (||r_dual|| + ||R||² e_u)/beta
|g_j-g_j,r| <= e_lambda ||b_j-A_j u_r||
              + (||lambda_r||+e_lambda) ||A_j|| e_u.
```

其中 r_dual=R^H(Ru_r-d)-A^H lambda_r。这解释了为什么波形误差很小也不能
独立保证速度梯度：原、伴随残差和稳定常数都参与。此界是代数条件界，尚没有
对本 Marmousi PML 系统计算可扩展的认证 beta。

总误差还包括空间 / 时间离散误差、降阶空间误差、频带截断、有限逆变换混叠、
浮点误差及速度离开建基点的误差。与原 FDTD 比较消除了共同离散误差的影响，
不能由此证明对连续真波场同样准确。

## 稳定性与因果逆变换

SPD 质量矩阵及刚度在原弹性 FEM 理论中很重要；本 PML Schur 补是非对称有理
算子，Q^T M Q 正定本身不足以证明其时间递推稳定。实现给出了等价辅助状态
递推的极点诊断。单个浮点谱只能描述该材料和该基，不能认证全速度区间；
单位圆上的 Jordan 块、非正规瞬态增长也不能仅由谱半径排除。

逆变换采用 |z|=exp(-log(alias_tolerance)/Nfft)>1。必须包住实际系统极点；
若某个降阶极点在该圆之外，积分仍可能给出平滑的有限记录，却不是因果递推解。
不做极点裁剪、不添加人为阻尼、不改变原 FDTD。频带 / 圆半径敏感性和实际极点
计算作为诊断单列，无法通过者不应作为反演替代器。

一个刻意很小的 CPU 强 PML 网格中，原 FDTD 和等价递推均出现谱半径约 1.05325，
尽管波速 Courant 数约 0.15。这是原边界离散在该玩具网格的实测现象，不是
Marmousi 不稳定的证明，也不能通过篡改原求解器掩盖。测试检查两个实现一致并
正确标记不稳定，而不是把该现象错误地认作适配代码故障。
