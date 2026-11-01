# SIAM 2025 地震降阶模型复现及 IFWI 适配

这是独立的研究分支，基于原项目 `9b0bdf4`。保留原 FDTD、原波形 MSE、原
IFWI 参数化与 Adam 更新顺序；没有引入新的神经网络，也没有覆盖任何历史实验。
数值实验用于判断降阶物理求解器是否值得继续投入，不以较小波形误差代替反演精度。

**实测结论、全部维数结果和一次实际更新后的失配见 [RESULTS.md](RESULTS.md)。**
建基点约30倍的物理调用耗时优势未能转化为可靠的同精度反演加速。

## 文件与入口

- `experiments/rom/paper.py`：原论文的加权 POD、Galerkin、Ricker-Laplace、Weeks 逆变换。
- `experiments/rom/reproduce_upstream.m`：在作者归档副本中运行未经修改的 Figure 8。
- `experiments/rom/upstream_core_fixture.m`：直接调用作者 Weeks 核心的独立小系统。
- `experiments/rom/discrete.py`：从原 FDTD 提取离散算子并精确消去 PML 状态。
- `experiments/rom/basis.py`：固定实基、源约束、频率采样，以及可选的伴随快照。
- `experiments/rom/solver.py`：可微分的离散频域压力 ROM，固定量全部为 buffer。
- `experiments/rom/stability.py`：等价递推及极点诊断，不把 PML 当作 SPD 系统。
- `experiments/run_rom_benchmark.py`：独立 CPU/GPU 基准及有限次 IFWI 试更新。
- `experiments/diagnose_rom_benchmark.py`：计时外的极点、逆变换及频带敏感性检查。
- `experiments/run_rom_local_pilot.py`：最多3次实际更新，逐状态与原 FDTD 比较后决定是否继续。
- `experiments/report_rom_benchmark.py`：从原始结果生成全秩表格和可追溯科学图。
- `verification/rom/`：CPU 测试及真实 MATLAB 导出的固定参考数据。

数学推导、已证明性质和不能直接搬用的结论见 [MATHEMATICS.md](MATHEMATICS.md)。
实际原方法复现结果和归档校验值见 [REPRODUCTION.json](REPRODUCTION.json)。

## 实验协议及指标

Marmousi 使用仓库 CSV、原 legacy header 读取规则、下采样 4 倍，实际 94×288 网格。
dx=15 m、dt=0.0019 s、nt=1000、8 Hz Ricker、15 格 PML、原 order=2 和 free_surface=true。
Gaussian σ=15，单位 m/s。用中央单炮 x=144、z=1，接收器 z=2、横向全网格。
这是物理模块的 S1 基准，**不是历史 49 炮、nt=2632、1000 次更新实验的复跑**。

基仅在背景速度建成，固定到所有测试状态：背景、独立的 2% 平滑正弦速度扰动、
完整真速度。真速度波场不参与建基；伴随模式用已有观测残差，需要一次背景校准。
两个求解器共用原实验由真模型最大值设定的 PML，不将此宣称为未知真模型部署。

报告指标：接收记录相对 L2、四个时间截面的全物理网格压力相对 L2、原波形 MSE
对速度的梯度相对 L2 / 余弦、同一原 SIREN 的非零末层状态参数梯度误差。
真速度时原残差为零，梯度相对误差无定义，保留 ROM 的绝对伪梯度。
自动准入要求背景及留出扰动的波形误差均 <1%、速度及 SIREN 梯度误差均 <5%。

硬件：RTX 5060 Ti 16 GB。原 FDTD float32，ROM float64 / complex128；TF32 关闭。
预热后交替顺序计时三次，报告中位数，包含在线质量矩阵投影、频域求解、逆变换
和反向传播。GPU 同步。离线建基、静态投影、每个炮集的背景校准另外列出。
两种求解器及公共数据同时驻留，因此显存主比较用相对调用前的图增量峰值；
完整 allocated / reserved / resident 值也保留，不能将其解释为各自独占进程峰值。

## 复现命令

在该分支根目录，使用包含 NumPy、SciPy、PyTorch、PyYAML、pytest 的项目环境：

```powershell
python -B -m pytest verification/rom verification/residual_ifwi -q -p no:cacheprovider

# CPU 小网格；output 必须是不存在的新目录。
python -B -m experiments.run_rom_benchmark --small --device cpu --output outputs/rom_cpu_new

# 原始正演快照基，与正演/伴随联合基的独立 GPU 结果。
python -B -m experiments.run_rom_benchmark --config experiments/configs/rom_marmousi.yaml --device cuda:0 --output outputs/rom_primal_new
python -B -m experiments.run_rom_benchmark --config experiments/configs/rom_marmousi_adjoint.yaml --device cuda:0 --output outputs/rom_joint_new
python -B -m experiments.run_rom_benchmark --config experiments/configs/rom_marmousi_enriched.yaml --device cuda:0 --output outputs/rom_enriched_new

# 额外诊断，不重建基；局部试更新不会取消留出状态失败的结论。
python -B -m experiments.diagnose_rom_benchmark --run outputs/rom_enriched_new --ranks 192 --device cuda:0
python -B -m experiments.run_rom_local_pilot --run outputs/rom_enriched_new --rank 192 --updates 3 --device cuda:0 --output outputs/rom_guarded_pilot_new

# 所有基准均保存完整配置、逐文件 SHA256、源码副本、基、ROM state_dict、
# 原始波场、每次计时样本和失败原因；仅通过精度门槛才运行配置中的 3 次试更新。
# 无任何自动 1000 次训练入口。
```

MATLAB 复现：先校验 Zenodo ZIP，再完整解压到单独副本，把两个 `.m` 封装复制到
副本根目录，以 MATLAB `-batch "run('reproduce_upstream.m')"` 执行；随后执行
`upstream_core_fixture.m`。所需工具箱及原文件名以作者 README 为准。原归档测试
版本为 R2023b，本机实跑 R2020b；此版本差异已经纳入记录。

## 数据与限制

紧凑的汇总、配置、计时明细、图及图源数据在本目录的 `measurements/`；大型波场、基和
checkpoint 保留在 GPU 机器的独立输出目录，没有上传数百 MB 的临时文件。
`LOCAL_ARTIFACTS.json` 给出实际运行目录及哈希；不需要 checkpoint 即可复核表格。

成功复现原论文的固定材料算例，不等于已经解决高维 Marmousi 反演。在固定背景
准确的 ROM 可能在速度改变后失效；动态刷新、认证稳定常数、13/49 炮扩展、
nt=2632 及长反演的收敛精度，需要分别验证。任何未通过梯度门槛的耗时比只称为
求解调用耗时比，不作为同精度 FWI 加速结论。
