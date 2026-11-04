# 条件 DFM 与弱灵敏度软辅助原型

本原型把新条件 DFM、采集灵敏度诊断和受物理审查的软目标分开。**没有合格的地质条件先验权重，不能运行正式先验反演；没有证据证明当前实现提升反演精度。** 训练、推理和科学资格是三件不同的事。

## 已实现与未证明

- 新的双时间条件输运网络，可选小 CNN 或三尺度 U-Net。输入条件为独立背景、估计速度锚点与二值锚点掩码；模型被冻结，生成从标准高斯的 t=0 走到地质域 t=1。旧平滑条件 checkpoint 不兼容，也不把当前速度加噪后冒称 DFM 去噪。
- 原 RNN-FD 接收域随机 VJP 灵敏度诊断、有限预算、缓存软速度/界面差分目标、严格实际物理损失审查、精确恢复。
- 另有条件训练入口与数据协议，见 `CONDITIONAL_DFM_TRAINING.md`。所有新训练产物默认未取得反演使用资格；几个合成训练步只验证接口，不是预训练完成或效果失败的根因。
- 未证明：三尺度或 attention 带来精度收益、复杂断层的全局一致性、误差条件下可靠补全、统计不确定性校准、16 GB GPU 可行性、完整 Marmousi/Overthrust 改善。没有硬地层界面或盐体二相假设。

## 1. 灵敏度不是残差梯度或置信度

固定当前独立背景 v₀，令 F 为原 RNN-FD 生成的全部接收记录。可选先低通再施加明确的数据域振幅掩码 M，得到 F̃=M·L·F。每个炮批独立生成接收点/时间/炮上的 Rademacher 随机符号 z，累积 `(J_bᵀ z)²` 的均值，并按全部记录数 B·S·T·R 归一化。其期望为 `diag(JᵀJ)/(B·S·T·R)`，J 是 F̃ 对物理 m/s 速度的 Jacobian。

没有读取观测残差、地质真值或模型误差，也不使用由观测能量计算的频率缩放。不同炮批的平方 VJP 分别相加，不能先把梯度相加再当作同一个确定性对角值。有限随机探针有方差，不同批大小的单次结果不要求逐位相等。

原始对角的单位是记录振幅²/(m/s)²。为了空间权重，先除以峰值，再用 `I=u/(u+mean(u))` 得到有界相对图。全零图标记为 `no_sensitivity`，不会被自动解释为“所有地方都应强用先验”。I 不是后验方差、可分辨率或速度估计正确性的概率。它受波源、接收孔径、记录长度、频段、背景与数值边界影响。

可显式传入数据掩码以排除预先指定的时间窗/接收点；它乘在预测记录上，分数权重会在 Gram 对角中平方。没有从数据残差自动推断“噪声位置”或假装去除了所有震源误差。非平凡掩码会改变返回的算子身份；当前物理适配器没有对应的掩码目标，因此只能诊断，不能静默接入未屏蔽的物理目标。

**无真值边界**：NativeSensitivity 只接收几何、子波、显式物理 PML 最大速度和背景。原仓库 `build_forward_problem` 会用真值最大速度设置 PML，不能直接把那条路径当作无泄漏证明。做公平对照时应让物理基线和灵敏度都使用同一独立声明的 PML 上界；仅在生成合成观测和最终评价时使用测试真值。

## 2. 软目标、强弱区与缓存

`IlluminationSoftPrior` 将 I 从 weak_threshold 到 strong_threshold 线性映射为弱/强区域权重。默认弱区1、强区0，高 I 的锚点仍只是当前估计，不是已知真值。候选不会被硬覆盖进反演，强区也不会因先验而直接冻结。

速度项匹配 `(v-target)/scale_mps`。边缘项匹配相邻差分，并乘 `edge_length_m/spacing_m` 保持物理长度定义。边上的权重取两端权重的最小值，因此零权重强区不会从跨界差分直接收到先验梯度。共享 INR 参数仍可能间接改变这些位置。

重要：现有梯度保护会把非零先验梯度归一化到 `prior_ratio`。因此空间权重控制**相对空间偏重**，统一乘小系数不会降低总先验梯度幅度；总体强度由 `ResearchConfig.prior_ratio` 控制。将该值设为0，或将软目标有效权重设为0，会完全跳过先验与其审查开销，保持 baseline 更新路径。公平研究还应另比同样审查频率的 guarded baseline。

候选每 `refresh_every` 个更新重新生成一次，中间保持固定，以节省网络调用并固定本轮软目标。候选为 detached 张量，生成网络 eval 且参数冻结。超出物理正速度或声明的 RMS 变化预算时直接拒绝，不裁剪成假合格样本，也不随机重抽直到看起来满意。

缓存绑定背景、灵敏度图、配置、先验权重与采样种子/NFE、几何/子波/PML 构造参数身份和频段。改波源或频段会在额外正演之前失败，须显式重算相应图并重建 prior。原型不自动编排多频段。原生数值卷积核和 PML 系数按构造后不可变处理；身份检查不是对任意 Python monkey-patch 或手工修改系数的完整检测。缓存目标、刷新位置、调用计数进入引擎 checkpoint。恢复时拒绝非有限/负速度目标。

所有活跃软目标步继续采用已有参数梯度保护与实际物理目标严格下降检查，失败可用纯物理梯度恢复。降低当前测量目标并不证明新地层是真实的；在严格物理驻点处，保护也可能不允许纯先验动作。没有声称其是精确 MAP、固定概率正则或后验采样。

## 3. 可直接运行的 CPU 诊断

```bash
python -m ifwi_research.demo_illumination --out outputs/illumination_diagnostic
python -m pytest -q verification/five_ideas/test_illumination.py verification/five_ideas/test_illumination_prior.py verification/five_ideas/test_illumination_engine.py verification/five_ideas/test_conditional_dfm.py
```

诊断使用显式选择的 12×20 背景、3炮、64时间样本，不创建真值或观测。输出 `sensitivity.npz` 与 `summary.json`，默认明确记录 `not_run_missing_qualified_checkpoint`。没有伪造 DFM 权重，也没有反演更新。可用 `--mute-before-s` 指定数据域时间窗，或 `--cutoff-hz` 指定低通。

`--checkpoint` 只接受新的独立审查过的条件 checkpoint。此诊断即使成功生成，也仅保存 `candidate_unaccepted_mps`，不把未经物理接受的候选称作反演结果。

## 4. 物理接入示例

下面假定 geometry、wavelet、background、observed、cfg 来自已核实的采集设置与数据；不是从测试真值生成 prior 条件。网格、单位与数组形状应先独立核对。

```python
from ifwi_research.illumination import NativeSensitivity
from ifwi_research.conditional_dfm import FrozenConditionalDFM
from ifwi_research.illumination_prior import IlluminationSoftPrior, SoftPriorConfig
from ifwi_research.repository import RepositoryObjective
from ifwi_research.engine import ResearchEngine, ResearchConfig

sensitivity = NativeSensitivity(
    geometry=geometry, wavelet=wavelet, dt_s=cfg['forward']['dt_s'],
    spacing_m=20., npad=3, free_surface=True,
    pml_vmax_mps=3000., shot_batch_size=1)
report = sensitivity.estimate(background, probes=2, seed=3, cutoff=None)
if report['status'] != 'ok':
    raise ValueError('No usable sensitivity; review geometry/source/context')

# Measurements enter ONLY the physical objective, not the sensitivity estimator.
problem = dict(
    propagators=[dict(first=a,last=b,solver=s) for a,b,s in sensitivity.batches],
    observed=observed.detach(), wavelet=wavelet,
    geometry_tensors=geometry, config=cfg)
objective = RepositoryObjective(problem)
source = FrozenConditionalDFM('reviewed_conditional_checkpoint.pt', 20., nfe=1)
prior = IlluminationSoftPrior(
    source, background, report['illumination'], spacing_m=20.,
    operator_identity=report['operator_identity'], cutoff=None,
    config=SoftPriorConfig(refresh_every=5))
engine = ResearchEngine(
    base_model, background, objective, 20.,
    ResearchConfig(method='dfm', prior_ratio=.025, prior_start=1, prior_every=1),
    prior=prior)
record = engine.step(cutoff=None)
```

`FrozenConditionalDFM` 检查新架构、归一化/物理间距、条件/时间/锚点训练声明、独立 train/validation 哈希及资格报告引用/哈希，不接受旧权重或不合格覆盖。报告字段仍是研究者声明，不是加载器对科学质量的认证。

## 5. 预算和验证边界

当前灵敏度实现限 CPU：网格≤4096像素、nt≤256、炮数≤16、探针2–8、总记录样本≤1,000,000、单炮批时间×带PML网格项≤2,000,000、估算工作项≤100,000,000。限制在传播前检查，仅为原型保护，不是峰值内存保证。每次只保留一个炮批图；原 FD 的缓存速度会 detach。

灵敏度诊断正演源数 S、伴随源数 K·S 独立记录；它们不算成免费的“准备”。候选网络 evaluations、缓存命中与刷新数单独记录。总预算应含训练、诊断、物理审查、网络调用和最终评估，而不是只数 epochs。

小网格测试覆盖分析矩阵 VJP、显式原生 Jacobian、随机性、屏蔽/低通、单位梯度、资源拒绝、零权重等价、实际物理下降、缓存恢复以及算子身份变化拒绝。解析候选 fixture 明确标记 `analytic_fixture_not_dfm`，只用于验证耦合，不是伪造的预训练地质模型。

## 6. 下一步怎样判断“效果差”

不能把20步 smoke、某次输出噪声或图像更锐利当作根因分析。至少分别检查：训练漂移目标与时间方向、归一化链式法则、条件来源/信息量、训练和推理的背景/锚点误差分布、EMA/采样时间网格、真实物理接口与计算预算。

完整父地质模型先分 train/validation/test 再裁块；相关裁块、同一父模型变体不能跨分割。数据条件来自明确的数据-only重建，并记录其误差/噪声生成过程；测试真值不得用于背景、掩码、模型选择或刷新决策。还需实验：错误背景/锚点、先验地质失配、未参与优化的炮、边界位置与连续性、薄层保留、模型误差、物理残差、多种子。样本间差异尚未校准为可信区间。

对照至少包括原 IFWI、同成本 guarded IFWI、非学习空间/TV控制、相同候选的均匀权重与灵敏度权重、同预算 CNN/多尺度模型。未运行这些实验前，不承诺提升，更不能用不同炮数、频段、初模或求解次数的已有结果作公平比较。

## 参考与许可

设计借鉴 [DOLCE 的先验/物理接口分离](https://github.com/wustl-cig/DOLCE) 和 [diffefwi 的交替更新](https://github.com/DeepWave-KAUST/diffefwi)，均核对 MIT 许可。未复制代码或权重，未移植 DDPM/CT 算子。详细出处见 `THIRD_PARTY_NOTICES.md`。
