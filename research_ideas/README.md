# IFWI 五条独立研究路线 · v0.1

从 `226be2262377a0bbde4f813205ceb7cbcd69f335` 扩展，只新增文件，不修改原来的 RNN-FD、`core_runner.py`、数据、测试或实验配置。五条路线分别运行，不组合。完整推导、消融与停止门槛见 [实验方案](../docs/research_ideas/PROTOCOL.md)。

**交付是可执行、经过 CPU 数值与原正演集成测试的研究核心，不是五个已证明有效的方法。** 当前没有全尺寸 Marmousi/Overthrust 的新结果，没有 CUDA 显存/吞吐测试，也没有合格的预训练 DFM 权重。不要根据 smoke 的 RMSE 判断科研价值。

## 1. 内容与边界

| 方法 | 实现 | 关键边界 |
|---|---|---|
| DFM 双梯度 | 两时间共享传输、同时间对粒子组漂移训练、单步缓存/多步提议、参数梯度保护、真实波形验收 | 地震应用是新实验设计，不是作者视觉模型复现；输入提升不是精确后验 |
| MiTA 思路迁移 | 炮集分箱压缩、残差/多样性 top-k、补集探索与重要性校正 | 没有移植 fast-weight attention；无偏针对梯度估计，不针对 Adam 轨迹 |
| 小波 Attention | 残差 Haar 四子带、低频/细节条件门控、高频逐渐开放 | 背景不滤波；空间小波不替代时间频率延拓 |
| 可学习界面 | 有序曲线、正层厚、米制软阶跃、背景锚定、可选曲率项 | 不能表示悬垂/分叉盐体；SDF 混合函数有，完整 SDF 优化器没有 |
| 物理梯度 Attention | 两组炮梯度自监督正门控、VJP、冲突保护、实际位移验收 | 不构造二阶波场；正门控不保证参数空间下降或地质正确 |

`physics.py` 逐炮调用原始求解器，复用原有 `training_objective` 与低通能量归一化。用 `shot_batch_size=1` 限制波场反传显存；所有基线也用相同设置。它不是原 `core_runner` 的完整替代：首版只支持普通残差 SIREN、Adam、可选原频率延拓/LR schedule，拒绝 FR、低秩、预训练初始化、super-shot/coarse/function-space、已有 diffusion prior 等组合。原 checkpoint 不能直接载入。

## 2. 环境与快速检查

沿用项目已有 Python/PyTorch 环境；新增代码只使用已有 torch、numpy、scipy、pandas、pytest 等依赖，不需要小波或拓扑专用库。CPU 验证环境为 Python 3.13、PyTorch 2.10.0+cpu；其他版本及 Windows CUDA 未做本轮验证。仓库原导入链需要兼容的 torchvision。

```bash
python -m pytest verification/research_ideas -q
python -m research_ideas.runner --smoke --method baseline --steps 3 --out outputs/five_smoke_baseline
python -m research_ideas.runner --smoke --method routing --steps 3 --out outputs/five_smoke_routing
python -m research_ideas.runner --smoke --method wavelet --steps 3 --out outputs/five_smoke_wavelet
python -m research_ideas.runner --smoke --method interfaces --steps 3 --out outputs/five_smoke_interfaces
python -m research_ideas.runner --smoke --method gradient_attention --steps 3 --out outputs/five_smoke_gradient
```

这些命令使用 9×11 网格、3 炮、48 时间步的**原始** RNN-FD。新输出目录必须不存在，防止覆盖旧实验。CUDA 试跑加 `--device cuda:0`，仅有 CPU 的环境不能验证显存安全。

DFM 完整通路 smoke（明确为玩具权重）：

```bash
python -m research_ideas.train_dfm --synthetic-smoke --out weights/dfm_smoke --steps 4 --width 8 --patch-size 8 --group-size 2 --groups 2
python -m research_ideas.runner --smoke --method dfm --prior weights/dfm_smoke/best.pt --steps 3 --out outputs/five_smoke_dfm
```

禁止把上述权重用于正式实验。非 smoke 的 runner 会拒绝 `smoke_only` 权重；`allow_smoke` 还要求命令行显式 `--smoke`。

## 3. 实际反演

先跑普通 PMS 与带相同安全验收的 PMS 两个基线：

```bash
python -m research_ideas.runner --recipe research_ideas/recipes/baseline.json --device cuda:0 --out outputs/five_pms
python -m research_ideas.runner --recipe research_ideas/recipes/baseline_guard.json --device cuda:0 --out outputs/five_pms_guard
python -m research_ideas.runner --recipe research_ideas/recipes/routing.json --device cuda:0 --out outputs/five_routing
python -m research_ideas.runner --recipe research_ideas/recipes/wavelet.json --device cuda:0 --out outputs/five_wavelet
python -m research_ideas.runner --recipe research_ideas/recipes/interfaces.json --device cuda:0 --out outputs/five_interfaces
python -m research_ideas.runner --recipe research_ideas/recipes/gradient_attention.json --device cuda:0 --out outputs/five_gradient
```

预设以 Marmousi smooth400 为起点，3→5 Hz→全频带（100/100/余下更新），总计 1000 步，每 50 步全炮评价。`--seed 0/1/3` 分别运行。`--steps 10` 可先做真实尺寸显存/耗时预检，但这不代表进入所有方法的启用阶段；DFM/梯度 Attention 默认第 101 步才启用。预检应另配 warmup=0 的配置，不能把“未启用模块”的时间当方法成本。

恢复到**新的输出目录**，只改变 recipe 的 steps/命令行预算，其他源码/数据/参数保持原样：

```bash
python -m research_ideas.runner --recipe research_ideas/recipes/routing.json --steps 2000 --resume outputs/five_routing/checkpoint.pt --device cuda:0 --out outputs/five_routing_extended
```

checkpoint 保存模型、Adam、gate 与其 Adam、路由 RNG、DFM 缓存、全局 RNG、源码与数据哈希。恢复会重新生成观测，额外成本会累计。只加载自己生成的可信 checkpoint；`--resume` 使用 PyTorch pickle。DFM 权重另用 `weights_only=True`。跨设备/不同 Torch 版本恢复主动拒绝。本轮逐位续训测试只在 CPU 上完成。

## 4. 训练 DFM

每个 `.npy` 是独立二维速度模型，单位 m/s；训练/验证按原始模型分割，不能把同一模型的不同裁块分到两边。提供的数据必须已经处于声明的物理网格间距，程序不会从无元数据的 npy 猜测或自动修正 spacing。训练不需要地震炮集。

```bash
python -m research_ideas.train_dfm --train-data datasets/prior_train_15m --validation-data datasets/prior_valid_15m --out weights/dfm_15m --steps 20000 --patch-size 32 --width 32 --group-size 8 --groups 2 --spacing-m 15 --device cuda:0
python -m research_ideas.runner --recipe research_ideas/recipes/dfm_dual.json --device cuda:0 --out outputs/five_dfm
```

`20000` 是实验预算，不是训练充分的断言。检查验证 energy distance、样本多样性、局部速度范围，再启动 FWI。该训练器提供 best 权重与历史，不提供离线训练 optimizer 的断点续训；反演 runner 的续训是完整的。训练和验证的准确重复模型会被拒绝，反演真值与记录的先验模型完全重复也会拒绝；不同采样、局部裁剪、近重复泄漏仍需人工按来源审查。

Overthrust 当前协议为 20 m，需要单独训练/验证 20 m 的 prior。此实验包不调用旧的物理尺度桥接，避免把跨尺度插值混进五项对照；同为 `.npy` 并不意味着尺度一致。

## 5. 生成实验矩阵与分析

```bash
python paper_supplement.py prepare-data
python -m research_ideas.experiments generate --out outputs/five_matrix --seeds 0 1 3 --datasets marmousi overthrust
python -m research_ideas.runner --spec outputs/five_matrix/specs/marmousi_s0.json --recipe outputs/five_matrix/recipes/marmousi_wavelet.json --device cuda:0 --out outputs/marmousi_wavelet_s0
```

生成器默认 2 数据集×3 seeds×7 配置＝42 个任务，**只生成 manifest，不自动启动任务**。加 `--include-dfm` 会生成单步、多步和双模式三项，要求各 spacing 的独立合格权重。困难背景用另一个目录生成 `--background gaussian` 或 `--background linear`，噪声用 `--noise-sigma 2`（标准差比，不是 2 dB）。

单个相同 seed/模型/背景/观测/采集/设备组的实测时间—误差前沿：

```bash
python -m research_ideas.experiments summarize --runs outputs/five_pms outputs/five_wavelet --out outputs/five_pareto.json
```

不允许把不同问题的点拼成一个前沿。JSON 保存所有评价点和非支配点索引；时间已包含先验、路由刷新、真实验收与评价。离线 DFM 训练时间另外报告，未自动摊销。固定 RMSE 达标时间只能事后计算，不能把测试真值塞进停止条件。

## 6. 可直接运行的消融

`recipes/` 还包括：`routing_uniform`、`wavelet_fixed`（固定小波课程，无可学习增益）、`wavelet_no_ramp`、`gradient_identity`、`interfaces_disabled`、`dfm_direct`、`dfm_multistep`。把相应 JSON 传给同一 runner 即可。更广泛的等参数普通 CNN、一般 SDF、断层分区、held-out 炮选择需要额外实现，已在完整方案里标注，不能称已完成对照。

## 7. 输出与判读

`run.json` 保存配置、数据/先验/源码身份；`history.json` 是逐更新日志；`evaluations.json` 是全炮全频带损失与事后速度指标；`metrics.json` 包含正演炮数、伴随炮数、观测合成炮数、总秒数、prior 调用数、拒绝步数、CUDA 峰值（CPU 为 null）。`final_velocity.npy` 为终态，`lowest_misfit_velocity.npy` 按波形失配选出，不能把它冒称最低真值 RMSE 的模型。

计数是 shot-equivalent PDE solves，不是把并行批次当一炮；伴随为自动微分反传的成本计数，不声称与正演耗时相等。所有原始方法同样逐炮，才能公平比较。当前合成观测沿用原仓库算子，存在同算子合成条件；正式论文还需独立网格/子波误差压力测试。

验证记录见 [VALIDATION.md](../docs/research_ideas/VALIDATION.md)。
