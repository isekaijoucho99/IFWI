# Muon：用正交化更新训练 IFWI 的隐藏层

对应论文：McGinnis, Hölzl, Shit, Bieder, Friedrich, Mühlau, Menze, Rueckert & Wiestler, *Optimizing Rank for High-Fidelity Implicit Neural Representations*, ICML 2026, PMLR 306（项目页 rank-inrs.github.io）。优化器为 Muon（Jordan et al., 2024，github.com/KellerJordan/Muon）。

> **GPU 结果（2026-10-15，13 炮、seed 3、1000 次）**：IFWI13 由 Adam 的 749.0 降到 519.9 m/s（仍在下降，但未低于初始模型 518.8）；FR13 由 Adam 的 375.3 变差到 470.9 m/s，数据残差为 Adam 的 180 倍，速度场有颗粒感、loss 振荡。FR + Muon 不再继续；分析与后续改进（多尺度频率延拓）见 [多尺度文档](multiscale_ifwi.md)。

## 论文给出的依据与边界

- 论点：INR 的低频偏差可视为训练中层稳定秩 s(W) = ‖W‖²_F / ‖W‖²₂ 退化的结果；单层梯度 ∇W = G Hᵀ 的稳定秩受输入激活 H 的秩限制（Eq. 2）。Fourier 特征、SIREN 等结构改动可看作提高 H 的秩；Muon 把每层更新映射到近似半正交矩阵（Eq. 3，Newton–Schulz 近似），从优化器层面保持秩。
- 做法（§3.3、附录 A）：隐藏层权重矩阵用 Muon；输入/输出矩阵和全部偏置用 Adam；两者 β = (0.9, 0.999)（PyTorch 默认）；不用权重衰减；两种优化器使用同样的学习率调度；Muon lr 与辅助 Adam lr 做网格搜索（各 9 个值）。
- 结果：图像、音频、SDF、NeRF 普遍提升；反问题中，稀疏角 CT 上 SIREN 30.58 → 33.69 dB（表 4），单图超分辨率上 SIREN、FINER 无提升（表 5）。
- 局限（§5）：稳定秩不给出最优目标秩；**对病态反问题，较低的稳定秩可能本身是隐式正则**；Muon 需要额外调学习率与调度。

IFWI 是严重病态的非线性反问题，因此关键不是"秩越高越好"，而是正交化更新能否转化为更好的速度模型。

## 实现

| 文件 | 内容 |
| --- | --- |
| `experiments/muon_optimizer.py` | Muon（参考实现的算法：EMA 动量 0.95 + Nesterov、bfloat16 五步五次 Newton–Schulz、系数 (3.4445, −4.7750, 2.0315)、max(1, 行/列)^0.5 缩放）；`MuonWithAdam` 组合；参数划分；`build_optimizer`；稳定秩工具 |
| `experiments/residual_ifwi_experiment.py` | `training.optimizer`（可选）；所有运行每步记录隐藏层 `hidden_weight_srank`、`hidden_update_srank`、`hidden_update_fro` |
| `experiments/rank_report.py` | 离线读检查点：隐藏层有效权重与各层输入特征（全网格）的稳定秩、谱范数 |
| `experiments/update_response_report.py` | 支持配置中的优化器；`--muon-lrs` 按多个 Muon lr 复制配置，用固定初始梯度比较速度更新规模 |

配置（未设置 `training.optimizer` 时与原 Adam 逐位相同，已对 5 份配置在旧新代码间核验）：

```yaml
training:
  learning_rate: 0.0001          # Adam：输入层、输出层、偏置、低秩核心 C
  optimizer:
    name: muon
    learning_rate: 0.01          # Muon：隐藏层矩阵
    momentum: 0.95
    nesterov: true
    ns_steps: 5
```

设计选择：

1. **按层位置选 Muon 参数**：每个原版 IRN 的第 1…n−2 层（隐藏到隐藏）。二维 SIREN 的输出层是 1×128 矩阵，按形状筛选会误判。FR 层交给 Muon 的是系数 `lamb`（`weight` 是 `lamb @ bases` 的非叶张量）；可训练 ω 解包后不进入 Muon，仍用其 Adam 组与学习率；低秩模型两支因子网络的隐藏层用 Muon，核心 C、因子输出层用 Adam。
2. **辅助优化器是 `torch.optim.Adam`**（论文的 β = (0.9, 0.999)，eps 1e-8），而非参考实现中 betas (0.9, 0.95)、eps 1e-10 的手写 Adam，使非隐藏参数的更新与基线完全相同。
3. **不原地修改梯度**（与参考实现算术相同），**先累加全部炮的梯度再执行一次 step**（对各批分别正交化再相加是另一种方法）。
4. 预训练（论文模式 Stage 1）仍用 Adam；续训必须使用同一优化器（配置不同会被拒绝），检查点中的状态为 `{"muon": …, "adam": …}`。
5. Newton–Schulz 用 bfloat16（参考实现）。在真实 IFWI 隐藏层梯度上与 fp64 比较：落在 σ < 10⁻³σ_max 方向上的更新能量均 ≤ 2.2 %，两者几乎相同。

## CPU 诊断（13 炮、σ = 15，未训练）

**隐藏层梯度与特征的秩很低。** 初始化时（全网格）各层输入特征的稳定秩：

| 网络 | 第 1 层 | 第 2 层 | 第 3 层 | 进入输出层 |
| --- | ---: | ---: | ---: | ---: |
| 普通 SIREN（IFWI13） | 28.1 | 2.3 | 1.8 | 1.6 |
| FR（FR13） | 28.1 | 21.4 | 17.2 | 15.8 |
| LR 因子网络（归一化坐标） | 11.4 / 10.5 | 2.2 / 2.4 | 1.7 / 1.7 | — |

普通 SIREN 的特征随深度退化到约 2，FR 保持在 16–21，符合论文"结构改动提高特征秩"的框架。做 5 步 Adam 使输出层非零后，隐藏层梯度稳定秩：IFWI13 为 12.0 / 1.38 / 1.04，FR13 为 2.4 / 2.1 / 1.2。Muon 把这些弱方向的步长拉到与主方向相同——这正是它在病态问题中的收益或风险所在。

**固定初始梯度下的更新响应**（`update_response_report`，第 1 步精确，之后梯度固定；残差模型输出层为零，第 1 步隐藏层无更新）：

| 配置 | 隐藏层累计更新稳定秩 | 第 20 步 RMS Δv（m/s） |
| --- | ---: | ---: |
| IFWI13 + Adam | 5–7 | 202 |
| IFWI13 + Muon，lr 3e-4 / 1e-3 / 3e-3 | 约 60 | 170 / 186 / 167 |
| FR13 + Adam | 2–5 | 216 |
| FR13 + Muon，lr 3e-4 / 1e-3 / 3e-3 | 约 53 | 57 / 84 / 137 |
| FR13 + Muon，lr 5e-3 / 1e-2 / 2e-2 | 45 / 36 / 28 | 170 / 204 / 208 |

- 二维残差模型的速度变化以 Adam 更新的输出层为主，对 Muon lr 不敏感。
- **Muon 去掉了 FR 在 Adam 下约 6 倍的步长放大**：Adam 在 Λ 坐标逐元素归一化而放大 ΔW = ΔΛ·B；Muon 按谱范数控制 ΔΛ，ΔW 的奇异值约为 lr·σ(B)（σ(B) ≈ 0.64–0.91），与普通 SIREN 上的 Muon 量级相同。因此系数空间与"有效权重空间"Muon 的差别受 B 的条件数（约 1.43）限制，第一版不实现后者。
- 学习率：`paper13_ifwi_muon`、`residual_ifwi_baseline1000_muon` 取 1e-3（隐藏层更新的 Frobenius 范数与 Adam 1e-4 相当，约 0.010 vs 0.013）；`paper13_fr_ifwi_muon` 取 1e-2（20 步速度变化与 FR13 + Adam 一致，避免把步长差异误当成优化器差异）。这只是筛选中心，正式比较应在其 0.3×、1×、3× 附近扫描；论文对每个架构都做了网格搜索。

## 实验计划

1. **IFWI13 + Muon**（`paper13_ifwi_muon.yaml`）：IFWI13 用 Adam 失败（RMSE 749），FR13 成功（375）。若 Muon 救回普通 SIREN，说明 13 炮失败与隐藏层更新的低秩有关。
2. **FR13 + Muon**（`paper13_fr_ifwi_muon.yaml`）对 FR13（375.3 m/s）：速度步长已匹配，差异更可归于更新几何。若 FR13 + Muon ≈ IFWI13 + Muon，FR 在 Adam 下的优势主要来自步长放大；若 FR13 + Muon 明显更好，则 FR 的高特征秩本身有作用。
3. 学习率扫描（各 3 个值）后再做多种子。
4. **49 炮基线 + Muon**（`residual_ifwi_baseline1000_muon.yaml`）对 baseline1000（438.4 m/s），检验在能正常收敛的协议上的通用性（每次约 9.7 h）。
5. 若 Muon 有收益：加一个"相近更新幅度、不正交化的动量更新"对照，排除收益只来自动量或步长；论文用 Lion 作类似对照。
6. LR 与 FR + LR 等 LR 在 Adam 下稳定后再加入。

评价沿用波形 MSE、速度 RMSE / SSIM、`residual_band_report`（分波数、分深度），并用 `rank_report` 与运行日志看秩的变化：稳定秩升高但速度更差，说明被增强的方向主要承载欠约束结构。

```bash
python -m experiments.update_response_report --configs experiments/configs/paper13_fr_ifwi.yaml experiments/configs/paper13_fr_ifwi_muon.yaml --muon-lrs 0.003 0.01 0.03 --steps 1 5 20 --output-dir outputs/muon_response
python residual_ifwi.py --config experiments/configs/paper13_ifwi_muon.yaml --device cuda:0 --output-dir outputs/paper13_ifwi_muon
python residual_ifwi.py --config experiments/configs/paper13_fr_ifwi_muon.yaml --device cuda:0 --output-dir outputs/paper13_fr_ifwi_muon
python -m experiments.rank_report outputs/paper13_fr_ifwi/RUN outputs/paper13_fr_ifwi_muon/RUN --labels fr13_adam fr13_muon --output-dir outputs/rank_report
```

`rank_report` 也可直接用于已有的 Adam 运行（如 FR13、IFWI13），不需要重跑；检查点每 50 次一个，逐步更新的稳定秩只在新运行的日志中有。

## 测试

```bash
python -m pytest verification/residual_ifwi/test_muon.py -q
```

覆盖：Newton–Schulz 压平奇异值且近似尺度不变、与参考算术一致且不改梯度；按层位置选参数（普通、FR + ω、低秩）；Adam 路径与原构造相同、Muon 与 Adam 恰好覆盖全部参数且 Adam 用 PyTorch 默认值；配置校验与续训规范化；Muon 运行记录稳定秩、续训逐位一致、换优化器续训被拒绝；`rank_report`；三份配置只改优化器。
