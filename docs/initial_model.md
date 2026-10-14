# 初始模型与第一步梯度方向（13 炮协议）

> **2026-10-14 更新**：σ = 22 的 IFWI13 未通过第 0 步门槛（RMSE 551 → 约 705 → 1000 次 638），第一步梯度方向为正并不能保证收敛，σ 不是 13 炮失败的充分解释。LR 系列此后的诊断与计划见 [LR-IFWI 文档：条纹与不稳定的来源](lr_ifwi.md)。

## 发现

13 炮协议（`paper13_*`：13 炮、炮距 300 m、1.9 s、每步全炮、Adam 1e-4）下，σ = 15 格的高斯初始模型上所有不带 FR 的运行都失败，速度 RMSE 从 518.75 m/s 上升：

| 运行（σ = 15，seed 3） | 结果 |
| --- | --- |
| IFWI13（residual） | 1000 次后 749.0 m/s |
| 残差 LR（gain 5.50） | 约 701 m/s，第 623 次越过 CFL 停止 |
| 残差 IFWI + 预训练特征 | 1000 次后 698.9 m/s（最高约 705） |
| 残差 LR + 预训练特征 | 第 291 次 787.8 m/s，第 292 次越过 CFL 停止 |
| FR13（FR + 可训练 ω） | 375.3 m/s，前约 90 次也先升到约 627 |

FR-IFWI 论文的 IFWI 在同类协议下单调下降（其 Fig. 6b 速度 MSE 约 0.30 → 0.18 (km/s)²，RMSE 约 548 → 424 m/s）。

`experiments/initial_gradient_report.py` 在每个候选 σ 处做一次全炮 FD 正演与反传，得到速度梯度 G，计算最速下降方向 −G 与 v_true − v_init 的余弦（全网格、150 m 以下、径向波数带）以及梯度能量在浅部的占比。13 炮、`paper13_ifwi.yaml` 几何、CPU：

| σ（格） | 初始 RMSE（m/s） | cos 全网格 | cos 0–1 c/km | cos 1–8 c/km |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 436.7 | +0.023 | +0.584 | +0.054 |
| 10 | 496.1 | −0.017 | +0.257 | −0.064 |
| **15** | **518.8** | **−0.059** | **−0.575** | **−0.106** |
| 18 | 530.7 | −0.049 | −0.567 | −0.036 |
| 20 | 539.9 | −0.006 | −0.411 | +0.030 |
| **22** | **550.6** | **+0.034** | **+0.231** | **+0.060** |
| 25 | 569.6 | +0.051 | +0.544 | +0.058 |

σ = 15 时第一步在所有尺度上把模型推离真值，长波长尤甚；关系随 σ 不单调，说明 σ = 15 恰处在部分到达波的周波跳跃范围内。梯度约 90 % 的能量在 150 m 以内。把检波器放到地表、震源放到 30 m（论文正文几何）时结论相同（3 炮检查：cos 全网格 −0.029 vs 代码几何 −0.027，深于 500 m 均为 −0.206），观测几何不是原因。σ = 15 是我们设计 residual 实验时选的，三篇论文都没有给出 σ。

## σ = 22 及其选择方式

σ = 22 的初始 RMSE 550.6 m/s 与 FR-IFWI Fig. 6b 的初始误差（约 548 m/s）最接近，且第一步在各尺度上方向为正。选择规则是"匹配论文初始误差"；但我们是在看到上表余弦后才定下 σ，因此并非完全不参考真值，报告结果时应注明。第一步方向为正只是必要性检查，不保证收敛，须由训练确认。

## 分阶段计划

前一步通过再做下一步：

1. **IFWI13，σ = 22**。通过标准：RMSE 自始下降、无早期上升，1000 次接近论文 IFWI 的约 424 m/s。不通过则 σ 解释不成立，转做 IFWI13 的学习率扫描。
2. **FR13，σ = 22**：IFWI 本身正常时 FR 的增益，对照论文约 11 %（速度 MSE）。
3. **LR 系列，σ = 22**：先 `lrpaper13_ifwi` 与 `lrpaper13_lr_ifwi`（论文模式），再 `paper13_pretrained_*`；LR + FR 最后（预训练已知不稳定，见 [LR-IFWI 文档](lr_ifwi.md)）。

σ = 15 的结果保留为"对初始模型敏感"的反例：同样的数据与方法，普通 IFWI 失败，FR 的约 6 倍有效步长越过了这一区域。

配置不改，命令行覆盖 σ（写入运行目录的 `config.json`，续训须带相同参数）：

```bash
python residual_ifwi.py --config experiments/configs/paper13_ifwi.yaml --sigma 22 --device cuda:0 --output-dir outputs/sigma22/paper13_ifwi
python residual_ifwi.py --config experiments/configs/paper13_fr_ifwi.yaml --sigma 22 --device cuda:0 --output-dir outputs/sigma22/paper13_fr_ifwi
```

## 换初始模型或协议前先检查

只做一次正演与反传，13 炮 CPU 每个 σ 约十几秒：

```bash
python -m experiments.initial_gradient_report --config experiments/configs/paper13_ifwi.yaml --sigmas 15 22 --device cuda:0 --output-dir outputs/initial_gradient
```

输出 `initial_gradient.csv/json`。49 炮、5 s 协议的梯度在 CPU 上内存需求大，宜在 GPU 上运行；该协议下 σ = 15 的 baseline 能收敛（438 m/s），与 13 炮不同，未检查其第一步方向。
