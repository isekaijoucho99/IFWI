# Warmup400 + cosine：AMFMS-inspired 学习率候选

## 这轮要检验什么

先比较恒定 `1e-4` 与一个预先确定的动态 LR 组合，查看波形 loss、梯度和最终速度结果，再决定是否继续调整。本轮不预设 warmup 一定改善 IFWI。

依据是 *Enhancing seismic inversion fidelity via adaptive multi-frequency and multi-scale fusion*，Knowledge-Based Systems 339 (2026), 115562，[DOI](https://doi.org/10.1016/j.knosys.2026.115562)。其 §2.3、式 (17) 使用总 140 epoch、14 epoch warmup、最大 LR `0.001`、最小 LR `0.0005`。

这里迁移的是约 10% warmup 和末端/峰值比 0.5，保留本项目原 Adam 峰值 `1e-4`。论文的监督式地震反演模型、训练单位和数据条件不同；这不是作者代码复现，也不是已证实适用于 IFWI 的最优参数。组合调度的消融不能单独证明 warmup 的效果。

本轮追加在已交付提交 `030a3d9e59198eaad63743633649143c3bbd7f0e` 之上。运行期 20 个 Python 文件没有改变，比较器继续按原 `22fa517bc9c9217e9c08d47c585a56a916702a91` 的源码清单核验；功能分支提交号与训练源码身份是两层不同记录。

## 批准的固定候选

新增配置：`experiments/configs/modern13_warmup400_cosine_half.yaml`。

- 总预算：4001 次更新
- 第 1..400 次更新：线性 warmup，LR = `1e-4 × update / 400`
- 第 400 次更新达到 `1e-4`
- 随后 cosine 衰减至第 4001 次更新的 `5e-5`
- scheduler 参数：`warmup_epochs=400`、`max_epochs=4001`、`eta_min=5e-5`

| 更新数 | 实际 LR |
|---|---:|
| 1 | 2.5e-7 |
| 399 | 9.975e-5 |
| 400 | 1e-4 |
| 401 | 9.999999048599254e-5 |
| 4001 | 5e-5 |

现有 scheduler 在 Adam 更新前设置 LR，warmup 和 cosine 在峰值处相接；第 401 次更新没有突然跳低。warmup 早期低于 `eta_min` 属于预期，`eta_min` 是 cosine 尾段的最低值。短测试不会重标定 4001 步周期。

固定其余设置：modern 运行器、13 炮、seed=3、四层 128 宽 SIREN、omega=30、随机初始化、原数据和采集几何、纯 MSE、Adam。无裁剪、分炮、先验、预条件或频带变化。作者模块、正演、原训练器和 modern runtime 均保持原样。

## 保留原对照，区分解释

原两个配置保持不变：

- `modern13_constant_lr.yaml`：恒定 `1e-4`
- `modern13_cosine_lr.yaml`：无 warmup，4001 步 cosine 至 `1e-5`

新候选既增加了 warmup，又把末端提高到 `5e-5`。它与旧 pure cosine 的差异不能解释成“只改变 warmup”。本轮主要检验 constant 与新 LR 策略整体；输出 `audit.json` 会列出允许变化的三个配置字段及解释限制。

如果整体方案有收益，再考虑“前 400 步恒定峰值、后接完全相同 cosine 尾段”的控制。这只是后续设计，本轮没有添加或运行该方案。

## 轻量核验

沿用项目兼容依赖与 `pytest`，从仓库根目录执行：

```bash
python scripts/compare_lr_control.py check-configs
python scripts/compare_lr_control.py check-warmup-configs
python -m pytest -q
```

第一条仍只接受旧 constant/pure-cosine pair，第二条单独严格检查 constant/W400 候选。两者均只检查配置，不启动训练、不写输出目录。正式比较时会核验记录的训练源码哈希是否匹配固定清单。

## 后续正式运行与比较

下面的完整训练命令没有在本次 CPU 验证中执行。先核对实际设备、依赖、数据、源码与资源预算，两组用同一环境从头运行，保留旧 suite 只读。

```bash
python experiments/run_experiment.py --config experiments/configs/modern13_constant_lr.yaml --device cuda:0 --output-dir results/warmup400_new
python experiments/run_experiment.py --config experiments/configs/modern13_warmup400_cosine_half.yaml --device cuda:0 --output-dir results/warmup400_new
```

把实际运行子目录传入独立比较入口：

```bash
python scripts/compare_lr_control.py compare-warmup --constant-run results/warmup400_new/CONSTANT_RUN --warmup-run results/warmup400_new/WARMUP_RUN --output results/warmup400_report_new
```

这个入口只需要 constant 与 warmup 两组，不要求先运行 pure cosine。旧 `compare --cosine-run ...` 仍拒绝 warmup 配置，不会静默放宽协议。

输出继续分开：

- `fixed_final.csv`：4001 步固定末模型
- `waveform_best.csv`：按日志评估点更新后波形 MSE 选出的模型
- `loss_grad_lr.png`：loss、参数梯度范数、每步实际 LR
- `audit.json`：输入/配置/分析脚本/源码清单哈希、环境和归因边界

比较器要求同初值、同观测、同源码、同预算和环境；检查完整轨迹、正确 LR、评估节奏及空间 RMSE，拒绝额外参数变化。输入只读，输出必须是新目录。真值只用于评价固定末模型和已按波形 loss 选择的模型，不据真值挑训练步数。首次版本仍只比较从头完成的运行，沿用原恢复元数据边界；详见[原比较器说明](lr_control.md)。

更低 loss 不保证更低速度误差。正式评估必须同时看末步全域/深层 RMSE 和速度结构；一个 seed 的结果只支持后续探索，不能证明普遍有效。

## CPU 验证与未验证项

新增 17 项针对测试；加上原 18 项，首次完整运行 **35 passed，10.43 s**。全部使用隔离 CPU 环境，未执行完整 Marmousi。测试分开验证：

- W400 配置、首步/峰值/末步和 400→401 连续性、分段单调性
- 在完成 399、400、401 步处恢复 scheduler，并继续真实 Adam 到 403 步；模型和 Adam 状态与连续运行逐 tensor 一致
- 真实 8×9 网格、2 炮、24 时间点 FD 的前 6 步 warmup；连续与 2+4 步恢复一致，last/best 重算 loss 与保存记录一致
- W100、399、错误端点、预算、额外字段等非法配置拒绝
- 独立 constant/warmup 报告、错误 401 步 LR 拒绝、原 pair 兼容与严格拒绝 warmup

报告测试中的合成产物只用于验证读取/拒错，不是反演质量结果。恢复边界的 403 步测试采用小型线性网络和真实 Adam；不把它称为 403 步 FD 反演实验。

未验证 GPU、完整 4001 步速度质量、多 seed，以及历史冻结 modern13 与当前源码的完整同版性。未运行作者未公开的旧 148 项测试。
