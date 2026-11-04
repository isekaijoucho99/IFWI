# 本地完整仓库安装核验

日期：2026-10-30（Asia/Shanghai）。

最新回归：2026-11-01（Asia/Shanghai）。原有 4 项指标测试失败已修复，GPU 批量求解已恢复；最新完整回归为 406 项通过、2 项跳过。2026-10-30 的安装过程与较早回归结果保留作历史记录。

## 来源与提交范围

- 目标仓库：`isekaijoucho99/IFWI`。
- 发布分支：`feature/ifwi-five-ideas-core-20261030`。
- 基提交：`226be2262377a0bbde4f813205ceb7cbcd69f335`，对应 `feature/g2ifwi-gpu-batch-tuning`。
- 来源：用户提供的 `IFWI_five_ideas_core_20261030.zip` 及三个独立附件。
- 安装前已核对交付清单内全部 39 个文件的 SHA256；三个独立附件与 ZIP 内对应文件逐字节一致。补丁应用检查通过，安装后的原始 39 个文件也通过反向补丁检查。
- 研究源码、配置和测试保持交付内容；仅为两个说明文档补充历史状态说明，并新增本记录。原有仓库文件未修改。

## 本地环境

Windows；Python 3.10.18；PyTorch 2.9.0.dev20250904+cu128；NumPy 2.2.6；pytest 8.3.4。

沿用现有 Python 环境，未重新安装或替换依赖。以下测试设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`MPLBACKEND=Agg`、`PYTHONUTF8=1`。

## 新增测试

```text
python -m pytest -q -ra verification/five_ideas --junitxml=outputs/five_ideas_publication/new_tests.xml
57 passed, 2 skipped in 27.82s

python -m compileall -q ifwi_research tools/publish_five_ideas.py verification/five_ideas
exit code 0
```

`test_repository_bridge.py` 的 6 项测试全部执行并通过，包括原 RNN-FD 编码等价性、baseline 更新一致性，以及 routing、wavelet、interface、gradient_attention 的真实正演反传。

两项跳过仅为当前 Windows 不允许创建文件/目录符号链接：`test_publish_rejects_existing_dangling_symlink` 和 `test_publish_rejects_destination_symlink`。测试报告保存在忽略的 `outputs/` 下，不提交生成文件。

## 原有回归套件的基线问题

在复制任何交付源码之前，对上述固定基提交执行：

```text
python -m pytest -q verification
4 failed, 323 passed, 2 warnings in 169.89s
```

失败均来自 `verification/residual_ifwi/test_reporting.py`：

- `test_constant_offset_reports_physical_rmse_and_mae_without_clipping`。
- `test_identical_velocity_fields_have_unit_ssim_and_zero_errors` 的 3 个 shape 参数用例。

原因是 `experiments/residual_ifwi_report.py::velocity_metrics` 已返回 `r2` 字段，而旧断言仍要求结果字典仅有 `rmse_mps`、`mae_mps`、`ssim`。这些失败发生在安装前；2026-10-30 的发布未修改相关实现与测试，当时未声称原仓库全部回归通过。后续修复与完整回归结果如下。

## 2026-11-01：指标测试修复与完整回归

在原发布提交 `c832ed9122f43203a497b2566663855b7f1340eb` 上再次运行指标测试，复现了同样的 `4 failed, 12 passed`。修复保留精确的字段与字典断言，将已有的 `r2` 纳入契约，并覆盖以下独立数值预期：

- 非常量真值与预测完全一致时，R² 为 1。
- 预测等于真值均值时，R² 为 0。
- 对 `0..34` 加平移的真值施加 800 m/s 偏差时，真值总体方差为 102，R² 为 `-6273.509803921569`；允许合法负值。
- 常量真值下，无论预测是否完全一致，R² 均保持未定义的 `None`，可用严格 JSON 序列化为 `null`。

仅更新指标测试与函数说明，未改变指标计算、物理求解器或训练行为。对测试与函数说明的只读代码审查未发现需要修改的问题。

沿用上述环境与线程设置，在项目根目录执行：

```text
python -m pytest -q verification/residual_ifwi/test_reporting.py --junitxml=outputs/metric_regression_20261101/reporting.xml
19 passed in 1.12s

python -m pytest -q -ra --junitxml=outputs/metric_regression_20261101/full_regression.xml
387 passed, 2 skipped, 2 warnings in 159.83s

python -m compileall -q experiments/residual_ifwi_report.py verification/residual_ifwi/test_reporting.py
exit code 0
```

完整回归无失败，包含原有回归套件、五个研究原型的新增测试与全部 6 项真实 RNN-FD 桥接测试。两项跳过仍仅因为 Windows 符号链接权限。两项警告分别为 PyTorch 的旧 TF32 设置弃用提示，以及原 `ifwi_modules.py` 梯度裁剪收到空参数生成器的提示，与 2026-10-30 的原有回归警告一致。

修复前、定向测试与完整回归的 JUnit XML 均保存在忽略的 `outputs/metric_regression_20261101/`，不提交生成报告。

## 2026-11-01：恢复 GPU 炮与编码源批量求解

原适配器强制 `shot_batch_size=1`，编码源也逐个求解。现在继承 baseline 的炮批设置，并支持 `--shot-batch-size` 显式覆盖；`null` 表示全部物理炮。路由的选中编码源使用相同批量上限，通过原 FD 的独立炮通道求解，每个通道的源注入带上该编码的权重。

原始 `rnn_fd.py`、物理网格、子波、频率日程、采样分布、优化器与 PDE 源求解计数定义均保留。路由能量分数保持逐编码的未加权损失，重要性权重只用于合成目标与梯度。批大小进入运行身份，batch1 仍可用于低显存或参考路径。

新增 19 项测试覆盖物理炮和编码源批量的目标/梯度等价性、逐编码能量、整批和尾部小批、自由表面开关、低通和全频目标、真实求解器调用分组、无梯度评估、参数校验与 CLI 恢复身份。独立只读审查未发现需要修改的问题，并额外检查了编码顺序/子集、缓存复用、不同源深度和两种边界模式。

在本机 RTX 5060 Ti（16 GB）上，以真实 Marmousi 94×288、15 m、13 炮、1000 时间采样和 3 Hz 阶段执行 CUDA 检查。只评估目标与速度梯度，**优化器更新数与新增基线训练步数均为 0**。生成观测与旧缓存逐值一致。

| 求解路径 | 耗时（秒） | 峰值 allocated（GiB） |
|---|---:|---:|
| 物理 13 炮批量正反演 | 11.624 | 9.457 |
| 同一组选中 3 编码源，串行正反演 | 19.378 | 1.499 |
| 同一组选中 3 编码源，批量正反演 | 6.188 | 2.834 |
| 全 13 编码源批量审查 | 12.140 | 9.509 |

同一组选中编码源批量正反演比串行快 3.132 倍；该计时不含网络 VJP 和 Adam，不能代替完整训练的端到端测速。测量过程最高 reserved 显存为 12.111 GiB。

- 选中编码源批量/串行的目标相对误差：`6.94e-8`。
- 对应速度梯度范数相对误差：`2.85e-6`。
- 对应逐编码能量最大相对误差：`8.51e-8`。
- 全编码审查相对全物理炮的目标误差：`2.57e-6`。
- 对应速度梯度范数相对误差：`1.24e-5`。

完整命令与结果：

```text
python -m pytest -q -ra --tb=short --junitxml=outputs/gpu_batch_validation_20261101/full_regression.xml
406 passed, 2 skipped, 2 warnings in 187.93s

python -m compileall -q ifwi_research verification/five_ideas/test_gpu_batching.py
exit code 0
```

跳过与警告原因同上。CUDA 测速脚本和原始结果保存在工作区外层的 `outputs/five_ideas_gpu_batch_validation_20261101/`。研究队列继续使用种子 3、四个新方案各 1000 步，直接复用旧基线；慢速试跑输出保留，未生成检查点的中间状态不冒充精确续训。

## 2026-11-04：DFM 官方核心适配

固定采用作者仓库 `merry7cherry/drift-flow-matching` 的提交 `619663f0e7cbd17e58d98501daaf389d2810077b`。六个 Sinkhorn/损失函数原函数体未修改，MIT 许可与来源已随代码保存。修复逐行 softmax 替代官方 Sinkhorn、端点噪声配对、时间分布、EMA 和验证/推理采样不一致；详见 `DFM_UPSTREAM_AUDIT.md`。物理求解器、13 炮批量和基线保留。

新增 14 项测试，覆盖直接执行官方函数生成的数值夹具、停止梯度、双边边际、组隔离、端点配对、时间采样、固定条件和噪声起点、旧权重拒绝、EMA 生成验证，以及续训源身份包含 vendor 子目录。初次新增测试 8 项预期失败；补充的源身份保护测试也先失败，再修复通过。

```text
python -m pytest -q verification/five_ideas/test_dfm_upstream.py --tb=short
14 passed in 1.62s

python -m pytest -q -ra --tb=short --junitxml=outputs/dfm_upstream_validation_20261104/full_regression_final.xml
420 passed, 2 skipped, 2 warnings in 120.70s

python -m compileall -q ifwi_research verification/five_ideas/test_dfm_upstream.py
exit code 0
```

两项跳过和两项警告均与原回归相同。独立只读审查未发现需要修改的问题；独立重跑新增 14 项测试通过，另核对了六个函数 AST 与上游源文件散列。

本机 CUDA 验证：

- 六个函数 AST 与固定提交原文件一致；CUDA 漂移结果与直接执行上游函数逐值相同，绝对差为 0。
- 对已有独立地质数据的 4 个训练模型和 2 个验证模型执行 64 步 CUDA 接口试跑，完成 v2 权重保存/冻结加载与 94×288 候选生成。该权重只用于接口检查，验证 RMSE 约 1676.72 m/s，不当作有效先验。
- 真实 Marmousi 94×288、15 m、13 炮、1000 时间采样上执行 2 步 DFM 桥接。此接口检查临时在第 1 步启用先验；正式配置仍为 201 步。两步均通过实际物理候选审查，目标分别从 `0.0067747422` 到 `0.0063568056`、再到 `0.0057603316`。耗时约 35.06 秒。
- 新增基线训练为 0 步。2 步桥接不用于精度或加速比结论。

上游源文件、CUDA 原始记录和桥接输出保存在外层 `outputs/dfm_official_audit_20261104/`；完整回归 XML 位于工作树忽略的 `outputs/dfm_upstream_validation_20261104/`。旧权重与旧实验记录保留。修正后的完整反演效果需用新训练的 v2 权重验证。

## 原始交付结果边界（2026-10-30）

原始交付核验仅补充了本地完整 checkout 的数值与真实正演桥接证据，当时未运行 Marmousi/Overthrust 全尺寸反演、地质 DFM 正式训练或 GPU 性能基准。后续核验与运行情况见上述日期记录；原始交付不构成有效性结论。
