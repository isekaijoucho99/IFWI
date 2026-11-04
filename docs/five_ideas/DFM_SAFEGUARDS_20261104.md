# DFM 安全性修复与 CPU 验证（2026-11-04）

## 范围与基底

- 仓库：`isekaijoucho99/IFWI`。
- 基底：`feature/ifwi-five-ideas-core-20261030`，提交 `e26452c6d629ca3316a800944a287b5b1a12b1e4`。
- 本次只修复优化器恢复与 checkpoint 资格保护，补足真实 RNN-FD 的 DFM 桥接测试。不选择新的先验架构、不改 DFM 数学目标、不加入 attention。
- 在独立 Linux 云端克隆中使用 Python 3.12.14、PyTorch 2.10.0+cpu、NumPy 2.2.6、pytest 9.2.0。未使用 GPU，未运行正式地质训练或全尺寸反演。

## 1. 用真实 Adam 复现并修复停滞

令 `f(x)=x²`、初值 `x=1`、Adam 学习率 `0.1`，每次最多 5 次减半审查。原实现第 12 次更新到达 `x=-0.002877172124205969`；随后第 13–16 次更新全部被拒绝，Adam 时钟一直停在 12。原因是旧动量产生上升位移，单纯缩小学习率不能改变方向，完整回滚又保存了同一个旧动量。

修复仍先尝试原优化器步。全部失败后，从同一快照对原始梯度下降步进行回溯；研究引擎显式传入**纯物理 VJP**，而不是混入先验的梯度。每个候选都必须通过有限性、物理有效性与实际同一目标的严格下降检查。成功的恢复步重置对应参数的优化器历史；失败或异常仍完整回滚参数、缓冲区、梯度、优化器与 RNG。不接受上升步。

同一真实 Adam 复现中，第 13 次更新恢复成功，物理损失从 `8.27811943230789e-6` 降至 `5.297996436677048e-6`；第 16 次降至 `1.3888379778962682e-6`。这只证明优化器故障已修复，不是地质反演效果实验。

`max_trials` 现在分别约束优化器与恢复两阶段，总试探最多两倍。新增 `optimizer_trials`、`recovery_trials`、`recovery` 字段，保留旧报告字段；真实额外目标评估计入 `forward_calls`。预算对照必须计入这些额外求解，不能只比较更新数。

## 2. checkpoint 默认不自动获得正式使用资格

训练器无论训练多少步，都输出 `toy_only=True` 与 `qualification.status="unqualified"`。默认 loader 要求明确的 `toy_only=False`、`qualification.status="reviewed_for_inversion"` 以及非空审查证据引用；缺失或不完整的旧 metadata 不再被默认为合格。

这是一份研究者声明，不是自动科学认证。加载器不读取证据报告，也不判断结果是否可信。实际审查要求与范围见 `README.md`。训练时长与去噪 RMSE 不能替代独立数据、适用域和匹配预算 FWI 评估。

Python 中的显式 `allow_unqualified=True` 仅用于带警告的 smoke/tests，不改写资格状态，也不绕过其他检查。正式运行器拒绝配置中的此类覆盖。加载器新增非有限权重拒绝检查。

## 3. 新增回归覆盖

49 个新增测试覆盖：

- 真实 Adam 停滞恢复、恢复步回溯、计数与学习率恢复。
- NaN/Inf、物理无效候选、异常、失败后的模型/缓冲区/梯度/优化器/Python/NumPy/PyTorch RNG 完整回滚。
- checkpoint 缺失或伪布尔标记、缺失/不完整审查、显式 smoke opt-in、非有限权重与 CLI 关闭覆盖。
- 原仓库真实 RNN-FD：冻结 DFM 权重、INR VJP、真实候选物理目标审查、启用前与 baseline 等价、精确续训、无效候选回滚，以及恢复使用纯物理 VJP 而非先验混合梯度。

桥接测试使用明确标记为未合格的一步合成训练 fixture，只验证接口和保护逻辑。它不是可用的地质 DFM 权重。

## 4. 完整测试与 CPU 后端差异

所有测试设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`MPLBACKEND=Agg`。对未修改的固定基底另行解包并运行，不用已经修改的工作树充当基线。

| 代码 | CPU 后端 | 结果 |
|---|---|---|
| 固定基底 | 默认 MKLDNN 开启 | 396 passed，12 failed，1 warning |
| 本次修改 | 默认 MKLDNN 开启 | 445 passed，12 failed，1 warning |
| 固定基底 | 显式关闭 MKLDNN | 408 passed，1 warning |
| 本次修改 | 显式关闭 MKLDNN | 457 passed，1 warning |

默认命令：

```bash
python -m pytest -q -ra --tb=short
```

用于定位后端差异的完整运行：

```python
import torch
import pytest

torch.backends.mkldnn.enabled = False
raise SystemExit(pytest.main(["-q", "-ra", "--tb=short"]))
```

未修改产品的全局后端设置，也未放宽测试容差。默认环境的 12 个失败在修复前后完全相同：

- `test_gpu_batching.py::test_code_batches_match_serial_objective_gradient_and_energies` 的 batch=2/3、cutoff=None/20.0、free_surface=True/False，共 8 项。
- `test_gpu_batching.py::test_physical_shot_batches_match_serial_objective_and_gradient` 的 cutoff=None/20.0，共 2 项。
- `test_experiment.py::test_shot_accumulation_preserves_full_objective_and_adam_update`。
- `test_stochastic.py::test_selected_microbatches_preserve_full_selected_gradients_and_adam_state`。

它们涉及批量/串行求解的数值或逐位等价；关闭 MKLDNN 后全部通过。因此不能将默认环境宣称为全绿，也不能把这些基底已有失败归因于此次 DFM 修改。仍需在实际目标硬件和 PyTorch 版本验证。

唯一警告来自原 `ifwi_modules.py:362` 中空参数生成器的梯度裁剪。`compileall` 与 `git diff --check` 通过。独立代码审查未发现需处理的问题；CPU 原生 DFM 桥接 5 项在默认 MKLDNN 开启时通过。

## 结果边界

本次没有证明 DFM 能降低 Marmousi/Overthrust 的模型误差，没有训练正式先验，也没有验证新 attention、照明条件或其他先验设计。仍需要明确的独立地质训练数据、预训练权重与公平消融实验。上述修改只保证已知数值故障不再阻止下降，以及测试权重不会被静默当作已验证地质先验。
