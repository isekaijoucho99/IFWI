# 本地完整仓库安装核验

日期：2026-10-30（Asia/Shanghai）。

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

原因是 `experiments/residual_ifwi_report.py::velocity_metrics` 已返回 `r2` 字段，而旧断言仍要求结果字典仅有 `rmse_mps`、`mae_mps`、`ssim`。这些失败发生在安装前；相关实现与测试在本次提交中均未修改。本次未修复该已有问题，也不声称原仓库全部回归通过。

## 结果边界

本次补充了本地完整 checkout 的新增数值与真实正演桥接证据；未运行 Marmousi/Overthrust 全尺寸反演、地质 DFM 正式训练或 GPU 性能基准。原始交付记录中的研究限制仍适用。
