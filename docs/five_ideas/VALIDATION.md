# 验证记录与交付边界

以下是原始独立交付环境的历史记录。本地完整 checkout 的后续安装核验见 [`LOCAL_VALIDATION.md`](LOCAL_VALIDATION.md)，其中包含真实 RNN-FD 桥接测试结果与已有回归失败说明。

日期：2026-10-30。此记录描述实际运行，不把设计中的实验写成完成结果。

## 1. 环境

Python 3.13.5；PyTorch 2.10.0+cpu；NumPy 2.3.5；pytest 9.0.2；CUDA 不可用。
测试环境是独立添加式交付目录，不是完整 IFWI checkout。GitHub 固定基提交的相关接口通过连接器读取，本地无法下载完整仓库。

## 2. 实际命令与结果

```text
python -m pytest -q -ra
53 passed, 6 skipped

python -m compileall -q ifwi_research tools verification/five_ideas
exit code 0
```

原始测试输出：`validation/pytest_cpu.txt`。

53 项通过覆盖：Haar 重建与 gradcheck、初始场匹配、界面梯度和物理单位、正交源基、采样目标及梯度的期望、正探索与重复抽样、缓存更新不改变速度、门控学习、状态恢复、DFM 双时间恒等性/条件分组/微型训练/冻结权重、先验梯度保护、实际物理候选审查、完整 Adam/RNG 回退、安全 checkpoint、数据划分重复拒绝、CLI、发布清单校验与路径边界。

开发过程中各模块先观察缺失/边界测试失败，再实现或修复。最终运行无失败、无警告。进行了本地自审与新增边界测试；没有独立审查代理，不能将其描述为独立第三方代码审查。

### 跳过的六项

`verification/five_ideas/test_repository_bridge.py`：

1. 原 RNN-FD 的逐炮与全正交编码目标/梯度一致性；
2. 新 runner baseline 与原 `train_update` 更新一致性；
3–6. routing / wavelet / interface / gradient_attention 在原正演中的反传。

原因仅为原仓库不存在。安装进实际 checkout 后这些测试会执行；一旦原文件存在，导入错误或数值不一致不会自动跳过。

**因此没有证据声称“已在你的原始正演完整跑通”或“原仓库全部回归通过”。** 原项目旧测试套件也未执行，不根据历史会话中的失败数推断当前旧套件状态。

## 3. 独立声学 smoke

另行执行 baseline、routing、wavelet、interface、gradient_attention 各 10 步，16×24 网格、4 炮、80 个时间步。五者均完成并得到有限结果。

| 路径 | 正演源求解 | 反传源求解 |
|---|---:|---:|
| baseline | 40 | 40 |
| routing | 31 | 31 |
| wavelet | 40 | 40 |
| interface | 40 | 40 |
| gradient_attention | 40 | 40 |

统计对应实际选择的源/编码源，不是先计算全部炮后减少日志数。此表只验证调用路径与计数，不是目标 GPU 加速结果。完整 toy 输出摘要见 `validation/toy_smokes.json`；其中 RMSE 只能被理解为该极小数值问题的结果。

DFM 在测试中用临时独立小数组训练 1–2 步，验证权重载入、冻结、候选、参数梯度保护和真实 toy 物理审查。这不是有效地质先验，不发布这些临时权重。

## 4. 核心规模（实际按 numel 计算）

WaveletVelocity 额外 643 参数；GradientAttentionVelocity 额外 609 参数；InterfaceVelocity 额外 1,186 参数；宽度 32 的 TwoTimeTransport 共 19,969 参数。

这些是实现规模，不是精度或显存上限。DFM 是小型机制验证网络，可能不足以学习复杂地质分布；正式研究需要容量与训练预算对照。

## 5. 自审中发现并修复的边界问题

- 禁用梯度门控时直接返回基础网络，避免两个大锚点相减的浮点抵消影响“关闭即基线”。
- 拒绝先验归一化元数据 NaN、无效核宽度，以及可广播但不同网格的候选。
- RNG checkpoint 使用安全可加载的基本类型，不依赖 NumPy 任意对象反序列化。
- 发布脚本拒绝路径遍历、符号链接逃逸、现存文件和悬空符号链接覆盖；不 force push。

所有上述边界均有对应回归测试。Windows 不允许创建符号链接的环境可能另外跳过两个路径测试；数值测试和真实 FD 测试不得借此跳过。

## 6. 尚无证据的事项

没有 CUDA 峰值显存、吞吐或 RTX 5060 Ti 实测；没有 Marmousi/Overthrust 全尺寸反演；没有五方向优于 P-MS 的结论；没有跨地质分布先验泛化结论；没有三维/弹性/真实资料结果；没有全部强对照的完成结果。

新原型互相隔离，不支持配置为五模块叠加。部分设计中的强消融、多界面和原论文完整模型明确尚未实现，详见设计文档。

## 7. 原始交付时的发布状态

**未推送 GitHub、未创建远程分支或 PR。** 当前 GitHub 工具只暴露读取操作，执行环境访问 GitHub 的网络不可用。不存在可引用的“新提交 SHA”。

交付 ZIP 与添加式 patch 可安装进固定基版本。`tools/publish_five_ideas.py` 在用户已认证的本地 clone 里建立独立工作树，测试后创建提交；只有显式 `--push` 才发布新分支。发布脚本的路径/清单逻辑已测试；本次没有执行真实网络推送。当前工作区与 main 均不被脚本重置或覆盖。
