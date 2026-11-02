# 验证与复核

最终CPU回归：

```
python -B -m pytest verification/rom verification/residual_ifwi -q -p no:cacheprovider
211 passed, 1 warning in 48.43s
```

唯一warning是原ifwi_modules.py空梯度裁剪generator，未修改该文件。
新增验证包括一至三阶残差恒等式、三阶改善、精确有限FFT实伴随/DC/MSE尺度、
源质量变化、源/正演子空间包含、稀疏设备投影、留出完整空间残差、策略触发/
MemoryError回退、并行与串行数值一致、原生LU创建/析构线程一致性及错误隔离。

CPU小网格先执行；GPU串行五组、修复后的并行三组均完成每组5次真实Adam更新。
保存的实际ROM buffers在训练全部结束后独立核验，没有重新建更好的基替代原基。
第一轮五组30个状态、第二轮三组18个状态均保留，包括不通过门槛的状态。

并行优化初次运行触发MemoryError；实际错误与中间进度保留，未作为成功结果。
针对SciPy1.15.3原生线程所有权的CPU对照复现和修复后内存记录见measurements/
lu_ownership_before.json与lu_ownership_after.json。没有修改库或安装新版本。

实现经独立只读数学/代码复核。当前入口每步原子checkpoint计入在线时间；GPU
运行结束后额外去除了导出阶段对同一checkpoint的直接重复写入，该I/O修正也在
最终回归范围内。它不改变本轮已记录的任何在线更新、物理结果或时序。

每组均有完整配置、源码快照和SHA256。两次完成GPU运行的source_hashes_unchanged
均为true；实际执行的版本分别为abab0c5与1dae8ed。串行首轮与后续轮的checkpoint
记录方式存在工程改动，因此跨轮耗时差不能全部归于并行。报告使用各轮同期FDTD
作主比较。原始正演器、MSE、SIREN、基线配置保留证明见PRESERVATION.json。

初始共享缓存最终释放只包含在experiment_wall_seconds；不摊入每个方法的初始
准备总量，避免把共同清理重复计数。新旧LU更换时的释放则始终属于在线时间。
核验、导出、初始准备和在线时间的实际值全部单列，未用阶段任务时间之和替代
包含同步与失败尝试的全循环墙钟。
