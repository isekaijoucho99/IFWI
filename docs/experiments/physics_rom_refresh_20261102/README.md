# IFWI 降阶基校正与动态刷新：小规模实测

**结论：三阶校正维持了五次实际更新的梯度精度，但总耗时18.803秒，对同期原
FDTD的16.396秒仍慢14.7%。本轮没有同精度净加速。** 初始准备6.011秒另计。
完整成本解释与后续假设见 [CONCLUSION.md](CONCLUSION.md)。

## Material Passport

- 类型：原始数值实验、实现与失败诊断；基于上一轮 SIAM/PML 降阶分支75abbe8。
- 问题：真实 Adam 更新中能否维持原 FDTD 梯度，并在计入所有在线刷新成本后净加速？
- 范围：Marmousi94×288、单炮、每方法最多5次更新；CPU小网格先行。无新神经网络。
- 结果与原始表格：[measurements/RESULTS.md](measurements/RESULTS.md)、[comparison.csv](measurements/comparison.csv)。
- 研究协议与数学公式：[DESIGN.md](DESIGN.md)。在线残差指标不等于认证误差界。

## 保持的设置

原声学二阶 FDTD/PML、dx15m、dt0.0019s、nt1000、8Hz Ricker、Gaussian sigma15、
原 residual SIREN128×4、omega30、零末层、seed3、Adam1e-4。中央单炮与完整接收线。
原源注入时序、质量/源项速度导数、padding、波形MSE和一次梯度组装后一次Adam均不变。
压力基192维，其中96维受保护正演空间；64个建基频率、32Hz截止、Nfft4096。
这是物理模块的单炮小验证，不是历史13/49炮或长程反演的收敛实验。

新入口的实际循环数由 `refresh.updates` / `--updates` 控制（上限5）；配置中继承的
`experiment.training.epochs` 不参与这个独立入口的循环。所有实际完成步数另行保存。

## 方法与计时口径

比较原 FDTD、冻结基、每步完整重建、缓存LU的一/二/三次残差校正加动态刷新。
旧分解作用于新方程的原、伴随残差；校正后用固定大小 POD 更新基空间。只在
校正残差、8个留出频率的最终空间指标或伴随源一致性不满足既定门槛时重新分解。
原、伴随均保留源质量变化及真实FFT伴随权重，含DC、1/Nfft和MSE尺度。

计时从每个实际更新的速度提取开始，包含传输、校正、正交化/POD、静态投影、
初步正演、伴随源一致性修正、留出监测、完整重分解、失败尝试、原训练更新、
基的CPU内存快照、每步原子checkpoint及回退。还计入在线缓存替换时的释放成本。
全循环CUDA同步墙钟是主口径，不把重叠任务的耗时相加当墙钟。

初始离散算子/缓存/基准备另列，也给出加回初始准备的总时间。新网络和优化器
实例准备、公共观测数据生成、预热单独于在线更新；没有把任何中途重建算作预训练。
每次运行保存全部实际使用的ROM buffers，**所有方法训练完成后**才在保存的状态上
计算FDTD波形、速度及网络梯度；核验不参与在线决策。核验与大文件导出成本单列，
同时给出包含这些成本的总量。最终更新后的状态也核验，但其基为上次更新的基；
若继续下一步，动态算法会先校正/刷新。精度准入依据实际用于更新的五个状态。

速度比较同时检查5步总时间与扣除第一个已准备状态后的4步，避免只靠初始免费
缓存显得快。一个种子、五步的优势也不能推断1000步性能或收敛精度。

## 并行实现的失败与修复

串行五组已完整执行。第一次并行实现把工作线程创建的原生LU交给主线程销毁，
触发本机SciPy1.15.3的线程局部分配登记行为：每次重锚约多留3.6–4GiB，最终
MemoryError。该失败实验未形成完整梯度核验，不能计作加速结果，原日志保留。

CPU对照确认跨线程创建/销毁时RSS约104→156→207MiB，同线程销毁保持稳定。
修复使用4个固定归属线程：各自创建、求解和清理自己的LU，只向外返回数组或
整数，错误也序列化以免traceback带出原生引用。修复后4轮RSS约53.58、53.58、
53.45、53.59MiB。没有修改或升级用户的SciPy环境。

这个限制有原版本源码支持：[分配与释放](https://github.com/scipy/scipy/blob/v1.15.3/scipy/sparse/linalg/_dsolve/_superlu_utils.c#L89-L147)、
[LU析构](https://github.com/scipy/scipy/blob/v1.15.3/scipy/sparse/linalg/_dsolve/_superluobject.c#L109-L123)。
修复的是本实现的LU生命周期，不能因此宣称解决了该库所有线程或长期内存问题。

并行时外层POD/探测BLAS4线程，独立频率任务内部BLAS1线程，4个原生所有者。
实际线程库/有效线程数与版本留在preparation.json。并发是工程优化，物理加速
方法的贡献仍需用同并发配置下的完整重建作消融，不能把开线程本身作为论文创新。

## 内存及数学边界

64个LU的数值/索引估计约4.12GB，另有约142MB原、伴随快照。实际主机驻留、
private commit、进程峰值与CUDA allocated/reserved峰值分别记录。成对实验保留
公共初始缓存，重锚短暂同时持有新旧缓存；这些是实测进程峰值，不是独占部署值。

若T=-A0^-1 D且||T||<1，k次校正的解误差可界为
||T||^k/(1-||T||)乘首次校正范数。没有计算全材料统一的该常数；残差和probe是
启发式指标。二阶校正的失败状态正说明小残差不足以认证反演梯度。
每次VJP期间基冻结；比较的是选定ROM对原FDTD梯度的逼近，不是对整个离散刷新
决策求导。长期轨迹、其他种子、13/49炮、nt2632和最终速度精度尚需独立验证。

## 复现入口

使用现有项目环境；额外依赖列在experiments/rom/requirements-refresh.txt。
所有输出目录必须新建，不覆盖历史数据。

```powershell
python -B -m pytest verification/rom verification/residual_ifwi -q -p no:cacheprovider
python -B -m experiments.run_rom_refresh_pilot --small --device cpu --output outputs/refresh_cpu_new
python -B -m experiments.run_rom_refresh_pilot --device cuda:0 --output outputs/refresh_serial_new
python -B -m experiments.run_rom_refresh_pilot --config experiments/configs/rom_marmousi_refresh_parallel.yaml --small --device cpu --output outputs/refresh_parallel_cpu_new
python -B -m experiments.run_rom_refresh_pilot --config experiments/configs/rom_marmousi_refresh_parallel.yaml --device cuda:0 --output outputs/refresh_parallel_new
```

大型checkpoint和逐步ROM buffers保留在GPU机器独立目录；紧凑配置、哈希、核验
及表格随分支保存。checkpoint不序列化原生LU；恢复必须重新分解并计入恢复成本。
