# 验证记录与剩余边界

## 源码与环境

上游基点 `226be2262377a0bbde4f813205ceb7cbcd69f335`。通过该提交 GitHub Actions 的 `ifwi-source` artifact 获取源快照；所有原有已核对源码、配置、测试文件保持哈希不变。新增文件为 `research_ideas/`、`verification/research_ideas/`、本目录文档、设计/实现计划及独立 CI。

本地：Linux，Python 3.13，PyTorch 2.10.0+cpu，CUDA不可用。OMP/MKL/OPENBLAS线程数均1。执行者在独立本地快照实现；没有修改用户工作目录或现有远程分支。

## 新增测试

`python -m pytest verification/research_ideas -q`：**43 passed**，本轮最终独立运行18.42秒。执行过 RED→GREEN：先缺模块/功能失败，再实现；review新增测试捕捉并修复了路由观测能量归一化、非法先验归一化尺度、完全重复测试真值泄漏、单模式DFM多付另一个模式成本等问题。

覆盖：参数空间冲突投影/范数界、零物理梯度、实际Adam位移回溯及拒绝回滚；Haar奇偶网格往返及梯度；有序界面、同背景初值和位置梯度；分组DFM与自身分布零漂移；原生尺度patch覆盖；路由估计器枚举期望与重复样本权重；物理gate有界且不对输入梯度构造二阶图；原RNN-FD方向导数有限差分、模型空间梯度及参数VJP与原Adam更新；六类runner；路由/小波/界面/gate/DFM续训；先验训练/载入smoke限制；配置/数据变化拒绝；实验矩阵与Pareto集合。

`python -m compileall -q research_ideas verification/research_ideas`：通过。

## 全仓库回归

`python -m pytest -q`：**364 passed, 6 failed, 1 warning**，82.32秒。新增43项全部通过。以下6项在变更前基线也失败，原测试及源码未改：

1. `verification/residual_ifwi/test_experiment.py::test_shot_accumulation_preserves_full_objective_and_adam_update`
2. `verification/residual_ifwi/test_reporting.py::test_constant_offset_reports_physical_rmse_and_mae_without_clipping`
3. `verification/residual_ifwi/test_reporting.py::test_identical_velocity_fields_have_unit_ssim_and_zero_errors[shape0]`
4. `verification/residual_ifwi/test_reporting.py::test_identical_velocity_fields_have_unit_ssim_and_zero_errors[shape1]`
5. `verification/residual_ifwi/test_reporting.py::test_identical_velocity_fields_have_unit_ssim_and_zero_errors[shape2]`
6. `verification/residual_ifwi/test_stochastic.py::test_selected_microbatches_preserve_full_selected_gradients_and_adam_state`

四项reporting断言没有包含已有的r2字段；两项对不同炮microbatch计算的浮点结果要求逐位一致，在本CPU环境失败。不能把此仓库描述为“全部测试通过”。独立CI只运行新增43项，不掩盖或改写旧失败。

## 审阅与明确取舍

本轮为作者自审，没有第二位独立审阅者。检查范围包括梯度空间、偏差与成本计量、数据泄漏、恢复身份、随机数隔离、状态回滚和配置静默忽略。

选择独立runner而非修改旧runner：减少影响范围，代价是旧checkpoint和所有既有加速组合不支持。选择有序界面而非完整自由SDF：适合层状结构并便于验证，代价是不能表达悬垂/分叉。DFM使用小CNN而非大视觉模型：降低工程门槛，代价是先验表达能力必须独立验证。源码未做完整静态类型标注，亦未测多GPU。

尚未完成：合格地质DFM预训练、全尺寸Marmousi/Overthrust对照、CUDA峰值/吞吐、跨设备逐位恢复、held-out炮检验、异网格观测、一般SDF反演器、等参数普通卷积对照、边界HD95/F1分析。完整方案分别列出了这些实验；不能把方案中的待实现对照算作已有结果。

新增测试证明的是所覆盖输入上的数值与软件行为，不证明反演精度、加速、论文新颖性或GPU显存一定安全。
