# IFWI 深层实验框架修复计划

> 历史计划（2026-09-29；执行记录截至 2026-09-30）：下面的任务、权限边界和验证状态属于当时的工作记录，不是当前执行指令，也不表示本次发布重新运行了测试或训练。计划中的四层 256 宽、seed=42 方法对照与当前单次原版参数入口不同；当前用法见[实验使用说明](../../../experiments/README.md)。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让新分支完成真实训练、正确保存和恢复状态，并使每项改进的实现、命名和对照实验含义一致。

**Architecture:** 保留原作者四个模块，在 experiments 的派生训练入口处理训练状态、有限值检查和速度场梯度预条件。将工程修复与方法修正分阶段验收，暂不引入新网络架构或替换正演器。

**Tech Stack:** Python 3.10、现有 PyTorch/CUDA、NumPy、PyYAML、scikit-image；使用已创建的 env，CPU unittest 与 RTX 5060 Ti 短跑。

**执行状态（2026-09-30）：** 用户批准后，任务 1–5 已实现并经独立审查；L1、L2、L3 已通过，L4 尚未执行。详见 ../../repair-validation-20260930.md。以下保留原始计划及逐项验收描述；未逐项勾选的细节不应被解释为逐条独立测试均已执行。

## 约束

- 原作者 ifwi_modules.py、rnn_fd.py、generator.py、plot_functions.py 不修改；原始数据及历史结果不覆盖。
- 当前分支为 feature/deep-layer-improvements-push，基准提交 3ae2aad91eb70c094e44bf4c7e9414ce55701c41。
- 五组正式对照均保留 94×288、13 炮、1000 时间点、dt=0.0019、四层 256 宽网络；这是新分支基线，不称为原 128 宽网络实验的逐值复现。
- 首轮修复短跑采用固定 seed=42，无 Dropout、无额外混合精度和加速。
- 默认不做梯度裁剪，显式 clip_grad > 0 才开启；启用时所有对照组使用同一设置并记入配置。不能为填补缺失属性而悄悄引入 0.25 裁剪。
- 原制定方案阶段不修改代码；后续已获用户批准执行修复及短跑。本轮未启动 4000 步实验，未提交或推送代码。

## 重点检查

- 更新后的网络参数确实变化，且有限；不以“无异常退出”代替有效训练。
- 改进网络替换后，优化器必须包含当前网络的全部且不重复的可训练参数。
- 空间深度轴与神经网络层号、权重矩阵轴严格区分。
- best、last、loss 的时间点一致；中断后可恢复同一下一步。
- 非有限波场不能先被原方法替换成零后才检查；失败运行不得进入结果比较。

## 阶段一：修好运行与结果记录

### 任务 1：打通真实单步更新

**修改：** experiments/run_experiment.py、experiments/configs/*.yaml。
**新增测试：** tests/test_experiment_training.py。

**接口：** `ImprovedIFWI.train_one_epoch(...)` 保留返回格式；裁剪从 `training.clip_grad` 读取，缺省 null。参数列表在最终网络和优化器创建后，从 optimizer.param_groups 展平获得。

- [ ] 添加小网格、真实波动正演的失败测试，复现 params 缺失；测试不伪造 shots 或 backward。
- [ ] 在外层训练中显式调用 vel_net.train()；zero_grad 后使用当前优化器参数，不保留一次性 parameters() 迭代器，不引用被替换网络的参数。
- [ ] 加入配置校验：迭代数/记录间隔为正，裁剪仅 null 或正有限值；损失、参数梯度和更新后参数均检查有限性。
- [ ] 在 experiments 的派生路径接收正演器原始输出并检查有限性，绕开原 forward_process 的 NaN/Inf 置零逻辑。差分正演公式、边界和数据不变。
- [ ] 验证单步后参数改变、梯度有限；无改进且关闭裁剪时，与相同初始化的参考 MSE/Adam 单步比较。CPU float32 的参数/预测采用明确 allclose 容差（rtol=1e-5、atol=1e-6），失败先定位，不扩大容差掩盖差异。
- [ ] 执行 `env/Scripts/python.exe -m unittest discover -s tests -p test_experiment_training.py -v`；运行基线完整网格 3 步，保存损失和实际更新数。此阶段不声称其余改进已正确。

**验收：** 实际完成 3 次有限的参数更新；原作者模块哈希不变。

### 任务 2：修复 checkpoint、best 和 resume

**修改：** experiments/run_experiment.py、experiments/improved_modules/optimizers.py。
**新增测试：** tests/test_experiment_resume.py。

**接口：** checkpoint 使用 completed_updates（完成更新数），保存 model、optimizer、scheduler、best_state、best_loss、Python/NumPy/Torch/全部 CUDA RNG、配置及数据/代码哈希。scheduler 增加 state_dict/load_state_dict。

- [ ] 写失败测试：保存 best 后继续更新，best 张量不得变化；连续 4 步与 2 步后恢复到第 4 步应匹配。
- [ ] best_state 深拷贝到 CPU；last 与 best 分别保存。每个评估/保存时刻重新计算更新后目标损失，用该损失评价同一时刻的权重；不用更新前 loss 为更新后权重排名。
- [ ] 保存到临时文件后原子替换；每轮记录 loss_before_update，评估时单独记录 loss_after_update。记录序号使用 completed_updates，旧 epoch 从零计数不能直接混用。
- [ ] 实现 --resume：校验数据、网络、损失、优化器和调度计划；恢复后 epochs 表示最终总更新数。续训写新目录；不兼容旧 checkpoint 明确拒绝，不能静默重新开始。
- [ ] 保留初始、最终及 best 速度，保存 status.json、失败原因、随机种子、依赖版本；评估后恢复原训练模式。
- [ ] 执行 `env/Scripts/python.exe -m unittest discover -s tests -p test_experiment_resume.py -v`；CPU 无 Dropout 下比较恢复轨迹及 LR，GPU 不要求逐位一致。

**验收：** best 快照不漂移；CPU 恢复重现同一下一步；失败和成功输出可明确区分。

## 阶段二：修正改进项的科学含义

### 任务 3：将“深度加权损失”改为速度梯度预条件

**修改：** experiments/run_experiment.py、experiments/improved_modules/losses.py、experiments/improved_modules/optimizers.py、相关 YAML。
**新增测试：** tests/test_velocity_gradient.py。

**决策：** 不把炮集的时间轴当成深度轴，也不对 [256,2] 等网络权重梯度广播 94 个深度权重。保留普通炮集 MSE；在参与波动正演的物理速度张量 `[batch,nz,nx]` 上加权其反传梯度。命名为“深度梯度预条件”，不声称改变了标量数据损失。

**接口：** `make_depth_weights(nz, config, device, dtype) -> Tensor[nz]`；`attach_velocity_preconditioner(v_wave, weights) -> RemovableHandle`。权重配置放在单一 `gradient_preconditioner` 节点；旧 loss/optimizer 双重开关迁移或报出明确错误。

- [ ] 写测试：nz=8 的分段权重原始值为 [1,1,1,1,2,2,5,5]，归一化为均值 1；测试广播、设备/dtype、非法权重和 nz 不匹配。
- [ ] 将速度分为同值两条分支：`v_wave = v.clone()` 进入原正演器，v 保留给先验。只在 v_wave 上注册 hook，数据梯度乘 w(z)，先验梯度不被重复加权。
- [ ] 去掉对 self.params 的 GradientModifier 循环；每轮 hook 在 backward 后移除，异常时也清理，避免重复注册造成权重幂次累积。
- [ ] 验证恒等权重与基线 loss/梯度一致；非恒等权重下前向炮集和标量 MSE 不变，速度分支梯度严格按权重改变，网络参数梯度通过链式法则改变。
- [ ] 对小网络手算/显式 autograd 比较加权后的参数梯度，不拿加权梯度与原标量 MSE 的有限差分相等作为标准；这是预条件步骤，不是原 MSE 的普通梯度。
- [ ] 执行 `env/Scripts/python.exe -m unittest discover -s tests -p test_velocity_gradient.py -v`，再完整网格 3 步。记录梯度范数及权重分布，不以短跑证明深层恢复改善。

**验收：** 不再有 256 与 94 维度错误；加权效果在正确的空间深度轴上可测；无改进时可退化到基线。

### 任务 4：统一注意力、学习率与先验的语义

**修改：** experiments/improved_modules/networks.py、optimizers.py、losses.py 及相关 YAML。
**新增测试：** tests/test_improvement_semantics.py。

- [ ] 测试并修复网络工厂：传递 outermost_linear、bias、use_attention；默认输出层保持线性。关闭 attention 后，加载同一组 SIREN 线性层权重，应匹配原网络输出。
- [ ] 注意力使用固定模型深度范围归一化坐标，不随输入坐标子集改变。测试同一点单独预测与整网格预测一致；use_attention=false 必须真正关闭注意力。
- [ ] 将 depth_adaptive 名称改为 layerwise_lr，说明这是网络层分组，不能直接称作地下深层学习率；验证所有可训练参数恰好属于一个参数组。旧名称只作为有明确提示的兼容别名。
- [ ] 修复 warmup/cosine：首次 optimizer.step 前设置本步 LR；eta_min 是绝对学习率，余弦段使用 `eta_min + (base_lr-eta_min)*(1+cos(pi*progress))/2`。校验总步数大于 warmup，测试首步、warmup 终点、余弦终点和恢复后的 LR。
- [ ] 先验约束独立成实验开关，避免把先验与深度梯度加权混在同一个“单项实验”。速度单位明确为 m/s；单调性项仅作用深层并默认关闭，因为 Marmousi 不保证速度单调随深度增加。
- [ ] 执行 `env/Scripts/python.exe -m unittest discover -s tests -p test_improvement_semantics.py -v`；用参数分组清单和前向一致性结果验收，不依赖“创建对象成功”。

**验收：** 配置实际生效，组名与算法含义一致；注意力关闭路径、学习率边界和先验作用区域可验证。

### 任务 5：重建可比较的配置与评估

**修改：** experiments/configs/*.yaml、experiments/compare_results.py、experiments/improved_modules/evaluate_deep.py、test_framework.py、两个实验 README、experiments/requirements.txt。
**新增：** experiments/configs/prior_only.yaml、tests/test_experiment_results.py。

- [ ] 固定基线、仅梯度预条件、仅注意力、仅 layerwise_lr、仅先验五组单项实验；组合组明确列出叠加项。同一 seed 的可比组共用同一观测炮集、归一化、预算和网络主干。
- [ ] 同时报告数据 MSE、各先验项、全域/深层/角落误差、SSIM、耗时和参数量。真值只用于生成基准观测及报告误差，不用真值误差挑 best 或自动调参。
- [ ] 先验组的 total loss 与无先验组不作为同一数值指标直接排序；跨组比较使用同一 data MSE 与外部评估指标。注意力额外参数量明确报告。
- [ ] 比较工具按实验和 seed 分组，拒绝不匹配的数据/几何配置，仅汇总 state=completed 的运行；避免同名实验覆盖，单行 CSV 也能读取。
- [ ] 补充评估边界测试：常量真值、空/过小深层区域、NaN/Inf、同 seed 重复目录；未定义指标明确标记或拒绝，不静默写出误导数值。
- [ ] 落实 save_plots 开关；图使用相同色标和物理坐标，输出原始速度和残差，不用平滑掩盖误差。
- [ ] 删除或明确标记文档中的未验证提升比例和示例结果；补充 torchvision/Pillow 等实际直接依赖、Windows UTF-8 命令及真实训练测试说明。
- [ ] 执行 `env/Scripts/python.exe -m unittest discover -s tests -p test_experiment_results.py -v`，再运行全部测试。

**验收：** 对比表不混入失败/不匹配实验；一组仅改变所声明的因素；文档不把预期当实测。

### 任务 6：分级试跑与进入正式实验的条件

- [x] L1：全部 CPU 小网格回归测试通过，覆盖真实正演、反传、更新、保存、恢复；原六项 mock 测试仅作辅助。
- [x] L2：每组完整网格 GPU 3 次更新全部通过；loss/梯度/模型有限，checkpoint 和评估文件可读，best/last 含义正确。
- [x] L3：基线和各单项各 100 次更新，记录损失、速度范围、梯度、每步耗时和显存峰值；结合实测耗时评估长跑预算。100 步只判断流程和稳定性，不保证收敛，也不要求 loss 每步单调下降。
- [ ] L4：通过前三层后另行安排正式 4000 次对照，建议至少 seed=3、42、123，报告均值和波动。先基线与单项，再根据证据选择组合；不将 combined_best 的名字当作已证实最优。
- [ ] 各项代码变更按独立任务审阅；提交或推送遵循执行时用户授权，不在本方案阶段操作远程仓库。

## 最小交付与预期

优先完成任务 1–2，得到可信的可运行基线；任务 3–5 完成后，才具备测试“改进是否有效”的条件。任务 6 每级通过后再进入下一级。

运行错误修复有明确验收条件；深层精度提升是实验问题，没有预先保证。若预条件或注意力未优于基线，应如实保留结果，而非为了得到好看指标继续无记录地调参。
