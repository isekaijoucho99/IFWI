# 学习率单因素对照：constant 与 cosine

## 本次改动

这是一组待检验的学习率实验，不承诺提高速度反演精度。

- 复用现有 `CosineAnnealingWithWarmup`，不新增调度算法、不修改训练循环
- 仅新增两个配置、源码身份清单、只读核验/比较脚本和专项 CPU 测试
- `experiment.py`、原 baseline 配置、作者核心模块、历史训练器均保持原样
- 实现依据当前 GitHub 提交 `22fa517bc9c9217e9c08d47c585a56a916702a91`
- 当前源码与历史冻结 modern13 的完整身份尚未匹配；旧结果不能直接充当本次新配置的控制组

## 固定设置与唯一变量

两组采用 modern `experiments/run_experiment.py`：13 炮、seed=3、四层 128 宽 SIREN、omega=30、随机初始化、mean=3/std=1、纯波形 MSE、Adam、4001 次更新，每 100 步及最后一步重新评估。相同数据、采集几何、float32、无噪声；无 warmup、梯度裁剪、分炮、先验、预条件或频带递进。

| 配置 | LR |
|---|---|
| `modern13_constant_lr.yaml` | 每步固定 `1e-4` |
| `modern13_cosine_lr.yaml` | 全程从 `1e-4` 平滑降至 `1e-5` |

两个配置都显式记录相同的 `scheduler_params`，唯一有效处理变量是 `optimizer.use_scheduler`。控制组关闭 scheduler 时这些参数不会改变 Adam。

cosine 使用 `warmup_epochs=0`、`max_epochs=4001`、`eta_min=1e-5`。更新编号为 1..4001，scheduler 接收 0..4000；第 1、2001、4001 次更新的 LR 分别是 `1e-4`、`5.5e-5`、`1e-5`。这是从首步开始的全程衰减，不是只在末段开启，也没有重启。

4001 是预定调度周期。短测试修改训练预算不会压缩它；超过该周期会拒绝运行。不要复用 `feature_baseline.yaml` 或 `adaptive_lr.yaml` 作为本对照：前者的网络/seed/预算不同，后者是网络层分组学习率。

## 先做轻量 CPU 验证

使用项目所需的兼容 `torch` / `torchvision` 和根 `requirements.txt`，另安装 `pytest`。建议单独环境；CPU 构建可从 PyTorch 官方 CPU wheel 索引安装。不要把 CUDA 依赖引入只做 CPU 测试的环境。

在仓库根目录：

```bash
python scripts/compare_lr_control.py check-configs
python -m pytest verification/lr_control -q
python -m pytest -q
```

`check-configs` 只读取配置，不训练，也不创建运行目录。它不是完整环境或源码预检。正式结果比较时，脚本会核对两组运行保存的源码哈希，且要求匹配 `lr_control_sources.json` 中固定的训练源码身份。

测试使用独立合成小网格，不需要下载或训练完整 Marmousi。它们验证软件行为，不证明调度能改善模型质量。

## 完整对照的运行方式

以下是后续正式实验命令，不属于本次 CPU 验证已经执行的内容。运行前先确定设备、依赖版本和资源预算；两组必须在相同环境、相同源码上从头运行，输出放在新的目录。

```bash
python experiments/run_experiment.py --config experiments/configs/modern13_constant_lr.yaml --device cuda:0 --output-dir results/lr_control_new
python experiments/run_experiment.py --config experiments/configs/modern13_cosine_lr.yaml --device cuda:0 --output-dir results/lr_control_new
```

每次创建唯一运行子目录。不要在旧 suite 中写入，也不要把旧恒定 LR checkpoint 改配置后续训成 cosine 组；恢复契约会拒绝 optimizer 改变。

如果随后能读取历史冻结源码：先核其完整清单、数据/初始化/配置及 scheduler 实现。若无法证明同版，则保持本次新版本的成对控制实验；不要用历史单个数值代替控制组重跑。

## 比较两个已完成的新运行

把实际生成的子目录传入：

```bash
python scripts/compare_lr_control.py compare --constant-run results/lr_control_new/CONSTANT_RUN --cosine-run results/lr_control_new/COSINE_RUN --output results/lr_control_report_new
```

比较脚本不会训练，不读取 pickle checkpoint，不修改输入运行。输出目录必须全新且位于两个输入运行之外。

它会拒绝：

- 网络、seed、loss、clip、数据、训练预算等额外变化，以及两个配置一起偏离注册基线
- 未完成或非 4001 步的运行，重复/缺失训练记录，错误 LR 序列
- 非预定的评估点，loss-best 时刻/score 与记录不一致
- 源码与固定清单不同、不同 Python/设备/依赖环境
- 初始速度不同、保存数据与其哈希不一致、非有限值或数组形状不符
- 保存的速度全域/深层 RMSE 与独立重算不一致
- 复用输出目录、把报告写入输入运行

首版比较器只接受从头完成的运行。原 runtime 对恢复运行的 `initial_velocity.npy` 语义与旧汇总脚本存在歧义，本次不扩大范围修改它；真实 CPU 测试独立检查了训练状态恢复的一致性。

输出：

- `fixed_final.csv`：预定 4001 步末模型，分别报告波形 MSE、全域/深层 RMSE、深层标准差、水平 TV 与速度范围
- `waveform_best.csv`：在记录点评估中按更新后波形 MSE 选择的模型，指标同上
- `loss_grad_lr.png`：更新前 MSE 曲线、记录点更新后 MSE、裁剪前参数梯度范数及实际 LR
- `audit.json`：完整输入文件哈希、实际配置哈希、分析脚本哈希、源码清单哈希、环境和验证边界

真值只用于评价已确定的末模型和 waveform-best，不用于选择训练步数或调度。报告不根据速度 RMSE 重新挑选“最优 checkpoint”。比较器核对导出的 loss/metrics 一致性，不重新正演 checkpoint；这一点由独立小网格测试覆盖。运行器未导出子波数组，因此报告核对契约中的子波哈希一致性，不声称独立重算其来源。

学习率衰减可能改善训练稳定性，但更低的波形 loss 不保证更准确的地下速度。先看固定末步全域/深层误差及结构，不将单 seed、短验证或更小的梯度直接写成方法优越性结论。

## 本次 CPU 验证记录

本次新增专项套件首次完整通过：**18 passed，8.79 s**。最终复核数值以随分支更新的测试摘要为准。

环境：Python 3.12.14、PyTorch 2.14.1+cpu、torchvision 0.29.1+cpu；基础分析依赖按根 `requirements.txt` 安装。此环境不同于旧实验环境，不保证跨平台逐位一致。

覆盖：

- 严格单因素配置及 modern 路由
- cosine 首尾、中点、单调性、调度状态恢复、超周期拒绝
- 关闭 scheduler 的 Adam 与直接构造 Adam 的更新完全一致
- 8×9 网格、2 炮、24 时间点的真实有限差分；6 步连续训练与 2+4 步恢复的模型、Adam、scheduler、历史和 best 状态逐 tensor 一致
- last/best 权重重新计算的 loss 与保存记录一致
- 比较脚本的完成状态、配置、数据、环境、源码、LR、初值、RMSE、输出路径拒错及输入只读检查

初次小网格夹具用 `expand` 产生了非连续接收器索引，原正演的 `view` 不接受它；已将测试夹具改为与原数据准备相同的 `repeat`。没有修改求解器规避该问题。

未验证：GPU、完整 Marmousi 4001 步、实际精度改善、跨 seed 稳健性、历史冻结源码同版性。原来未公开的 148 项本地测试没有在本次运行，不将新增测试冒充那套历史套件。

## 其他学习率方法：仅作为后续候选

本次没有实现、执行或证明下面的方法有效。优先保留可解释的单因素比较。

1. **指数衰减**：适合作为下一项简单候选。先与 cosine 对齐初始 LR、末端 LR 和总预算，再比较曲线形状。若要求 4000 个衰减间隔后剩余比例 `q=0.1`，应取 `gamma = q**(1/4000)`，约 `0.9994245`。直接复制 `gamma=0.999` 会让末端约为 `1.83e-6`，引入末端强度差异。参考 [PyTorch ExponentialLR](https://docs.pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.ExponentialLR.html)
2. **平台期衰减**：可以利用固定全炮、更新后重新计算的波形 MSE；`patience` 单位必须明确为评估次数，不是更新步数，并预定相对阈值和最低 LR。不使用速度真值调度，也不跨不同频段直接比较 loss。参考 [ReduceLROnPlateau](https://docs.pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)
3. **OneCycle**：默认还会调整 momentum；对 Adam 通常涉及 beta1，因此若要声称“只变 LR”，必须显式关闭 `cycle_momentum`，并另定峰值 LR 的安全范围。本轮不加入。参考 [OneCycleLR](https://docs.pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.OneCycleLR.html)
4. **Warmup / cosine 重启**：先看真实初期不稳定或停滞的证据，再增加相应阶段，避免把多种处理同时塞入首轮对照。当前实现不重启。参考 [CosineAnnealingWarmRestarts](https://docs.pytorch.org/docs/2.9/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html)
5. **L-BFGS**：这是优化器变化，不是单纯 LR 变化；一次外层 step 可多次调用 closure，比较必须同时记录正反演次数及计算时间，不能直接对齐外层步数。参考 [PyTorch LBFGS](https://docs.pytorch.org/docs/2.9/generated/torch.optim.LBFGS.html) 和 [Deepwave FWI 示例](https://ausargeo.com/deepwave/example_fwi)

[IFWI 的一个学习率衰减先例](https://doi.org/10.1093/gji/ggag277)包含其他方法与分频段设置，不能把其结果单独归因于 LR，也不应照搬跨不同训练预算的 decay 系数。这里列出的是候选理由，不是本项目的精度证据。PyTorch 文档链接固定为 2.9 版本说明；本次 CPU 实测环境版本单独记录，不暗示两者是同一构建。
