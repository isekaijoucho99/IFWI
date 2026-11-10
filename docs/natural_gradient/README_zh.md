# IFWI 的阻尼自然梯度实验

## 实现范围与结论边界

本分支提供完整的可运行实验入口，不只是优化器片段：原始残差 SIREN、原版有限差分正演、全炮或随机抽炮、Adam 对照、模型/波形输出、检查点、失败记录和计算成本账本。没有运行大型 Marmousi GPU 实验，因此**尚未证明比 Adam 快或精度更高**。不使用 DFM，不是完整物理 Gauss–Newton，也不是精确复现自然梯度参考论文。

保留 `residual_ifwi.py` 及其 Adam 默认行为。原作者 `ifwi_modules.py`、`rnn_fd.py` 没有改动。新入口只接受原版纯波形 MSE 配置，不支持先验/LoRA 等扩展；旧实验不得直接当作匹配对照。

## 针对什么问题，核心改在哪里

神经表示参数的欧氏梯度会受到参数化的伸缩影响。尝试把更新的局部几何换成归一化速度空间的 L2 度量，而不是重复调整正演或使用测试真值指导优化。

令 m(θ)=b+s q(θ)，其中 m 为 m/s，s=1000×std_kmps，b 为固定初始速度。先用原正演与反向传播获得 g_m=∂L/∂m，再取 g_q=s g_m。冻结本次 θ，定义 J=∂q/∂θ，求解：

    (JᵀJ + λI)p = Jᵀg_q，θ_new = θ − ηp。

JᵀJ 是离散像素 **sum** L2 度量，不除像素数。λ 和 η 不能无条件跨网格复用。λ>0 时系统正定，精确解满足 g_θᵀp≥0，因此负方向在局部一阶意义下是下降方向；截断 CG 检查方向内积与残差，但有限步不保证真实损失下降。不自动线搜索或重启，避免隐藏额外 PDE 成本。

核心代码在 `experiments/natural_gradient.py`：

```python
# 每次外层更新先计算原始波形目标对物理速度的梯度。
g_m = autograd.grad(loss, velocity)[0]
g_q = scale * g_m
# JVP/VJP 只经过神经表示，不经过波动方程。
_, Jp = jvp(q_fn, (theta,), (p,))
product = pullback(Jp)[0] + damping * p
# CG 得到 p 后直接更新参数，不再交给 Adam。
```

`ResidualSirenView` 共享原模型参数，绕开作者 forward 内部与 torch.func 不兼容的坐标 requires_grad_ 操作。测试核对实际原版 SIREN 的前向和参数梯度。此处只资格验证固定 omega SIREN；不是对所有神经表示变体的保证。

## 安装与两步真实正演检查

使用已有 PyTorch 环境，并安装仓库 requirements.txt 与 pytest；不要为此随意更换正在工作的 CUDA 环境。参考测试环境为 Python 3.12 / PyTorch 2.10 CPU。

在仓库根目录执行：

```bash
python -m pytest verification/natural_gradient -q
python natural_gradient_ifwi.py --config experiments/configs/natural_gradient/smoke_natural_gradient.yaml --device cpu --threads 1
python natural_gradient_ifwi.py --config experiments/configs/natural_gradient/smoke_adam.yaml --device cpu --threads 1
```

smoke 使用随代码附带的 9×11 合成模型、原版 FD、两步训练。其作用是检查真实计算路径、保存和恢复，不是科学精度或性能证据。两个配置除 optimizer.kind 外一致。

## Marmousi 实验

```bash
python natural_gradient_ifwi.py --config experiments/configs/natural_gradient/marmousi_natural_gradient.yaml --dry-run
python natural_gradient_ifwi.py --config experiments/configs/natural_gradient/marmousi_natural_gradient.yaml --device cuda --threads 1
python natural_gradient_ifwi.py --config experiments/configs/natural_gradient/marmousi_adam.yaml --device cuda --threads 1
```

配置继承仓库 stage1 的数据、单位、网格、SIREN 和 FD 条件，microbatch=1，400 步，每50步评估。需要本地已有的 `vel_marmousi_376x1151.csv`；数据解析由原加载器统一完成。配置中的 NG 阻尼0.1、步长0.01、CG上限5仅为待验证起点，**不是推荐最优超参**。先在独立开发设置上固定调参规则，禁止用最终测试真值挑模型。

高斯初速度由合成真值平滑而得，是已知的合成实验条件，不代表真实野外可获得的信息。与对照必须相同；需在论文明确说明此假设。先检查已有结果，避免重复跑已经完成的基线。

## 检查点与旧结果复用

- `--resume PATH/checkpoint.pt`：本方法同一配置、源码和数据续跑，允许设置不小于已完成更新数的总 epochs；等于已完成更新数时仅重新完成评估与导出。恢复参数、Adam（若适用）、RNG、历史与累计成本。线程数/硬件/PyTorch 等环境变化不保证逐位一致，必须额外核对 environment.json。
- `--initialize-from PATH/checkpoint.pt`：从本 runner 或原 `residual_ifwi_stage1_v1` 检查点开始新比较分支，保留模型和随机状态。切至 NG 不使用旧 Adam 状态，不能叫作 Adam 轨迹的无损续跑。Adam→Adam 保留状态；NG→Adam 无状态会拒绝。
- epochs 是包含起点在内的**累计更新数**。从50步开始再跑20步应设置70。
- 分支不会覆盖旧目录，先前成本不计入新分支账本，在 lineage.json 明示；resume 成本累计，并在 metrics.json 列本次新增量。
- 只加载可信的自己生成的 checkpoint；PyTorch pickle 可执行代码。强杀/断电只能恢复最后完整检查点，不能保证零工作丢失。

## 如何验收，而不是只看 loss

正常训练输出目录含 config/provenance/source_hashes/environment、status.json、work.jsonl、history.jsonl、evaluation.jsonl、checkpoint.pt、逐阶段检查点、metrics.json 及终点速度/波形/观测数组。状态初始化或恢复成功后的失败保存 interrupted.pt 与失败状态；恢复校验失败不产生可续跑检查点，不自动再次训练。

history 的波形损失与速度误差明确对应更新前模型；evaluation 对应更新后模型。终点用固定更新数选取，不按真值挑 best。metrics 同时保留生产 float32 波形 MSE 与已保存数组 FP64 重算值。

work.jsonl 对正演/反向分别记 attempted 与 completed 的炮数（不是严格的全算子 FLOP 等价）；包含观测生成、评估、失败尝试和续跑重新生成。CG 额外记网络乘积次数，inner_pde_calls=0；网络前向/JVP/VJP、I/O、初始化并非零成本。total_seconds 含初始化，training_and_evaluation_seconds 含阶段评估，另报峰值 CUDA allocated memory，不等于整卡占用。

公平比较要固定初值、数据、随机种子、采集/FD、评价频次和硬件，记录达到相同误差的总墙时及实际物理调用量。不能把更少外层步数直接当作加速，更不能称 DFM 优势。优先使用旧可比检查点，新的 GPU 验证需另外执行。

## 参考文献与贡献边界

- Nurbekyan, Lei, Yang (2023), *Efficient Natural Gradient Descent Methods for Large-Scale PDE-Based Optimization Problems*. https://doi.org/10.1137/22M1477805
- Müller & Zeinhofer (ICML 2023), *Achieving High Accuracy with PINNs via Energy Natural Gradient Descent*. https://proceedings.mlr.press/v202/muller23b.html
- Best McKay et al. (ICML 2025), *Near-optimal Sketchy Natural Gradients for Physics-Informed Neural Networks*. https://proceedings.mlr.press/v267/best-mckay25a.html

这里借鉴函数空间度量和矩阵无关自然梯度思想；没有使用后两篇的 energy metric 或随机截断伪逆。JᵀJ 不是含 PDE 灵敏度 A 的 JᵀAᵀAJ。创新价值需要后续实验，而不能由术语或公式本身推出。原 IFWI 来源继续沿用仓库 THIRD_PARTY_NOTICES.md。

仅完成评估/导出的恢复不会产生 history.jsonl 或 evaluation.jsonl，终点评估见 metrics.json，历史保留在 checkpoint.pt。
