# G²IFWI RED + P-IFWI-MS + LoRA：使用说明（实验分支）

**当前定位：原作者 U-Net 架构家族 + 短程 RED 去噪方向，与已有 P-IFWI-MS 物理反演结合。** 不是 G²IFWI 作者原始实验的严格逐位复现；没有原作者已训练的 `model_randy.pt`，也没有作者的 40,000 裁块训练集。

## 代码结构

- `author_unet.py`：提供 G²IFWI 作者同构的 `G2AuthorUNet`，支持原论文 128 通道、[1,2,4,8,16]、三个残差块、4 头以及激活检查点；保持原结构权重命名。
- `model.py`：`build_model` 支持旧轻量 U-Net 和 `g2_author`；Attention qkv/proj 的卷积 LoRA。
- `training.py`：训练 DDPM、验证、EMA、LoRA 适配；支持 `--batch-size 1 --accumulate-steps 16`、bf16、fp16 和激活检查点；预先进行显存保守估算。
- `guidance.py`：`mode=g2_red`。当前速度归一化后**直接**启动反向 DDPM 后验均值计算，不在推理时额外添加随机噪声；默认单步、`v - D(v)`，将参数梯度与波形梯度融合。原 `diffusion` score 模式仍在。
- `core_runner.py`：沿用 `rnn_fd.py` 和既有 P-IFWI-MS 3Hz → 5Hz → 全频带协议。先验只在所有炮批梯度累积后注入一次。

## 16GB 显存安全界限

作者原版 128 通道模型 **1,307,810,433 参数**。FP32 权重+梯度+2×Adam+EMA 约 24.4 GiB，未计激活，**不可能用普通 FP32 Adam 在 16GB 上从零训练**。对此启动前拒绝。推荐 32 通道、原作者五级 U-Net 结构（参数约 81.8M），这是缩窄宽度的版本，论文中不能标成“完全相同参数量”。LoRA 只能节省**已有基础模型后的适配训练**；首次基础预训练仍须进行。内存检查是保守拦截，**不能证明任何设置在 RTX 5060 Ti 上绝不 OOM**。

### 1. 准备独立地质训练/验证模型

```powershell
python -m experiments.diffusion_prior.training generate --out data/g2_source --train-models 512 --val-models 64 --shape 152 708 --profile mixed --grid-spacing-m 15 --seed 3
```

生成器是一个程序化近似，不能声称复现原作者的地质训练分布；测试 Marmousi 和 Overthrust 的真值不得进入训练与适配集合。

### 2. 先用 200 次更新测试 CUDA 的吞吐和显存

```powershell
python -m experiments.diffusion_prior.training train --architecture g2_author --channels 32 --patch-size 128 --grid-spacing-m 15 --mean-mps 4000 --scale-mps 3000 --train-dir data/g2_source/train --val-dir data/g2_source/val --out weights/g2_author_16g --steps 200 --batch-size 1 --accumulate-steps 16 --mixed-precision bf16 --lr 1e-6 --device cuda:0
```

同样参数恢复到 20,000 **总更新**，不覆盖已有结果：

```powershell
python -m experiments.diffusion_prior.training train --architecture g2_author --channels 32 --patch-size 128 --grid-spacing-m 15 --mean-mps 4000 --scale-mps 3000 --train-dir data/g2_source/train --val-dir data/g2_source/val --out weights/g2_author_16g --steps 20000 --batch-size 1 --accumulate-steps 16 --mixed-precision bf16 --lr 1e-6 --device cuda:0 --resume weights/g2_author_16g/last_training.pt
```

原论文 40,000×100/16 相当于约 250,000 次更新；20,000 步只是初期测试预算。检查 `history.json`、`summary.json`，在独立验证模型上比较去噪前后 RMSE。

### 3. LoRA 适配（在训练好基础权重以后）

```powershell
python -m experiments.diffusion_prior.training generate --out data/g2_target --train-models 128 --val-models 32 --shape 152 708 --profile folded --grid-spacing-m 15 --seed 33
python -m experiments.diffusion_prior.training adapt --base weights/g2_author_16g/best.pt --train-dir data/g2_target/train --val-dir data/g2_target/val --out weights/g2_lora --rank 4 --alpha 4 --steps 3000 --batch-size 1 --accumulate-steps 16 --mixed-precision bf16 --device cuda:0
```

Adapter 与基础权重 SHA256 绑定。对目标地质的有效性须通过独立验证，不能用被反演的真值裁块做 LoRA 训练。

### 4. P-IFWI-MS 受控对照

下列三份 JSON 都继承 `experiments/paper_supplement/specs/B_E2MS_smooth400_s3.json`：13 炮、smooth400 初始背景、残差 SIREN、3Hz/100 + 5Hz/100 + 全频带/800 updates，总共 1000 次。**只新增先验配置，不改波形正演。**

```powershell
python core_runner.py --spec experiments/configs/diffusion_prior/g2_pms/01_pms_baseline.json --out outputs/pms_no_prior --device cuda:0
python core_runner.py --spec experiments/configs/diffusion_prior/g2_pms/02_pms_g2red.json --out outputs/pms_red --device cuda:0
python core_runner.py --spec experiments/configs/diffusion_prior/g2_pms/03_pms_g2red_lora.json --out outputs/pms_red_lora --device cuda:0
```

后两组须先有真实基础先验/adapter 权重。RED 默认从第 101 次更新开始，每轮一次，单次去噪，整图到 128×128；归一化采用**固定训练范围**而不是作者反演代码的逐模型 min-max，以保持训练与推理一致（`red_normalization=per_model` 仅在匹配数据训练协议时使用）。`ratio=0.025` 是参数空间梯度比例起点，不等同于作者论文的固定 λ。Overthrust 20m 模型可结合已有 `physical_scale_bridge:true` 后单独构造对照 spec。

## 与原作者实现的边界

G²IFWI 作者部分 Overthrust 实现采用 `torch.mm(v.T, noise_residual).sum()`，其导数与论文写出的逐点 (v-D(v)) 并不严格等价。本实现**显式采用论文的 RED 去噪差方向**，并使用自己的 P-IFWI-MS 波动正演，不冒称原作者的代码逐行复现。

```powershell
python -m pytest -q verification/diffusion_prior
python -m compileall -q experiments/diffusion_prior core_runner.py
```

本分支不提供实际训练好的扩散权重；CPU 小网格数值与集成测试不能证明全尺寸 GPU 运行时间、峰值显存或地质反演精度。第一步只执行上述 200-step CUDA benchmark，记录实际峰值显存再增加预算。

## GPU microbatch tuning on RTX 5060 Ti (16 GiB)

A Windows Task Manager screenshot showing ~4.3/16 GiB **dedicated GPU memory**
indicates possible headroom but **does not provide peak allocated/reserved CUDA
memory or training throughput**. The Task Manager 3D graph is not a reliable
CUDA training-utilization metric. Do **not** enlarge the U-Net's channel count
in an existing run: a 32-channel checkpoint cannot be resumed as 64-channel
weights. Also do not start the profiler while another GPU training job is
running; terminate/finish the old job first so the GPU's free-memory baseline
is meaningful.

The author-style 32-channel model previously estimated six GiB of activation
memory **per micro-sample**, blocking microbatch 2 or 4 even when their real
memory use might fit. The static budget is now less pessimistic for base
training but is **not a GPU measurement**. The 128-channel, 1.308-billion-
parameter original still fails preflight on 16 GiB.

Use the following GPU probe to test real forward, backward, Adam, EMA,
validation and accumulation with isolated subprocesses. It starts at batch 1,
projects upper bounds before trying larger batches, and stops when the measured
peak reserve would consume more than 75% of the GPU's initially **free**
memory. This is a headroom policy, not an OOM guarantee. The probe consumes
compute but does not modify trained weights or run data-dependent FWI.

```powershell
python -m experiments.diffusion_prior.batch_probe `
  --model-channels 32 --patch-size 128 --effective-batch 16 `
  --candidates 1 2 4 8 --mixed-precision bf16 `
  --device cuda:0 --out outputs/g2_batch_probe_bf16.json
```

The resulting JSON contains `peak_reserved_bytes`, `peak_allocated_bytes`,
`samples_per_second` and the recommended `microbatch` / `accumulate_steps`.
Its throughput comparison uses synthetic input tensors rather than the full
`.npy` dataset loader; it is a **short compute benchmark**, so for publication
run a longer real-data interval and compare actual step times. A likely
first experiment is `--batch-size 2 --accumulate-steps 8`; advance to `4 / 4`
only if the probe confirms there
is sufficient headroom and better throughput. Do not change model channels,
normalization, patch size or learning-rate when comparing these batches.

An optional second probe checks whether removing activation recomputation is
faster with the remaining free VRAM:

```powershell
python -m experiments.diffusion_prior.batch_probe `
  --model-channels 32 --patch-size 128 --effective-batch 16 `
  --candidates 1 2 4 --mixed-precision bf16 `
  --no-activation-checkpointing `
  --device cuda:0 --out outputs/g2_batch_probe_no_ckpt.json
```

If this second probe has better throughput and a safe peak, include
`--no-activation-checkpointing` in the **new** training run. Always leave
approximately 3–4 GiB free for other Windows/CUDA allocations and unexpected
peaks. Keep a watch on VRAM and power using `nvidia-smi -l 1`.

**Checkpoint compatibility:** The existing trainer intentionally refuses an
exact `--resume` if the `batch_size`, `accumulate_steps`, or activation-checkpoint
setting changed. This prevents silent trajectory changes. Finish/continue the
original job with its original arguments, or start a **separate** training run
using the selected new settings. The new memory-probe module and adjustment to
static preflight do not alter the existing trainer's loss or optimizer steps.

**Limit:** A higher GPU-memory figure is not itself evidence of faster training.
In particular, the benchmark may recommend batch 2 instead of batch 4 if its
measured samples/second are higher. GPU-dependent recommendations cannot be
validated by the repository's CPU CI jobs.
