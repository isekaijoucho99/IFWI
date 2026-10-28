# 原网格扩散先验与 LoRA：使用说明

本实现是一个**默认关闭的实验插件**，保留 `rnn_fd.py`、速度参数化、数据读取、采集配置和主线 Adam 更新。通过 `core_runner.py` 的 `diffusion_prior` 顶层配置启用。它不是已训练好的地质先验，也未验证能提高 Marmousi/Overthrust 的精度或速度。

## 1. 实现范围

- 复用本仓库 `g2ifwi-repro@802150e` 的单通道噪声预测 U-Net 架构，不迁移其另一套正演求解器。
- 固定物理速度归一化 `x=(v-mean_mps)/scale_mps`；单位、网格间距、裁块尺寸、网络结构与 DDPM 调度写入 checkpoint。
- 原网格重叠裁块，末端补齐覆盖；只拼接/裁切，不把整个速度模型拉伸成正方形。小于裁块的网格明确报错，不隐式填充。
- 所有裁块使用同一张全网格噪声场；重叠区平均分数方向。U-Net 使用 `no_grad`，其参数冻结。
- 每次 IFWI 更新在所有炮批累积后最多注入一次先验，再执行一次原来的 `optimizer.step()`。不增加波动正演调用。
- 默认在第 51 次更新开始、每 5 次更新注入；噪声水平按 `eta=sqrt((1-abar)/abar)` 从 0.2 对数下降到 0.02。以上仅为起步配置。
- 默认在**参数空间**将先验梯度限制到物理梯度范数的 2.5%，记录夹角、范数、U-Net 调用数和耗时。也支持固定权重。
- 支持基础 DDPM 训练、独立验证集、参数 EMA、精确恢复 RNG 的训练续训，以及 attention `qkv/proj` 的 1×1 卷积 LoRA。LoRA 单独导出，绑定基础权重 SHA256。
- 提供无先验、仅插值、已训练基础先验、LoRA 先验、显式未训练先验五组配置。

当前版本只接 `core_runner.py` 的标准精细网格 Adam 路径；不支持同时启用 `speedup`/`accel` 或 Muon。普通 MS-IFWI 和 FR-IFWI 可以使用。旧 `g2ifwi/invert.py`、`residual_ifwi.py` 入口不自动启用新插件。

## 2. 环境与测试

沿用仓库依赖，额外测试依赖只有 pytest。新模块要求支持 `scaled_dot_product_attention` 和 `torch.load(weights_only=True)` 的 PyTorch；本地验证环境为 PyTorch 2.10.0 CPU。没有进行 CUDA 显存或速度测试。

```bash
python -m pytest -q verification/diffusion_prior
```

不修改原始核文件，`source_manifest.json` 检查仍保留。没有新增第三方模型下载、网络请求、遥测或自动 GPU 训练。

## 3. 准备先验训练数据

输入文件为多个独立的 `.npy`，每个文件一个 `[nz,nx]` 二维速度模型，单位 m/s。当前加载器不接收 RGB 图片或 `[N,1,H,W]` 打包数组。训练集和验证集必须先按完整模型分离，再裁块。加载时检查相同数值模型跨训练/验证集合的重叠。

```text
my_prior_data/
  train/model_00000.npy
  train/model_00001.npy
  val/model_00000.npy
```

外部数据的网格间距必须由使用者明确核对并通过参数声明；程序不能从数组猜出物理单位与尺度。自带生成器会在 `dataset.json` 写入间距，并在训练时核对。

先跑程序化数据的完整流程：

```bash
python -m experiments.diffusion_prior.training generate --out data/prior_source --train-models 512 --val-models 64 --profile mixed --grid-spacing-m 15 --seed 3
python -m experiments.diffusion_prior.training train --train-dir data/prior_source/train --val-dir data/prior_source/val --out weights/diffusion_base --steps 20000 --batch-size 8 --device cuda:0
```

**生成器只是独立层状/褶皱/断移模型的起步数据，不声称复现 G²IFWI 的训练分布或真实地质。** 这些训练步数、样本数和 batch size 没有做最优性或 16 GB 显存保证；先用较小 batch 进行烟雾测试。不要将测试 Marmousi/Overthrust 的真值或其裁块放入训练/适配集。

输出：`best.pt`（验证集选出的 EMA 权重）、`last_training.pt`（优化器和 RNG 等续训状态）、`training_config.json`、`history.json`、`summary.json`。日志同时报告固定噪声下去噪前后 RMSE。去噪误差改善不等于 FWI 改善。

续训使用总更新预算，不是追加步数；学习率和数据等身份信息必须一致：

```bash
python -m experiments.diffusion_prior.training train --train-dir data/prior_source/train --val-dir data/prior_source/val --out weights/diffusion_base --steps 30000 --batch-size 8 --device cuda:0 --resume weights/diffusion_base/last_training.pt
```

如果更改非默认参数，续训时也要显式提供原来的参数。`best.pt` 不能代替 `last_training.pt` 续训。

## 4. LoRA 适配

先验已有可用基础权重后，使用独立目标域模型适配。以下示例只演示域的设置，不构成 Marmousi 迁移成功的证据：

```bash
python -m experiments.diffusion_prior.training generate --out data/prior_target --train-models 128 --val-models 32 --profile folded --grid-spacing-m 15 --seed 33
python -m experiments.diffusion_prior.training adapt --base weights/diffusion_base/best.pt --train-dir data/prior_target/train --val-dir data/prior_target/val --out weights/diffusion_lora --rank 4 --alpha 4 --steps 3000 --batch-size 8 --device cuda:0
```

`best_adapter.pt` 只保存低秩增量，基础权重不被改写。推理必须同时指定基础 checkpoint 和 adapter；二者 SHA256 不匹配时拒绝加载。适配不能更改基础模型的归一化、网格间距或裁块尺寸。基础数据若与目标域几乎相同，LoRA 不一定有意义；建议另做同数据全参数微调基线，不能据低参数量直接宣称反演加速。

## 5. 启用 IFWI 对照实验

仓库提供 `experiments/configs/diffusion_prior/01_baseline.json` 到 `05_untrained.json`。共同设置为 Marmousi、13 炮、每炮微批累积、smooth400 背景、普通 MS-IFWI、固定 1000 次更新。没有真值停止条件。

```bash
python core_runner.py --spec experiments/configs/diffusion_prior/01_baseline.json --out outputs/diffusion_baseline --device cuda:0
python core_runner.py --spec experiments/configs/diffusion_prior/03_diffusion.json --out outputs/diffusion_base --device cuda:0
python core_runner.py --spec experiments/configs/diffusion_prior/04_lora.json --out outputs/diffusion_lora --device cuda:0
```

输出目录必须不存在，避免覆盖旧实验。先确认上述权重已经训练并通过独立验证。

其他控制实验：

```bash
python core_runner.py --spec experiments/configs/diffusion_prior/02_resize_only.json --out outputs/diffusion_resize --device cuda:0
python -m experiments.diffusion_prior.training init-control --out weights/diffusion_random.pt --seed 3
python core_runner.py --spec experiments/configs/diffusion_prior/05_untrained.json --out outputs/diffusion_random --device cuda:0
```

`init-control` 生成的权重标记为未训练，只有 `allow_untrained: true` 才能用于反演。不能把它当作可用地质先验。

生成自己的配置（不会执行反演）：

```bash
python -m experiments.diffusion_prior.examples --checkpoint weights/diffusion_base/best.pt --adapter weights/diffusion_lora/best_adapter.pt --untrained-checkpoint weights/diffusion_random.pt --out configs/my_prior --backbone ms --background smooth400 --updates 1000 --seed 3
```

`--backbone fr` 生成匹配的 FR-IFWI 对照；`--background gaussian` 保留原 sigma 设置；`--background linear_depth` 使用原深度线性背景。改变网格间距，例如 Overthrust 使用 20 m 时，必须使用相同尺度训练的先验，不能直接加载 15 m 先验冒充尺度一致。

## 6. 配置与数学约定

```json
"diffusion_prior": {
  "checkpoint": "weights/diffusion_base/best.pt",
  "adapter": null,
  "warmup": 50,
  "every": 5,
  "ratio": 0.025,
  "fixed_weight": null,
  "eta_start": 0.2,
  "eta_end": 0.02,
  "anneal_updates": 400,
  "patch_stride": null,
  "patch_batch_size": 4,
  "weighting": "kl",
  "seed": 3,
  "allow_untrained": false
}
```

默认 stride 为 patch_size/2。`patch_batch_size` 仅控制每次 U-Net 前向的裁块数，不改变覆盖定义。固定权重实验设置 `fixed_weight`，这时它覆盖 ratio 标定。

对归一化裁块 x，z=a*x+sigma*epsilon。`kl` 权重使用 q=(a/sigma)*(epsilon_pred-epsilon)，`epsilon` 权重只用噪声残差。裁块方向按覆盖次数平均，并除以全网格像素数与物理归一化尺度；再通过 `<v_theta, stopgrad(q_v)>` 回传到 IFWI 参数。

注意：覆盖平均相当于空间预条件，参数梯度范数标定又引入动态权重。这是**分数引导更新策略**，不声称每个步骤精确优化某一个固定 KL/RED 标量目标。代理点积数值不能用作“正则能量下降曲线”。小于 1 的参数梯度范数比例给出普通梯度方向的一阶物理下降界，不保证 Adam 实际更新单调。

## 7. 记录、续训与评估

`diffusion_prior.json` 保存 checkpoint/adapter 哈希和元数据。`history.jsonl` 每次更新的 `diffusion_prior` 字段记录是否启用、noise timestep、eta、覆盖、U-Net 调用次数、物理/先验参数梯度范数、夹角和 prior_seconds。`gradient_norm` 记录实际合成梯度。

`metrics.json` 增加 `diffusion_prior_seconds` 和 `diffusion_unet_calls`。计时包含先验，不能只比较 epoch 数或训练参数数量来证明加速。在线反演成本与离线预训练/适配成本要分别报告。

先验的随机噪声只由配置 seed 与一基更新编号决定，不消耗主线全局随机数；续训会重新核对基础/adapter 文件哈希及全部新增源代码。替换同名权重文件会拒绝续训。

```bash
python core_runner.py --spec experiments/configs/diffusion_prior/03_diffusion.json --resume outputs/diffusion_base/checkpoint.pt --out outputs/diffusion_base_resumed --device cuda:0
```

继续已达到预算的训练应使用原 runner 的 `--extend ... --epochs ...`，固定退火参数不会随新预算改变。正式比较保持初始化、炮点、频率递进、微批大小、停止协议和墙钟预算一致。只有 B（基础先验）优于 A（无先验），C（LoRA）又优于 B，才有证据支持保留对应组件。

## 8. 文件与参考

`model.py`：U-Net、DDPM、LoRA；`io.py`：版本与权重身份；`guidance.py`：原网格先验方向；`training.py`：训练、验证、EMA、数据生成；`examples.py`：受控实验配置。

算法参考而非代码复制：LoRA (Hu et al., arXiv:2106.09685)，DDPM (Ho et al., arXiv:2006.11239)，score distillation (Poole et al., arXiv:2209.14988)，以及 `lu-group/red-diffeq` 的停止梯度噪声残差和分块处理思路。不同时间权重与空间归一化的数值不可直接共用 lambda。
