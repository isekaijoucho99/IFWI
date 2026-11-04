# 使用入口：五个独立 IFWI 研究原型

本地完整 checkout 的安装与回归核验见 [`LOCAL_VALIDATION.md`](LOCAL_VALIDATION.md)：已修复指标测试的 R² 契约断言、恢复 GPU 批量求解，并于 2026-11-04 将 DFM 核心适配为固定版本的官方实现。最新完整回归 420 项通过、2 项因 Windows 符号链接权限跳过；真实 RNN-FD 桥接与 CUDA 一致性检查通过。DFM 的修复证据与研究边界见 [`DFM_UPSTREAM_AUDIT.md`](DFM_UPSTREAM_AUDIT.md)。`VALIDATION.md` 保留原始交付状态。

## 先看哪些文件

- `design_and_experiments_zh.md`：五个方案的数学模型、训练过程、消融、失败判据、统一实验协议。
- `VALIDATION.md`：本次实际执行的验证与未验证边界。
- `ifwi_research/`：核心模块、训练器、独立测试求解器和原仓库适配器。
- `verification/five_ideas/`：数值、恢复、CLI、数据划分和真实 FD 桥接测试。
- `experiments/configs/five_ideas/`：11 个单因素研究配置。
- `tools/publish_five_ideas.py`：从交付 ZIP 在本地已认证仓库中建立独立分支，测试后可推送。

**原始 ZIP 交付时没有远程提交。** 此包是添加式源码，不含原仓库代码、数据或有效的地质 DFM 权重。不要把独立小网格测试结果当成 Marmousi/Overthrust 实验结果。

## 1. 独立环境检查

Python 3.10+，PyTorch 2.2+，NumPy 1.24+，pytest 7+。在 CUDA 机器使用兼容目标 GPU 的 PyTorch 构建，不要为了这个包覆盖已有的可用 CUDA 环境。原仓库的其他依赖仍需保留。

在解压目录执行：

```bash
python -m pytest verification/five_ideas -q
python -m ifwi_research.demo --method routing --steps 10 --out outputs/demo_routing
python -m ifwi_research.demo --method wavelet --steps 10 --out outputs/demo_wavelet
python -m ifwi_research.demo --method interface --steps 10 --out outputs/demo_interface
python -m ifwi_research.demo --method gradient_attention --steps 10 --out outputs/demo_gradient
```

每次输出目录必须新建。`demo` 是独立的极小二阶声学差分，不是原仓库正演，也不是地质性能基准。单独交付目录缺少原文件时，真实 RNN-FD 桥接测试会显示 `SKIPPED`。

## 2. 安装并发布到已有 GitHub 仓库

脚本从**解压目录**运行，需要本地 `git` 已能正常访问 `isekaijoucho99/IFWI`。无需将 token 发给任何人；脚本不会读取、保存或修改凭据。

Windows 示例，路径替换为实际路径：

```powershell
python tools/publish_five_ideas.py --repo "C:\Projects\IFWI" --dry-run
python tools/publish_five_ideas.py --repo "C:\Projects\IFWI" --push
```

默认新分支：`feature/ifwi-five-ideas-core-20261030`。
默认新工作树：原仓库同级的 `IFWI-five-ideas`。可使用 `--worktree`、`--branch` 指定其他**未存在**的名称。不使用 `--push` 时只创建本地提交。

执行顺序：验证 origin→校验文件 SHA256 与路径→获取基分支→检查远程同名分支不存在→从固定提交 `226be2262377a0bbde4f813205ceb7cbcd69f335` 建立工作树→复制新增文件→运行包括真实 RNN-FD 的新增测试→编译检查→只提交清单中的新增文件→可选推送。

任何一步失败都会停止；保留工作树供排查，不回滚/删除用户当前工作区，不覆盖已有文件，不 force push。需要已有 Git 作者配置。它不会运行正式 GPU 训练。测试包含原仓库适配器验证，因此安装后可能暴露本次无法在独立环境检验的兼容问题；出现失败时应修复，而不是跳过后发布。

另附添加式补丁可手动 `git apply --check`、`git apply`。不要把不含原项目历史的交付目录本地 Git 提交直接推到原仓库主线。

## 3. 原仓库 P-MS 对照与正式路径

进入安装后的工作树，先运行：

```bash
python -m pytest verification/five_ideas/test_repository_bridge.py -v
```

随后用同一既有 baseline spec：

```bash
python -m ifwi_research.run --baseline experiments/configs/diffusion_prior/g2_pms/01_pms_baseline.json --research experiments/configs/five_ideas/00_baseline.json --epochs 10 --shot-batch-size 13 --device cuda:0 --out outputs/five_ideas/pms_smoke
python -m ifwi_research.run --baseline experiments/configs/diffusion_prior/g2_pms/01_pms_baseline.json --research experiments/configs/five_ideas/01_routing.json --epochs 10 --shot-batch-size 13 --device cuda:0 --out outputs/five_ideas/routing_smoke
```

10 步仅用于接口与显存检查。正式运行去掉 `--epochs 10`，沿用 baseline 的固定预算与频率日程。其余方向只替换 `--research` 和新输出目录。

2026-11-01 起，运行器继承 baseline 的炮批大小，`null` 表示全部炮；也可用 `--shot-batch-size` 显式覆盖。该大小同时控制物理炮与选中编码源的批量求解。Marmousi 的 13 炮可以在本机 16 GB GPU 上批量执行，不再强制单炮串行；需要低显存路径时显式指定 `--shot-batch-size 1`。

路由方案仍只求解原采样规则选中的编码源，逐编码重要性权重、能量分数和真实源求解计数均保留。批量仅改变执行分组，不改变网格、物理算子、随机种子、优化器或频率日程。有效批大小写入 `identity.json`，更改批大小不能冒充同一精确续训协议。实际 CUDA 一致性与耗时记录见 `LOCAL_VALIDATION.md`。

| 配置 | 作用 |
|---|---|
| `00_baseline.json` | 新 runner 内匹配的 P-MS 对照 |
| `01_routing.json` | 共享源 + 动态对比源路由 |
| `02_wavelet.json` | 单级 Haar 跨带门控 |
| `03_interface.json` | 锚定单界面修正 |
| `04_gradient_attention.json` | 物理梯度上下文 + 块内增量门控 |
| `05_dfm.json` | 训练 DFM + 受保护先验梯度 + 实际候选审查 |
| `06_guarded_baseline.json` | 匹配 DFM 的审查频率，无先验 |
| `07_uniform_contrast.json` | 共享源 + 均匀对比源采样 |
| `08_wavelet_identity.json` | 关闭小波门控的精确恒等路径 |
| `09_interface_regularized.json` | 显式开启界面正则，权重需在开发集检验 |
| `10_gradient_attention_identity.json` | 梯度门控强度为零，精确返回 base |

随机种子与学习率默认继承 baseline，可在研究配置显式指定；正式实验使用开发种子 3 之外的新种子 10–14。独立研究配置不自动更换数据集；Overthrust 应使用对应原仓库 baseline spec 和物理采样间距，不把 Marmousi 的数据路径替换后就默认其他参数全适用。

`--snr-db 20` 或 `--snr-db 10` 注入固定种子 AWGN，噪声按全部干净记录样本功率定义；不会重用含糊的旧 `noise_sigma` 语义。

## 4. 先训练 DFM，再进行先验反演

准备相互独立的完整二维地质速度模型 `.npy`，单位 m/s。先分模型，再裁块。禁止包含测试真值或同一父模型的其他裁块。

```bash
python -m ifwi_research.train_dfm --train-dir DATA/train_models --val-dir DATA/val_models --grid-spacing-m 15 --patch-size 64 --width 32 --group-size 64 --groups-per-step 4 --sinkhorn-iters 20 --steps 20000 --validate-every 100 --device cuda:0 --out weights/dfm_official_15m_v2
```

20,000 步是可调的研究预算，不是已验证的收敛步数。当前版本直接使用固定版本的官方分组 Sinkhorn 与匹配损失，采用同一噪声端点、官方 logit-normal 时间采样和 EMA 推理权重。来源、旧实现差异与适配边界见 [DFM_UPSTREAM_AUDIT.md](DFM_UPSTREAM_AUDIT.md)。

独立验证从 `t=0` 高斯噪声生成，报告 1/2/4 NFE 的条件生成 RMSE，`best.pt` 按固定 4 NFE 选择。验证条件来自独立模型的平滑场，仍不代表反演估计条件下有效；同时报告纯条件场的 RMSE 作为参考。训练输出包括 `dataset_manifest.json`、`history.jsonl`、`last.pt`、`best.pt`、`summary.json`。旧 `ifwi_small_dfm_v1` 权重会被明确拒绝，需重新训练，不覆盖原文件。

`best.pt` 只按独立验证模型选取；测试真值不参与。调试训练可用很少步数，但这种权重不能据此被当成有效地质先验。当前训练器会保存优化器与 RNG，**尚未提供训练 CLI 的续训参数**；这与反演 runner 的精确续训是不同能力。

确认研究配置 `05_dfm.json` 的 checkpoint 路径后：

```bash
python -m ifwi_research.run --baseline experiments/configs/diffusion_prior/g2_pms/01_pms_baseline.json --research experiments/configs/five_ideas/05_dfm.json --device cuda:0 --shot-batch-size 13 --out outputs/five_ideas/dfm_run
```

15 m 与 20 m 先验不能直接混用。本版本严格检查物理采样间距。默认从独立噪声开始，使用当前速度平滑场作为固定条件，以 4 NFE 生成候选。DFM 只在第 201 步开始注入，避免低波数背景未建立时强加结构。物理梯度保护和候选审查是 IFWI 适配，不属于原论文的生成算法。

## 5. 反演续训和结果文件

只允许同一预算、同一源代码、同一配置/数据/先验权重/设备的精确续训到新目录：

```bash
python -m ifwi_research.run --baseline experiments/configs/diffusion_prior/g2_pms/01_pms_baseline.json --research experiments/configs/five_ideas/04_gradient_attention.json --resume outputs/five_ideas/gradient_run/checkpoint.pt --device cuda:0 --out outputs/five_ideas/gradient_resumed
```

若原运行用了 `--epochs` 或 `--snr-db`，恢复时必须相同。不支持把已跑完预算的 checkpoint 静默扩展成另一协议。

输出：
- `identity.json`：源文件、观测、初值与配置身份；
- `history.jsonl`：预更新物理目标、频带、路由选择、候选审查、累计 PDE 成本与时间；
- `checkpoint.pt`：网络、Adam、路由 RNG、注意力上下文/锚点与运行状态；
- `velocity_XXXXX.npy`：固定间隔速度；
- `summary.json`：固定终点的 RMSE/MAE、全炮全频目标、成本、显存；
- `FAILED.txt`：异常说明，存在异常时不要把目录计为成功运行。

SSIM、空间波数与界面指标等属于设计中的正式评价，需要对保存快照使用统一的离线分析器；当前 runner 未把所有论文指标都内置。不得以最低真值 RMSE 快照替换固定终点而不说明。

## 6. 文件职责

| 文件 | 主要接口 |
|---|---|
| `dfm.py` | `TwoTimeTransport`、`conditional_group_loss`、`FrozenDFMPrior` |
| `train_dfm.py` | 模型级数据划分检查、分组训练、固定验证 |
| `routing.py` | `common_contrast_basis`、`ContrastRouter`、`CodePlan` |
| `wavelets.py` | `haar_forward`、`haar_inverse`、`WaveletVelocity` |
| `interfaces.py` | `InterfaceVelocity` |
| `gradient_attention.py` | `GradientAttentionVelocity.refresh` |
| `safety.py` | 参数空间梯度保护、完整 Adam/RNG 回退 |
| `engine.py` | 一次只选择一种思路、物理梯度到网络 VJP、检查点 |
| `repository.py` | 原 IFWI 的延迟导入、真实源编码、原数据目标 |
| `run.py` | 固定预算正式实验入口 |
| `toy_wave.py` / `demo.py` | 独立数值验证，不是正式求解器替代 |

## 7. 必须保留的限制

没有本次会话的远程推送、CUDA 性能结果或全尺寸反演结果。当前只写出有测试的核心原型，正式实验表中的若干强对照仍待实施。真实 FD 测试未通过之前，不应开始长期 GPU 训练。原仓库完整旧测试套件未在本次独立目录执行。
