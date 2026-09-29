# IFWI — Implicit Full Waveform Inversion

基于隐式神经表示（SIREN）与有限差分波动方程的二维地震全波形反演实验代码，包含 Marmousi、Overthrust、含噪观测、不确定性估计，以及传统 FWI 对照入口。

本仓库是本地 `IFWI_clean` 的源码快照（打包日期：2026-09-29），对应近期使用的实验版本。七个 Python 文件均逐字节保留，未因打包修改算法。它不是 `IFWI_paper` 的论文参数版本，也不代表论文全部实验的精确复现。

## 目录

```text
IFWI_GitHub/
├── main.py                     # IFWI 统一入口、续训、队列与加速验证
├── fwi.py                      # 独立传统 FWI 对照入口
├── pretrain_marmousi.py         # 平滑模型预训练
├── ifwi_modules.py             # 原作者隐式网络和训练模块
├── rnn_fd.py                    # 原作者有限差分正演
├── generator.py                # 子波与时间分段
├── plot_functions.py            # 绘图辅助
├── data/                       # 两份 Marmousi CSV、Overthrust NPZ
├── weights/                    # 小体积 Marmousi 预训练权重
├── docs/                       # 原始记录、环境、打包与验证说明
├── requirements.txt
├── source_manifest.json        # 原作者模块和 CSV 的历史校验值
├── package_manifest.json       # 本次完整文件清单与 SHA256
└── THIRD_PARTY_NOTICES.md
```

训练输出默认写入 `outputs/`，已在 `.gitignore` 中排除。压缩包不含旧训练结果、检查点序列、日志、缓存、其他版本或绑定本机路径的 PowerShell 调度脚本。`data/` 中保留的是运行必需的基准模型；`weights/` 中保留约 200 KB 的预训练权重，并非历史反演检查点。

## 环境安装

本机验证环境为 Python 3.10.18。PyTorch / torchvision 使用已有 CUDA 12.8 nightly 构建，具体版本见 [环境记录](docs/environment.md)。若已有可用环境，直接激活即可。

新环境示例（CPU 安装方式；完整训练建议使用 GPU）：

```bash
conda create -n ifwi python=3.10
conda activate ifwi
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

GPU 环境请安装与设备和驱动匹配、彼此兼容的 PyTorch / torchvision。`requirements.txt` 固定本机验证过的其他直接依赖；上面的通用 PyTorch 安装命令并非原 nightly 环境的精确复刻。本次没有验证新建环境。

在仓库根目录先检查入口：

```bash
python main.py --help
python fwi.py --help
python pretrain_marmousi.py --help
```

`main.py` 没有 `--check` 参数。不要把其他版本的命令与本版本混用。

## 运行 IFWI

下列命令会开始训练，按需单独运行。`--epochs` 表示最终总更新数。示例显式指定 `cuda:0`；无 GPU 可改为 `cpu`，完整规模运行会很慢。

```bash
# 从随包平滑模型权重初始化
python -u main.py --experiment pretrain --device cuda:0 --epochs 4000 --plots

# 随机初始化
python -u main.py --experiment random --device cuda:0 --epochs 4000 --plots

# 含噪观测：噪声标准差为观测标准差的 2 倍（也可设为 4）
python -u main.py --experiment noisy --noise 2 --device cuda:0 --epochs 4000 --plots

# Dropout 不确定性；显式采用 1000 次 MC 采样
python -u main.py --experiment uncertainty --dropout 0.2 --samples 1000 --device cuda:0 --epochs 4000 --plots

# Overthrust
python -u main.py --experiment overthrust --device cuda:0 --epochs 4000 --plots
```

默认不启用有效梯度裁剪或加速。需要时显式添加 `--clip-grad 0.25`；`--accelerate` 仅支持 random/pretrain 无噪声、无 Dropout 模式。不要把这些可选设置当成所有历史实验的统一配置。

重新生成预训练权重：

```bash
python -u pretrain_marmousi.py --device cuda:0 --epochs 5000 --output-dir outputs/pretrain_new
python -u main.py --experiment pretrain --device cuda:0 --epochs 4000 --pretrained outputs/pretrain_new/ifwi_pretrain_marmousi.pth
```

## 传统 FWI 对照

```bash
python -u fwi.py --experiment fwi_smooth --device cuda:0 --epochs 4000 --plots
python -u fwi.py --experiment fwi_random --device cuda:0 --epochs 4000 --plots
python -u fwi.py --experiment fwi_noisy --noise 2 --device cuda:0 --epochs 4000 --plots
```

## 输出与断点续训

每次运行生成独立目录，记录配置、状态、逐轮 loss、速度模型和 checkpoint；`--plots` 生成结果图。已有非空输出目录拒绝覆盖。

IFWI 续训示例，将示例路径换成实际保存的文件：

```bash
python -u main.py --experiment random --device cuda:0 --epochs 4000 --resume "outputs/random_TIMESTAMP/checkpoints/MarmousiI_random-checkpoint-1001.pth"
```

请保留 checkpoint 配套的 `config.json`，并使用与原运行一致的实验、噪声、后端和裁剪设置。续训另建输出目录；Dropout 续训不保证逐位一致。随包不含历史 checkpoint，因此不能直接恢复本机过去的训练。

## 已知限制

- 本版本 Marmousi 震源深度为 15 m、接收深度为 30 m，沿用原 Notebook；与论文描述及 `IFWI_paper` 版本有差异。
- `regularization` 保留原 auto-TV 逻辑问题，不能作为有效 TV 正则化对照。
- 原 IFWI 最佳模型选择比较更新前 loss、保存更新后权重；原 IFWI 波场非有限值处理也保留。传统 FWI 入口另有有限值检查和速度边界。
- 4000 次反演和 5000 次预训练是示例预算，不保证论文精度。GPU、依赖版本和随机性可能影响结果。

更多实现差异与历史验证记录见 [原始说明](docs/original_notes.md)。该文件保留原文，其历史“六个 Python 文件”和验证结论不作为本次包结构或本次验证结果；以本 README 和 [本次打包记录](docs/packaging.md) 为准。

## 来源与许可

原作者及参考文献信息见 [第三方来源说明](THIRD_PARTY_NOTICES.md)。现有目录没有可确认的上游 LICENSE，本包保留原文件作者标注，不擅自添加 MIT/Apache 等许可。

## 上传 GitHub

解压后把 `IFWI_GitHub` 文件夹中的内容作为仓库根目录。也可进入该文件夹执行：

```bash
git init
git add .
git commit -m "Package IFWI experiment source"
git branch -M main
```

然后关联自己的远程仓库并推送。本次只生成本地源码包，没有创建远程仓库或上传文件。
