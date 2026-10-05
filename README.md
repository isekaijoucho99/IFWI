# IFWI — Implicit Full Waveform Inversion

基于隐式神经表示（SIREN）与有限差分波动方程的二维地震全波形反演实验代码，包含 Marmousi、Overthrust、含噪观测、不确定性估计，以及传统 FWI 对照入口。

本仓库的基础版本来自本地 `IFWI_clean` 源码快照（打包日期：2026-09-29），原包的七个 Python 文件在打包时逐字节保留。本分支另增深层反演实验框架，见下文。基础版本不是 `IFWI_paper` 的论文参数版本，也不代表论文全部实验的精确复现。

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
├── experiments/                # 深层反演实验原型、配置、模块与对比工具
├── test_framework.py           # 新模块的模拟输入测试脚本
├── README_IMPROVEMENTS.md       # 新框架的早期说明（状态以本 README 为准）
├── requirements.txt
├── source_manifest.json        # 原作者模块和 CSV 的历史校验值
├── package_manifest.json       # 原始源码包清单与 SHA256，未覆盖本分支新增文件
└── THIRD_PARTY_NOTICES.md
```

原有入口的训练输出默认写入 `outputs/`；新框架的输出路径见下文，`outputs/` 与 `results/` 均已在 `.gitignore` 中排除。原始压缩包不含旧训练结果、检查点序列、日志、缓存、其他版本或绑定本机路径的 PowerShell 调度脚本。`data/` 中保留的是运行必需的基准模型；`weights/` 中保留约 200 KB 的预训练权重，并非历史反演检查点。

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

## 当前分支：深层反演实验原型

本分支 `feature/deep-layer-improvements-push` 在原有入口之外新增了独立的 `experiments/` 框架，目标是研究 Marmousi 深层、左下角和右下角区域的反演质量。原有 `main.py`、`fwi.py`、`ifwi_modules.py` 和 `rnn_fd.py` 保持不变。

**当前状态：实验原型，尚有训练阻碍和未接完功能。** 下文列出的是配置与代码现状；本次文档更新仅做静态核对，没有执行模块测试、完整训练或性能验证。新增说明中的提升百分比和结果示例不能视为已验证的实验结果。

### 五组实验与模块

- [baseline.yaml](experiments/configs/baseline.yaml)：随机初始化的普通 SIREN/IRN + MSE + Adam，作为新框架内部对照
- [depth_weighted_loss.yaml](experiments/configs/depth_weighted_loss.yaml)：配置深度加权、地质先验和梯度修改；深度加权的接线问题见下方限制
- [attention.yaml](experiments/configs/attention.yaml)：在隐式网络输出上增加依赖深度坐标 `z` 的可学习标量门控
- [adaptive_lr.yaml](experiments/configs/adaptive_lr.yaml)：按网络中 Linear 层的先后分组设置学习率，并使用 warmup / cosine 调度；该分组不是地下空间深度分区
- [combined_best.yaml](experiments/configs/combined_best.yaml)：组合上述策略的候选配置；文件名中的 `best` 不代表已通过实验选优

实现位于 [损失函数](experiments/improved_modules/losses.py)、[网络](experiments/improved_modules/networks.py)、[优化器](experiments/improved_modules/optimizers.py) 和 [深层评估](experiments/improved_modules/evaluate_deep.py)。另外提供 [单实验入口](experiments/run_experiment.py)、[结果对比](experiments/compare_results.py)、Linux/macOS 与 Windows 批量脚本，以及 [模块测试脚本](test_framework.py)。

补充说明见 [README_IMPROVEMENTS.md](README_IMPROVEMENTS.md) 和 [experiments/README.md](experiments/README.md)。这两份早期说明中的“误差降低 20–60%”“角落误差降低 50–70%”等属于预期，示例数值没有随该功能提交附上可复核的实验结果；使用前应以本节的实现状态与限制为准。

### 与原有入口的实验条件差异

- 新配置的网络为 `[2, 256, 256, 256, 256, 1]`，原 `main.py` 使用 `[2, 128, 128, 128, 128, 1]`；默认随机种子也从原入口的 `3` 改为 `42`
- 五组新配置均随机初始化，没有加载 `weights/ifwi_pretrain_marmousi.pth`；不要把新 `baseline` 当作原 `--experiment pretrain` 的等价对照
- Marmousi 读取与四倍下采样、15 m 网格、`dt=0.0019`、`nt=1000`、8 Hz 子波、15 m 震源深度及 30 m 接收深度沿用原设置；速度归一化仍为 `mean=3.0`、`std=1.0` km/s
- 当前五组配置均为无噪声、4000 次迭代；新数据入口读取 CSV，尚未接入原有 Overthrust、平滑预训练或 MC Dropout 不确定性实验流程

因此，新框架内的对照与原入口的历史结果应分别记录网络、种子、初始化及训练条件，再做比较。

### 依赖、入口与输出路径

先完成上方原环境安装，包括彼此兼容的 PyTorch / torchvision。`ifwi_modules.py` 仍会导入 torchvision，而 `experiments/requirements.txt` 没有列出它。新框架还需要 PyYAML 和 scikit-image；在仓库根目录安装额外依赖：

```bash
python -m pip install -r experiments/requirements.txt
```

以下是**修复下节训练阻碍后**的入口示例，均从仓库根目录执行。前两条命令会开始训练，本次未执行：

```bash
python experiments/run_experiment.py --config experiments/configs/baseline.yaml --output-dir results --device cuda:0
python experiments/run_experiment.py --config experiments/configs/attention.yaml --output-dir results --device cuda:0

# 完成实验并生成 metrics.json 后比较结果
python experiments/compare_results.py --results-dir results
```

另外三组实验替换为对应配置路径即可。Linux/macOS 批量入口为 `bash experiments/run_all_experiments.sh cuda:0`；Windows 的 cmd 入口为 `experiments\run_all_experiments.bat cuda:0`。这两个脚本设计为依次运行五组配置，但仍受训练代码问题限制。Windows 脚本另在 `FOR` 块内用 `%CONFIG%` 和 `%ERRORLEVEL%` 读取变化值，未启用延迟展开，需一并修正。修复训练阻碍后，可先使用上面的单实验命令；本次未执行批量脚本。

新入口的 `--output-dir` 相对当前工作目录解析；上例显式统一到仓库根目录的 `results/`。若先 `cd experiments`，应改用 `--output-dir ../results`，后续比较也使用 `--results-dir ../results`，避免单跑与批量结果落入不同目录。

训练与评估流程完成后，每组输出目录设计为 `results/<实验名>_<时间戳>/`，包含 `config.yaml`、checkpoint、`loss_history.csv`、`metrics.json`、`v_true.npy` 和 `v_pred.npy`。比较脚本设计为将汇总 CSV 与图写入 `results/comparison/`。默认深层区域为模型下半部，左右下角各取底部 25% 深度与相应侧 25% 宽度；评估包含相对误差、RMSE、SSIM、深度误差剖面及梯度保真度等。

### 新框架的已知限制

- **训练状态未初始化：** [新训练循环](experiments/run_experiment.py) 直接调用 `train_one_epoch`，没有执行父类 `IFWI2D.train` 中对 `self.params`、`self.clip` 的初始化，却在梯度修改/裁剪时使用它们；当前五组配置不能据此宣称已端到端跑通
- **深度加权尚未接通：** `DepthWeightedLoss.__call__` 当前返回普通 MSE，梯度 hook 部分仍为 `pass`。独立加权 helper 未接入训练；现有 runner 把空间深度权重应用到网络参数梯度，默认 `nz=94` 与首层 `[256, 2]` 参数梯度形状不匹配。即使补齐训练状态初始化，也仍需修正这一连接
- **策略名称不等于已证实的作用：** 自适应学习率按神经网络层分组；注意力门控依赖 `z`，但没有约束其随地层深度单调增大。它们对深层反演的实际收益需要独立实验验证
- **续训与绘图选项未接入：** 新入口声明了 `--resume`，但没有加载该参数对应的 checkpoint；配置中的 `evaluation.save_plots` 也没有被单实验入口使用。原 `main.py` 的续训与 `--plots` 说明仍只适用于原入口
- **最佳模型快照需修正：** 新训练循环直接保存 `state_dict()` 引用，未复制独立权重快照；后续训练可能改变内存中的 `best_model`，因此不能保证最终评估加载的就是最低记录 loss 对应模型
- **比较图的列表类型问题：** `depth_profile` 保存到 JSON 后会被读成列表；深度剖面绘图直接计算 `profile * 100`，会重复列表而非逐元素乘法，导致与深度坐标长度不匹配。应转回 NumPy 数组后再计算，修复前不能保证完整比较报告生成
- **测试与结果范围：** `test_framework.py` 使用模拟输入检查独立模块，未覆盖新的完整训练入口；本次未运行它，也未验证文档中的耗时、显存要求、精度目标或提升比例

下面保留原有 IFWI / FWI 入口的使用说明；其输出和续训机制与 `experiments/` 框架分别管理。

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

## 原有入口的已知限制

- 本版本 Marmousi 震源深度为 15 m、接收深度为 30 m，沿用原 Notebook；与论文描述及 `IFWI_paper` 版本有差异。
- `regularization` 保留原 auto-TV 逻辑问题，不能作为有效 TV 正则化对照。
- 原 IFWI 最佳模型选择比较更新前 loss、保存更新后权重；原 IFWI 波场非有限值处理也保留。传统 FWI 入口另有有限值检查和速度边界。
- 4000 次反演和 5000 次预训练是示例预算，不保证论文精度。GPU、依赖版本和随机性可能影响结果。

更多实现差异与历史验证记录见 [原始说明](docs/original_notes.md)。该文件保留原文，其历史“六个 Python 文件”和验证结论不作为本次包结构或本次验证结果；原始包的范围与验证见 [打包记录](docs/packaging.md)，本分支新增内容的状态见本 README。`package_manifest.json` 仍为原始包清单，未更新新增文件或本分支修改后的文件校验值。

## 来源与许可

原作者及参考文献信息见 [第三方来源说明](THIRD_PARTY_NOTICES.md)。现有目录没有可确认的上游 LICENSE，本包保留原文件作者标注，不擅自添加 MIT/Apache 等许可。

## 原始源码包的上传示例

解压后把 `IFWI_GitHub` 文件夹中的内容作为仓库根目录。也可进入该文件夹执行：

```bash
git init
git add .
git commit -m "Package IFWI experiment source"
git branch -M main
```

然后关联自己的远程仓库并推送。以上是原始本地源码包的历史上传说明；当前仓库已有 Git 历史，在本分支工作时无需重新初始化或重命名分支。
