# IFWI：单主入口版本

日常只使用 `main.py` 和 `pretrain_marmousi.py`。作者四个模块保持字节不变。
旧代码、notebook 与历史结果仍在原目录，此目录可以独立复制或打包。

## 文件

```text
main.py                   所有反演实验、队列、加速验证
pretrain_marmousi.py       平滑模型预训练
ifwi_modules.py           作者网络和原训练函数
rnn_fd.py                 作者正演模块
generator.py              作者子波/分段模块
plot_functions.py         作者绘图模块
data/                     两个原 CSV
weights/                  已完成的预训练权重
outputs/                  新实验输出（不会自动读取旧运行目录）
source_manifest.json      作者文件及数据 SHA256
```

## 环境与开始运行

在本目录打开 PowerShell，激活现有环境：

```powershell
conda activate cu128
python -u main.py --experiment pretrain
```

已附带 mean=3/std=1 的预训练权重，不必重复预训练。依赖为 PyTorch、torchvision、NumPy、pandas、SciPy、Matplotlib；不需要 Jupyter、W&B、py-spy 或 TensorBoard。
GPU 版 PyTorch 需与显卡兼容，现有 cu128 环境可直接使用。运行记录保存实际实验配置。

## 实验选择

```powershell
# 加载平滑模型预训练权重再反演：默认 1001 轮
python -u main.py --experiment pretrain

# 随机初始化：默认 4001 轮；默认不修复作者的裁剪行为
python -u main.py --experiment random

# 明确启用有效裁剪和已验证的加速，做 4000 轮
python -u main.py --experiment random --epochs 4000 --accelerate --clip-grad 0.25

# 只开启有效梯度裁剪
python -u main.py --experiment random --epochs 4000 --clip-grad 0.25

# 噪声标准差为观测数据标准差的 4 倍（也可传 2）
python -u main.py --experiment noisy --noise 4

# 保留原 notebook 的自动正则化实验，已知限制见下文
python -u main.py --experiment regularization

# Dropout 训练及 100 次预测的不确定性统计
python -u main.py --experiment uncertainty --dropout 0.2 --samples 100
```

当前 `--accelerate` 限 random/pretrain 两组无噪声、无 dropout 的实验。
每次独立设 seed=3；均值 3、标准差 1 km/s 按此前选定的论文设置。
所有模式共用同一训练与输出流程，没有独立入口文件，也不会自动开始下一个实验。

## 进度、保存和续训

每轮打印 `Completed/Loss/DataLoss/RegLoss`，同步写到 `progress.log`。
`--log-interval 100` 控制 checkpoint 保存与最佳模型选择间隔，而非逐轮文本间隔。
每次运行保存 `config.json`、`status.json`，训练结束保存 `loss.csv`、`best_velocity.npy`、`metrics.json`。
不确定性另存 `uncertainty.npz`；需要图才添加 `--plots`。
默认输出在 `outputs/实验名_时间`；也可通过 `--output-dir` 指定空目录，已有结果不会覆盖。

```powershell
python -u main.py --experiment random --epochs 4000 --accelerate --clip-grad 0.25 --resume "outputs/原运行目录/checkpoints/MarmousiI_random-checkpoint-1001.pth"
```

`--epochs` 是最终总更新数，不是新增轮数。续训要求 checkpoint 上两级目录有配套 config.json，数据、种子、后端与裁剪参数一致。
支持此前脚本默认未加速/未修复裁剪的配置（缺失的键按默认值处理），不会悄悄改变续训算法。
原作者 checkpoint 未保存 RNG，dropout 续训不能保证与不间断运行逐位相同。

## 顺序运行

```powershell
python -u main.py --sequence random noisy regularization uncertainty
```

每组一个独立进程，结束后再启动下一组，失败记录后继续；各组保存单独日志和目录，总状态在 queue_status.json。
`--epochs 10` 可将所有组限制为 10 轮做短测试。Windows 运行期间临时防止空闲休眠，不改电源方案；结束时解除。
Ctrl+C 会终止正在运行的子进程并记录中断状态。恢复时针对该组 checkpoint 单独续训，不会自动重跑已完成组。

## 仅验证加速

```powershell
python -u main.py --validate-speed
```

使用完整 94×288 网格、13 炮、1000 时间步；两个后端均启用 0.25 裁剪，以相同初始状态运行 20 轮。
需要额外中后期状态对照时，加 `--validation-checkpoints "中期.pth" "后期.pth"`；应使用同一架构、同一 mean/std 的 checkpoint。
固定 seed=3/cuda:0/clip=0.25。输出波场、梯度、短程速度差异及耗时，验证完成后**不会启动长训练**。
`--validation-steps 3` 只用于检查执行链路，正式速度/精度判断建议保留 20 或更多轮。

判据：波场相对 L2 <=1e-6、梯度 <=5e-5；短程速度差异 RMSE <=0.05 m/s、最大差 <=0.5 m/s；loss 相对差 <=1e-4。
速度中位数至少 1.15 倍且数值全部通过，才给出 prepared 建议。异常或不达标不建议启用加速。
短程测试不保证 4000 轮最终模型逐位相同，也不保证达到论文精度。

## 重新预训练

```powershell
python -u pretrain_marmousi.py --epochs 5000 --output-dir outputs/pretrain_new
python -u main.py --experiment pretrain --pretrained outputs/pretrain_new/ifwi_pretrain_marmousi.pth
```

预训练为补充实现，沿用作者平滑目标和网络，使用论文 mean=3/std=1、Adam 1e-4。
5000 轮是工程预算，不是论文指定。默认输出 weights，已有权重时拒绝覆盖；重新训练请指定新目录。

## 保留与明确改变的内容

- 作者四模块原样保留；loss、Adam、差分公式没有重写。
- 有效裁剪只由主入口的薄派生类启用：每步把优化器参数整理成可重复遍历的列表，使用指定阈值调用作者原步骤。未开启时保留原空迭代器警告。
- 加速只缓存接收器索引、每次正演填充一次速度；仍调用作者原单步差分函数，不改精度/炮数/时间步，不用混合精度。
- 反归一化固定 3/1，与旧主实验一致；原 notebook 则从真实模型计算均值标准差。
- 保留 notebook 的 CSV 默认表头处理、间隔 4 采样、15 m 网格、震源 15 m/接收器 30 m；后两者与论文描述不同，未在结构整理时修改。
- 保留原版 alpha='auto' 的逻辑问题：当前 loss 除以自身，正常情况下无法启用 TV。不要把 regularization 的结果当成有效 TV 对照。
- 保留原最佳权重选择逻辑：只在保存间隔比较，比较的是更新前 loss，保存的是更新后权重。
- 保留原 NaN/Inf 波场处理。默认运行不能据此保证数值稳定。
- 顺序运行、日志、加速验证全部合入 main.py；不再需要旧监视器/队列脚本/派生副本。

## 打包

直接打包本目录的六个 Python 文件、README、source_manifest.json、data 和 weights。
outputs 与 __pycache__ 不属于必需代码；测试保存在外部 tmp/clean_checks，不需要打包。

验证环境：Python 3.10 环境 cu128，torch 2.9.0.dev20250904+cu128、torchvision 0.24.0.dev20250905+cu128、NumPy 2.2.6、pandas 2.3.3、SciPy 1.15.3、Matplotlib 3.10.7。

本次整理验证：五类实验入口 GPU 短运行、顺序切换、有效裁剪、加速两轮与旧实现对照、断点续训到第三轮、预训练相对路径、dropout 均值/标准差、独立加速验证入口均通过。模型迁移差异不超过 0.00025 m/s；原模块和数据哈希未变。加速入口只做 3 轮流程测试，正式精度/速度结论仍应使用默认 20 轮。未启动长训练。


## Overthrust 泛化性实验（2026-09-22 新增）

使用 `python main.py --experiment overthrust --epochs 4000 --device cuda:0 --clip-grad 0.25 --plots`。
本次只添加入口，没有启动长训练或改动现有队列。4000 是运行预算，不是论文声明的收敛阈值；裁剪 0.25 是当前项目的可选设置，不是已确认的论文参数。

- 数据复制自 v1 的 overthrust2d.npz，vp 为 94×401、km/s，读取乘 1000 转 m/s；不使用 vi，不预训练。
- 论文 §3.4：20 m 网格、炮深 20 m、10 炮间距 800 m、8 Hz Ricker、dt=2 ms、nt=1500；归一化 mean=4.412、std=1.116 km/s，直接采用论文报告的井曲线统计值，不重新估算井位。
- 实现选择：炮点 x=400,1200,...,7600 m；接收器地表每 20 m 一个，共 401 个。炮点首位置和接收器间距尚未与原图精确核准。数据与论文切片是否逐点一致尚未确认。
- 原 SIREN、Loss、Adam、正演及边界实现保持不变；沿用原始网络随机初始化，不能据此声称复现 Figure 14a 的颗粒图。保存/最佳 checkpoint 逻辑也保持原样。
- 只修改 main.py：新增模式及专用参数、NPZ 读取和单位转换、参数化时间轴/观测几何/归一化/分段长度、checkpoint 名称，以及结果图的物理尺寸和速度色标。其余 Marmousi 模式使用原参数。
- 新增 data/overthrust2d.npz 和此说明；逐行代码差异见 overthrust_changes.diff。未修改作者四个核心模块。
- 验证使用 CPU 缩短时间窗（32 点）的一次训练更新，检查真实正演、反传、checkpoint、结果图与指标保存；不代表完整 1500 点长训练已验证。


## 2026-09-22 不确定性实验参数修改

按用户要求，main.py 的 --dropout 默认值由0.4改为论文的0.2，运行示例同步更新。新实验使用新输出目录、从头初始化，不续训原3201步的0.4模型。其他设置不变：仍是Notebook观测几何，MC默认100次，并不代表整个实验已与论文全部一致。未修改作者核心模块、历史配置或checkpoint，未启动新训练。
