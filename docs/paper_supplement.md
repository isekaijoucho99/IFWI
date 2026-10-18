# 论文补充实验（审稿后 A–K）

对应论文工程 `docs/补充实验设计_20261017.md`。所有运行都用 `core_runner.py`，它就是论文 E2/E3/E4、Overthrust 与噪声实验的运行器：上一提交原样加入，本提交只做扩展。未用到新功能的 spec 与原批次算术相同。基础配置（E2、E2-MS、E3、E4、E4 5 m/s、直接网络预训练、Overthrust 与 σ=2/4 的 E2/E3/E4）已与结果包中的冻结配置逐项核对一致。

## 准备（GPU 机器，一次）

```bash
git pull
python paper_supplement.py prepare-data
```

`prepare-data` 从 `data/overthrust2d.npz` 生成 `data/overthrust_94x401_mps.csv`，并核对 sha256 与稳健性批次一致（244891a5…）。

`--shared-dir` 指向稳健性批次的 `shared` 目录（例如 `F:\project\IFWI\outputs\robustness_overthrust_noise_20261016\shared`）。这样实验 B、C、J 直接复用原有的带噪观测和 Overthrust 观测（按哈希核对）；干净 Marmousi（`marmousi_clean`）和线性背景（K）会在第一次运行时生成到同一目录。

## 队列与顺序

spec 已冻结在 `experiments/paper_supplement/specs/`，队列在 `queues/`。每个队列在同一会话内串行运行；`--wait-idle` 会在每次运行前等待 GPU 空闲，保证计时条件一致：

```bash
python paper_queue.py experiments/paper_supplement/queues/q1_quick_F.json --out-root outputs/paper_supplement --shared-dir SHARED --wait-idle
```

| 顺序 | 队列 | 内容 | 运行数 |
|---|---|---|---|
| 1 | `q1_quick_F` | F：零输出 E2，seed 3，前 5 步（逐层 ΔW、梯度范数、δv_k） | 1 × 5 步 |
| 1 | 实验 I（无训练） | 见下文命令 | — |
| 2 | `q2_screen_CDH` | C：网格 FWI，lr 3/10/30 m/s；D：E2、E3 各取 η 3e-5/1e-4/3e-4，另加 E2 s_v=300；H：输入 Fourier 特征，s=1/3/10 c/km；均跑 200 步 | 13 × 200 步 |
| 3 | `q3_B_E` | B：带噪 E2-MS（σ=2、4）；E：直接表示 + Fourier（固定 ω，预训练 5000 步后重置 Adam） | 3 |
| 4 | `q4_A_minimal` | A 最小版：seed 0/1/3 的 E2-MS、E3（零输出）、E4（5 m/s 输出），加上 E 的 seed 0/1 | 11 |
| 4 | `q4_A_full_extra` | A 完整版另加：E4（零输出）、E3（5 m/s 输出）、E2（全频带） | 9 |
| 5 | `q5_C_formal`、`q5_DH_formal` | 筛选后生成：C 正式（4 个条件 × 全频带/多尺度），D、H 正式 | 8 + ≤ 3 |
| 6 | `q6_G` | G1 warmup 20 步，G2 沿用预训练的 Adam 矩，G3 η=1e-5，G1F（直接 + Fourier + warmup） | 4 |
| 6 | `q6_J` | σ=4 的 E4 + 学习率安排（warmup 20，第 201–1000 步余弦衰减到 0.1），以及 β₂=0.99 | 2 |
| 7 | `q7_K`（可选） | 线性深度背景 v0 = 1500 + 2.0·z（m/s，z 以 m 计，不用真值）上的 E2-MS、E3、E4 | 3 |

**筛选后生成正式 spec**：只用全炮全频波形 MSE，不用真值。

```bash
python paper_supplement.py select --name C_grid_lr --runs outputs/paper_supplement/C_screen_grid_lr3/attempt_01 outputs/paper_supplement/C_screen_grid_lr10/attempt_01 outputs/paper_supplement/C_screen_grid_lr30/attempt_01
```

```bash
python paper_supplement.py specs --selection experiments/paper_supplement/selection.json
```

D 用 `--name D_E2_lr`、`--name D_E3_lr`，H 用 `--name H_scale`，传入对应的三个筛选目录。若 D 选中 1e-4，E3 的正式运行与 A 的 `A_E3_s3` 相同，可不重复跑。

## 实验 I（v0 处梯度方向，无训练）

```bash
python -m experiments.gradient_report --states initial --config experiments/paper_supplement/configs/I_overthrust.json --max-hz full 3 5 --device cuda:0 --output-dir outputs/paper_supplement/I_overthrust
```

```bash
python -m experiments.gradient_report --states initial --config experiments/paper_supplement/configs/I_marmousi.json --max-hz full 3 5 --device cuda:0 --output-dir outputs/paper_supplement/I_marmousi
```

## core_runner 新增功能（spec 不用时不起作用）

| spec 字段 | 用途 | 实验 |
|---|---|---|
| `experiment.training.frequency_schedule` | 多尺度阶段（与 E2-MS 相同的滤波和能量归一，按训练观测计算，带噪条件即带噪观测） | A、B、C、K |
| `experiment.training.lr_schedule`、`optimizer.betas` | warmup、余弦衰减、Adam β | G1、J |
| `model: {kind: grid}` | v = v0 + δv，Adam 直接更新 δv，学习率单位 m/s | C |
| `model: {kind: fourier_features, scale_cpkm}` | γ(q) = [sin 2πGq, cos 2πGq]，G ∈ R^{128×2} ~ N(0, s²)（c/km，坐标以 km 计）；ReLU MLP 4×128；输出层零初始化；残差尺度 1000 m/s | H |
| `background: {kind: linear_depth, ...}` | 用只随深度变化的线性背景替换高斯背景 | K |
| `control.keep_optimizer_state` | 预训练后不重置 Adam，用最后一步的矩继续反演（模型取最优状态） | G2 |

每次运行的输出与原批次相同：逐步 `history.json`（data_mse、gradient_norm、velocity_delta_rms_mps、omega、cumulative_seconds、各阶段的 max_hz/objective、lr_factor）、`mechanisms.jsonl`（前 5 步与每 50 步的逐层 ΔW 和梯度范数）、每 50 步的速度场与全炮全频波形 MSE（`evaluation_history.json`）、计时分项、环境信息与提交号。`core_runner.py --epochs 5` 可以对任何 spec 只跑前 5 步做快速检查。

**未在本机验证**：新旧运行器对同一 E2/E3/E4 spec 是否逐位一致。建议在 GPU 上先跑 `core_runner.py --spec experiments/paper_supplement/specs/A_E3_s3.json --epochs 5 ...`，并与论文 E3（core_missing3 批次）`history.json` 的前 5 行对照 data_mse 和 RMSE。

## 加速实验（`q8_speed`）

正演和反传占每次更新 99% 以上的时间，网络本身的计算可以忽略，所以这三组实验都在减少每次更新的模拟量（`core_speed.py`，原正演文件不改）。

队列按顺序运行，基线与加速运行在同一会话内计时：

| 运行 | 内容 | 与谁只差一项 |
|---|---|---|
| `A_E3_s3` | E3 基线（与 q4 同名，已跑完会被跳过） | — |
| `A_E2MS_s3` | E2-MS 基线（同上） | — |
| `S1_E3_encoded_s3` | E3 + 震源编码，3000 次 | E3：编码 |
| `S2_E2MS_coarse_s3` | E2-MS，3/5 Hz 阶段在 30 m 网格上模拟，100/100/800 | E2-MS：粗网格 |
| `S4_E2MS_encoded_s3` | E2-MS + 编码，100/100，3000 次 | E2-MS：编码 |
| `S3_E2MS_coarse_encoded_s3` | E2-MS + 粗网格 + 编码，100/100，3000 次 | S4：粗网格；S2：编码 |

E2-MS、S2、S4、S3 构成"粗网格 × 编码"的 2×2 设计。无学习率衰减时预算只决定停在哪一步，3000 次与 1000 次运行前 1000 步的设置相同。比较统一用达到 ε=396.95 m/s 的实测时间 t* 和 RMSE–时间曲线，不比更新次数。若 S2 变差，可再加一组"细网格、震源 30 m"的 E2-MS，检验震源深度近似本身的影响。

- 编码前提是所有炮共用同一排检波器（本例满足）。编码预测与逐炮预测加权求和的相对误差为 1e-6（线性性检查）；炮间串扰相当于梯度噪声，所以给了更多更新次数，比较时以达标时间 t* 为准。
- 粗网格阶段的震源只能放在 30 m 深（原为 15 m），自由表面虚反射使低频振幅约差 2 倍；在背景模型上用最小二乘估计一个固定的子波振幅因子校正（记录在 `coarse_lowpass.json`）。
- 日志中：编码时 `data_mse` 是编码超级炮的失配；粗网格阶段 `data_mse` 记的是该阶段目标函数。可比的全炮全频波形 MSE 和 RMSE 看每 50 次的 `evaluation_history.json`；其评价时间计入累计时间。

```bash
python paper_queue.py experiments/paper_supplement/queues/q8_speed.json --out-root outputs/paper_supplement --shared-dir SHARED --wait-idle
```
