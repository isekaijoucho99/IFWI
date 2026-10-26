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

## 走时坐标变换（`q9_warp`，方向 1）

SIREN 的输入从 (x, z) 改成穿过背景模型 v0 的走时坐标：z' = c_z ∫₀^z dz/v0，`axes: "xz"` 时再加 x' = c_x ∫₀^x dx/v0（逐行）。坐标网络在输入空间里各处的频谱偏置相同，变换后这种偏置在“v0 的局部波长”意义下均匀：v0 慢的浅部更细，快的深部更平滑，与数据分辨率 k_res ≈ 2f/v0 一致。c_z、c_x 保持各轴的平均坐标范围不变，因此全局带宽（ω₀ × 坐标范围）与原网络相同，只改变空间分布。只用 v0，不用真值；网络、种子对应的初始化和零输出层都不变，第 0 步仍精确等于 v0。Marmousi 高斯背景下深度方向的局部拉伸为浅部约 1.45、深部约 0.69（`coordinate_warp.json` 记录）。

spec 写法：`"model": {"kind": "siren", "coordinate_warp": {"kind": "traveltime", "axes": "z"}}`，可与 E2/E3/E4、多尺度、编码等组合（低秩模型除外）。

| 运行 | 内容 | 与谁只差一项 |
|---|---|---|
| `A_E3_s3` | E3 基线（与 q4/q8 同名同配置，已跑完会被跳过） | — |
| `W_E3_warpz_s3` | E3 + 深度走时坐标 | E3：坐标 |
| `W_E3_warpxz_s3` | E3 + 双向走时坐标 | E3：坐标 |
| `A_E2_s3` | E2 基线（与 q4_A_full_extra 同名同配置） | — |
| `W_E2_warpz_s3` | E2 + 深度走时坐标：单靠坐标能否消除 E2 的竖条纹 | E2：坐标 |
| `W_E3_overthrust_s3` | Overthrust E3 基线（同一运行器、同一队列） | — |
| `W_E3_warpz_overthrust_s3` | Overthrust E3 + 深度走时坐标：逆冲岩片内的纹理 | Overthrust E3：坐标 |

```bash
python paper_queue.py experiments/paper_supplement/queues/q9_warp.json --out-root outputs/paper_supplement --shared-dir SHARED --wait-idle
```

看点：除 RMSE、SSIM 外，主要看深部（下 1/3）RMSE 和超出 k_res 的扰动能量比例。预期 SSIM 和纹理改善的可能性较大；Marmousi 底部高速薄层是 RMSE 的主要来源，深部被压缩后 RMSE 可能持平甚至略升。单种子差别小于 10 m/s 不作结论，有希望的组合再补 seed 0/1。

**先做的无正演检查**（CPU 每组约 2–3 分钟）：`experiments/warp_fit.py` 用论文的 Adam 设置把残差网络直接拟合到目标速度，比较 (x, z) 与走时坐标下各深度段的拟合速度和超出 k_res 的能量。目标 `resolved` 是按 k_res(q) 做空间变化平滑后的真值扰动，相当于带限数据能约束的部分；拟合结果在 k_res 以上的能量就是没有数据要求的纹理。

```bash
python -m experiments.warp_fit --case marmousi --method E3 --axes none z xz --target full resolved --steps 2000
```

Overthrust 需要先 `prepare-data` 生成 CSV，再用 `--case overthrust`。结果在 `outputs/warp_fit/<case>_<method>/summary.json`。

## 完整方法 Ours 的补充条件（`q10_ours`）

论文中的 Ours = E4（可训练 ω）+ 5 m/s RMS 输出初始化（`A_E4small5_*` 的同一配置），此前只在干净 Marmousi 上跑过。本队列在 Overthrust、σ=2（−6 dB）、σ=4（−12 dB）各跑一次 seed 3，观测与稳健性批次共用（按哈希核对）。表 III 中原有的 Ours-ZF/Ours-Z 来自另一会话，比较时间时注明。

```bash
python paper_queue.py experiments/paper_supplement/queues/q10_ours.json --out-root outputs/paper_supplement --shared-dir SHARED --wait-idle
```

## 续跑未收敛的基线到 4000 步（IFWI、FR-IFWI、Res-IFWI）

直接从原运行的最终 checkpoint 接着跑，前 1000 步不重算；新结果写到新目录，原目录不动。无学习率衰减，所以续跑与一次跑满 4000 步算术相同（`test_extend_continues_a_finished_run_exactly` 在 CPU 上逐位核对 3+2 步 = 5 步）。

IFWI（论文 Direct_prefit_v0，core_missing3 批次）与 FR-IFWI（E_directFR_prefit_s3）用 `core_runner.py --extend`：spec 从 checkpoint 读出，只允许改更新预算；此后修改过的源文件列在新目录的 `extend.json`。

```bash
python core_runner.py --extend F:\project\IFWI\outputs\core_missing3_20261016\runs\02_direct_prefit_sigma15\attempt_01\checkpoint.pt --epochs 4000 --out outputs\paper_supplement\X_direct_prefit_to4000 --shared-dir SHARED
```

```bash
python core_runner.py --extend F:\project\IFWI\outputs\paper_supplement_AK_0d9ef20_20261018_014331\runs\E_directFR_prefit_s3\attempt_01\checkpoint.pt --epochs 4000 --out outputs\paper_supplement\X_directFR_prefit_to4000 --shared-dir SHARED
```

Res-IFWI（论文 E2_residual，paper13_fresh_pair 批次）来自 `residual_ifwi.py`，它本身支持加长预算续跑：

```bash
python residual_ifwi.py --config F:\project\IFWI\.worktrees\offline-diagnostics-561c068-review\outputs\paper13_fresh_pair_20261012\01_paper13_ifwi\residual_seed3_20261012_091341_8f26bf\config.json --epochs 4000 --resume F:\project\IFWI\.worktrees\offline-diagnostics-561c068-review\outputs\paper13_fresh_pair_20261012\01_paper13_ifwi\residual_seed3_20261012_091341_8f26bf\checkpoint.pt --output-dir outputs\paper_supplement\X_E2_to4000
```

建议先各用 `--epochs 1001` 试一步，确认 checkpoint 能读入、身份校验通过、第 1001 步的 RMSE 与原轨迹衔接，再改成 4000。三组各约 5 小时。

## 审稿补充实验（`q13_review`）

对应审稿报告（report2）中需要新数据的意见，9 次运行，约 16–18 小时（Overthrust 上的 Res-IFWI-SE 已移到 q14，按达标口径运行）：

| 运行 | 回应 | 内容 |
|---|---|---|
| `R_E3_se_enc{3,1,2}_s3` | major 1：编码随机性、单种子 | RFS-IFWI（E3 + SE，与 q8 的 S1 同配置）用三组编码种子在同一会话里各跑 1500 步；种子 3 应复现 S1，用作同会话计时基准。报告 k*、t*、第 1000 步 RMSE 的均值与标准差 |
| `G1/G2/G3_direct_*`、`G1F/G2F/G3F_directFR_*` | major 8：直接表示失败是否源于 Adam 冷启动 | 预训练后分别用 20 步 warmup、保留预训练 Adam 矩、η=1e-5 重启；IFWI 与 FR-IFWI 各三组（G2F、G3F 为新增，其余与 q6_G 同名同配置，已跑完会被跳过） |

```bash
python paper_queue.py experiments/paper_supplement/queues/q13_review.json --out-root outputs/paper_supplement --shared-dir SHARED --wait-idle
```

无学习率衰减，编码运行的预算只决定停止点；精度仍按第 1000 步比较。

## 达标口径（time-to-target）：`--stop-rmse`、`target_plan.py` 与 `q14_overthrust`

论文改为比较“达到目标精度 ε 所需的更新次数 k* 与时间 t*”，并展示各方法在 k* 时的速度。`core_runner.py` 新增 `--stop-rmse ε`（或 spec 中 `"target": {"stop_rmse": ε}`）：第一次 RMSE ≤ ε 时保存 `target_velocity.npy`、`target_reached.json` 并停止。续跑是精确的（测试：从中间检查点续跑到 k* 与一次跑到 k* 的速度逐位相同）。ε：Marmousi 396.95，Overthrust 229.75，含噪 Marmousi 440.9375（相对 v0 降低 15%）。

**1. `target_plan.py`**：按 `experiments/paper_supplement/target_plan.json` 依次运行，已完成的跳过。

- `K_*`：从 k* 之前最近的检查点续跑到 k*，得到 Fig. 2/4 用的 k* 速度（每个只需几十步）。
- `X_*`：把没达标的运行接着跑，最多到 4000 步，达标即停（IFWI、FR-IFWI、Res-IFWI-SE、两组含噪 Res-IFWI；Res-IFWI 来自 residual_ifwi.py，无停止选项，跑满 4000 后从历史读 k*）。这些运行的 RMSE 在第 1000 步前后仍在上升，可能到 4000 也不达标。

```bash
python target_plan.py experiments/paper_supplement/target_plan.json --out-root outputs/paper_supplement/targets --shared-dir SHARED --roots SPEED=F:/project/IFWI/outputs/speed_q8_0c82ad3_20261018_1451 Q12=F:/project/IFWI/outputs/q12_e3_se_205c956_20261019_2208 MISSING3=F:/project/IFWI/outputs/core_missing3_20261016 AK=F:/project/IFWI/outputs/paper_supplement_AK_0d9ef20_20261018_014331 ROB=F:/project/IFWI/outputs/robustness_overthrust_noise_20261016 PAPER13=F:/project/IFWI/.worktrees/offline-diagnostics-561c068-review/outputs/paper13_fresh_pair_20261012/01_paper13_ifwi/residual_seed3_20261012_091341_8f26bf
```

先加 `--dry-run` 检查路径（尤其 Q12 下 `runs/<名称>/attempt_01` 是否存在）；也可用 `--only K_RFS_marmousi ...` 只跑部分条目。

**2. `q14_overthrust`**：表 III 中 Overthrust 还缺的四行，都带 `target`，最多 4000 步、达标即停：IFWI、FR-IFWI（直接表示 + 预训练）、Res-IFWI-MS、Res-IFWI-SE。

```bash
python paper_queue.py experiments/paper_supplement/queues/q14_overthrust.json --out-root outputs/paper_supplement --shared-dir SHARED --wait-idle
```

## 加速候选方案（`core_accel.py`，spec 字段 `accel`；队列 `q15_accel`）

| 选项 | 作用 | 依据 |
|---|---|---|
| `function_space` {velocity_step_mps, damping, cg_iters} | 函数空间 Gauss–Newton：一次正演+伴随得 g_v；在速度空间做 Adam 步（单位 m/s）；用共轭梯度解 (JᵀJ+λI)d = Jᵀdv 把它投影到网络参数（J、Jᵀ 只过网络，不做波场模拟）；λ = damping·tr(JᵀJ)/P | ENGD（ICML 2023）、KFAC-PINN（NeurIPS 2024） |
| `supershot_schedule` [{from, supershots}] | 超级炮数随迭代递增，`null` 为精确逐炮 | Friedlander & Schmidt 2012 |
| `time_window` [{from, fraction}] | 记录时长由短到长：早期只用每道前一部分（时间步少、单步快） | 课程学习 / 层剥离 |
| `coordinate_anneal` {start, updates} | 输入坐标乘以 c(k)（余弦从 start 升到 1），早期降低第一层频率 | BARF（ICCV 2021）、SAPE（NeurIPS 2021） |

全部选项不启用时与原运行逐位相同（测试 `test_accel_options_run_and_reduce_to_the_plain_update`）；Gauss–Newton 投影与稠密雅可比的阻尼正规方程一致（相对误差 <1e-3）。每条历史记录的 `accel`、`function_space` 字段给出该步的超级炮数、时间比例、坐标尺度、CG 迭代数、残差与 λ。注意：Gauss–Newton 每步多做 cg_iters 次网络正反传，开销请看日志中的 step 时间。

`q15_accel`（干净 Marmousi，均画全程曲线、不按目标停止；9 组，约 12 小时）：

| 运行 | 内容 |
|---|---|
| `P1_E3_fs_d1em1/1em2/1em3_s3` | Res-FR-IFWI + Gauss–Newton，阻尼 0.1/0.01/0.001，速度步 10 m/s |
| `P1_E3_se_fs_d1em2_s3` | RFS-IFWI + Gauss–Newton（1500 步） |
| `C1_E3_sched_s3` | 超级炮 1→2（300 步）→4（500）→逐炮（700），1500 步 |
| `B2_E3_se_cos_s3` | RFS-IFWI + 余弦学习率衰减（201–1000 步降到 0.1） |
| `C2_E3_coord_s3` | Res-FR-IFWI + 坐标由粗到细（0.25→1，前 200 步） |
| `W_E3_warmup_s3` | Res-FR-IFWI + 20 步学习率 warmup（检验早期 RMSE 上升是否只是步长过大） |
| `P4_E3_se_window_s3` | RFS-IFWI + 前 200 步只用前一半记录 |

```bash
python paper_queue.py experiments/paper_supplement/queues/q15_accel.json --out-root outputs/paper_supplement --shared-dir SHARED --wait-idle
```

建议先跑 `P1_E3_fs_d1em2_s3`，确认 Gauss–Newton 的单步开销和前几十步的 RMSE 走向，再跑其余。

## 背景诊断（spec 字段 `background: {kind: smooth400}`；队列 `q16_background`）

最早的 IFWI（2026-09-20，原作者 `IFWI2D` 入口）先预训练到 smooth400 背景，在约 1000 次更新内把 RMSE 从 537 降到 418 m/s，随后在第 2500–2600 次之间发散。论文中的 IFWI（`core_runner`，σ15 背景）则从 518.75 一路升到 835 m/s。两者有两处不同：背景和代码路径。`smooth400` 背景逐步复现 `pretrain_marmousi.py`：用 pandas 默认表头读取（首行被当作表头）、在原分辨率上做 Gaussian σ=50（reflect、truncate 4），再每 4 点取 1 点。与真值的 RMSE 为 537.36 m/s。该文件原始生成配方未知，因此不能声称它不含真值信息。

| 运行 | 内容 |
|---|---|
| `D1_direct_prefit_smooth400_s3` | 论文的 IFWI（直接形式，预训练 5000 次后重置 Adam），只把背景换成 smooth400，1000 次更新，约 1.6 小时 |
| `D2_E3_se_smooth400_s3` | 可选：PFS-IFWI 在同一背景下 1500 次。只在 D1 表明背景起作用时才需要，用于同背景对比 |

判读：D1 若与最早的结果一致（约 1000 次时降到约 420 m/s），说明差异来自背景，论文中"直接网络失败"的结论需要限定为 σ15 背景；D1 若仍然发散，说明差异来自代码路径，需要逐项对齐 `IFWI2D.train` 与 `core_runner` 的损失、初始化和优化细节。

```bash
python paper_queue.py experiments/paper_supplement/queues/q16_background.json --out-root outputs/paper_supplement --shared-dir SHARED --wait-idle
```

先只看 D1；D2 可在 D1 结束后视结果决定是否继续。

## smooth400 背景上的试点（队列 `q17_smooth400_pilot`）

D1 表明 IFWI 在 σ15 背景上失败、在 smooth400 背景上收敛，原因在背景而不是代码：smooth400 背景下 1000 次更新后 RMSE 为 415.4 m/s，与最早的 418 m/s 基本一致。smooth400 的生成方式可以从数据反推：在原分辨率上对慢度做 σ=36 点（约 135 m）的高斯平滑，结果与该文件的 RMS 差为 2.6 m/s；再接 σ50 速度平滑和 4 倍下采样后，与 D1 的背景差 1.4 m/s。它的垂直双程走时误差为 13.5 ms，而 σ15 背景为 43.6 ms；速度 RMSE 前者反而更大（537 对 519 m/s）。

试点在同一会话中、smooth400 背景上依次运行消融链（各组只改了背景）：

| 运行 | 方法 |
|---|---|
| `B_E2_smooth400_s3` | P-IFWI，1000 次 |
| `B_E2MS_smooth400_s3` | MS-IFWI，1000 次 |
| `B_E3_smooth400_s3` | FR-IFWI，1000 次 |
| `D2_E3_se_smooth400_s3` | PFS-IFWI，1500 次 |
| `D1_direct_prefit_smooth400_s3` | IFWI 重跑（可选）：RMSE 轨迹直接沿用已完成的 D1；只有需要同会话的单步耗时才跑 |

ε 沿用原规则：取非 Fourier 方法（IFWI、P-IFWI、MS-IFWI）中最终 RMSE 最低者，k\*、t\* 事后从逐次记录中计算。约 7 小时，加可选的 D1 约 1.6 小时。之前完成的 D1 每次更新 6.91 s，高于同类运行的约 5.8 s；如果当时 GPU 上还有其他任务，它的计时就不能用于加速比。

```bash
python paper_queue.py experiments/paper_supplement/queues/q17_smooth400_pilot.json --out-root outputs/paper_supplement --shared-dir SHARED --wait-idle
```

## 收敛协议（队列 `q18_converge` 与 `converge_plan.json`）

审稿要求：所有方法都必须跑到收敛再比较，不能和没收敛的结果比。

- **背景**：Marmousi（干净和噪声）统一用 smooth400；Overthrust 用 `slowness_gaussian` 背景，即对真值慢度做 135 m 高斯平滑，再对速度做 187.5 m 高斯平滑。在 Overthrust 上，该背景 RMSE 为 403 m/s、走时误差 3.4 ms；σ15 背景分别为 421 m/s、19.2 ms。在 Marmousi 上，同一配方与 smooth400 的 RMS 差为 4.1 m/s。
- **停止**：`target.plateau {window: 300, rel: 0.005}`。当"迄今最低 RMSE"在最近 300 个 epoch 内的改进不足 0.5% 时，判定收敛并停止，最多 4000 个 epoch。也可以用命令行参数 `--stop-plateau 300 0.005`，`--extend` 时同样适用。
- **E_conv**：离线计算，取 RMSE 首次进入该方法收敛 RMSE 的 1% 以内的 epoch；T_conv 为对应的墙钟时间。
- **Ours**：E3 加上渐进编码（`accel.supershot_schedule`：从第 1 个 epoch 起 1 个超炮，第 300 起 2 个，第 500 起 4 个，第 700 起逐炮），与 q15 的 C1 相同。

`q18_converge` 中的新运行（共 13 组）：

| 组 | 运行 |
|---|---|
| Marmousi 干净 | `V_ours_smooth400`、`V_directFR_prefit_smooth400`、`V_E2_se_smooth400` |
| −6 / −12 dB | `V_ours`、`V_E3`、`V_E2`（σ2、σ4 各一组） |
| Overthrust | `V_ours`、`V_E3`、`V_E2`、`V_direct_prefit` |

续训（`converge_plan.json`，从 q16/q17 的 checkpoint 出发，到收敛为止）：MS-IFWI、P-IFWI、FR-IFWI、IFWI、FR-IFWI+SE。

建议顺序：先跑 `V_ours_smooth400` 和 `C_E2MS_smooth400`，确认 Ours 收敛后是否最好，再决定是否跑完其余各组。

```bash
python paper_queue.py experiments/paper_supplement/queues/q18_converge.json --out-root outputs/paper_supplement --shared-dir SHARED --wait-idle
python target_plan.py experiments/paper_supplement/converge_plan.json --roots Q17=<q17 输出目录> D1=<D1 的 attempt_01 目录> --out-root outputs/paper_supplement/converge --shared-dir SHARED
```

注意：`verification/residual_ifwi/test_reporting.py` 中有 4 个测试在本次改动之前就已失败（SSIM 相关），与本次改动无关。

## 试点 q19：放弃 FR（A）或稳定 FR（B）

q18 的结论：Fourier 层前期更快，但后期会失稳。FR-IFWI 每个 epoch 的速度变化中位数是 P-IFWI 的 4 倍；第 1400 次左右，单步变化达到 74–107 m/s，RMSE 从 403 升到 502，之后没有恢复。Ours 在第 440–450 次也出现同样的尖峰，最后停在 420 m/s。P-IFWI 收敛最好（355.7 m/s，跑满 4000 次时仍在下降）；P-IFWI+SE 在 420–380 m/s 区间最快。

新增：

- `accel.fr_collapse {at: N}`：在第 N 次更新前，把 Fourier 层精确合并成普通权重 W = ΛB（网络输出不变，改变量记录在 `fr_collapse.json`），并重建 Adam；之后按普通 SIREN 训练。从合并后的 checkpoint 续训，与不中断连续运行的结果一致（测试 `test_fr_collapse_keeps_the_function_resumes_exactly_and_plateau_is_per_stage`）。
- 平台判定按阶段计算：300 步窗口只统计当前阶段内的更新，阶段由超炮数切换和 FR 合并划分，避免切换阶段时被误判为收敛。q18 中噪声和 Overthrust 上的 Ours 受过这个问题影响。

`q19_pilot`（Marmousi smooth400，最多 4000 个 epoch，平台判定停止）：

| 运行 | 方向 | 内容 |
|---|---|---|
| `Y_E3_collapse300_prog` | B | FR 前 300 个 epoch，之后合并为普通权重，同时超炮数逐步增加 |
| `Y_E2_prog` | A | P-IFWI + 超炮数逐步增加 |
| `Y_E3_collapse300` | B | 只做合并（不用 SE），与 q18 的 P-IFWI 对比 |
| `Y_E3_cos` | B | FR + 余弦学习率衰减（第 201–2000 个 epoch 降到 0.1） |
| `Y_E2_cos` | 对照 | P-IFWI + 同样的学习率衰减 |

判断标准：谁先达到 P-IFWI 收敛精度的 1% 以内（≤ 359 m/s），并且明显快于 MS-IFWI 的 328 min。

```bash
python paper_queue.py experiments/paper_supplement/queues/q19_pilot.json --out-root outputs/paper_supplement --shared-dir SHARED --wait-idle
```
