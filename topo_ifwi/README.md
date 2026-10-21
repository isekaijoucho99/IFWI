# topo_ifwi — 拓扑先验 IFWI 最小可行性实验

对应项目书 [docs/topology_ifwi_proposal_v0.2.md](../docs/topology_ifwi_proposal_v0.2.md)。代码自成一体，不依赖仓库原有的 RNN 正演或 SIREN 模块。

| 文件 | 内容 | 项目书章节 |
|---|---|---|
| `persistence.py` | 超水平集持续图（并查集 + 对偶求 H1）、阈值带损失 R_PH、软异常分数 | 4.2、4.3、5.5 |
| `wave.py` | 二维声学有限差分（时间二阶、空间四阶、阻尼吸收层），频带滤波 | 4.1、6.1 |
| `models.py` | 合成模型（细颈哑铃、两分离体）、相对阈值参数、残差 SIREN | 4.2、6.2 |
| `run_minimal.py` | 热启动、阶段 0 判定、B0–B4 结构阶段、候选审查、指标与日志 | 5.1–5.4、6.3、6.4 |
| `figures.py` | 阈值带损失示意图 | 4.3 |
| `tests/test_persistence.py` | 阶段 A 单元测试 | 3.1、6.2 |

## 运行

```bash
.venv/bin/python -m pytest topo_ifwi/tests -q
```

```bash
.venv/bin/python -m topo_ifwi.run_minimal --smoke --out outputs/topo_ifwi/smoke
```

```bash
.venv/bin/python -m topo_ifwi.run_minimal --model dumbbell --out outputs/topo_ifwi/dumbbell
```

```bash
.venv/bin/python -m topo_ifwi.run_minimal --model twobody --groups B0 B2 B3a B3b --out outputs/topo_ifwi/twobody
```

默认设备依次为 CUDA、MPS（Apple GPU）、CPU，也可用 `--device` 指定。每个进程约占 6 GB 显存或统一内存（反传保存全部时间步的波场）；在 16 GB 统一内存的机器上不要并发多个运行。

## 在 GPU 机器上运行预实验

依赖只有 torch、numpy、scipy、matplotlib、pytest，可用仓库已有环境。以下命令在仓库根目录执行（Windows 下把 `python` 换成环境中的解释器路径）。

先跑单元测试和冒烟测试：

```bash
python -m pytest topo_ifwi/tests -q
```

```bash
python -m topo_ifwi.run_minimal --smoke --out outputs/topo_ifwi/smoke
```

细颈哑铃模型（先验正确），全部六组：

```bash
python -m topo_ifwi.run_minimal --model dumbbell --out outputs/topo_ifwi/gpu_dumbbell
```

两分离体模型（错误先验压力测试）：

```bash
python -m topo_ifwi.run_minimal --model twobody --groups B0 B2 B3a B3b B4 --out outputs/topo_ifwi/gpu_twobody
```

汇总成项目书 8.2 节的表格：

```bash
python -m topo_ifwi.summarize outputs/topo_ifwi/gpu_dumbbell -- outputs/topo_ifwi/gpu_twobody
```

两个模型各自独立，GPU 显存足够时可以并行。把 `outputs/topo_ifwi/gpu_*` 目录（不含 `warm_state.pt` 也可以）拷回后即可填写项目书。

## 输出

每次运行写入 `--out` 目录：

- `log.txt`：进度与每 60 步的指标；
- `phase0.json`：热启动后的阶段 0 判定（是否定位主体、带内是否有拓扑错误）；
- `summary.json`：配置、建模误差下限、各组指标与计算量；
- `reviews_<组>.json`：每次审查的争议条形、候选残差、可行性、可分辨度 χ、决定类别和事后真值核对；
- `v_<组>.npy`、`panels.png`：速度结果与对比图。

真值只用于指标和事后标签，不参与任何选择；所有权重按阶段起点的梯度范数比确定。
