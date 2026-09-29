# IFWI Improvements Experiment Framework

针对深层（特别是左下右下角）地层反演效果改进的完整实验框架。

## 📁 目录结构

```
IFWI/
├── experiments/
│   ├── configs/                    # 实验配置文件
│   │   ├── baseline.yaml          # 基线实验
│   │   ├── depth_weighted_loss.yaml
│   │   ├── attention.yaml
│   │   ├── adaptive_lr.yaml
│   │   └── combined_best.yaml     # 最佳组合
│   │
│   ├── improved_modules/          # 改进模块
│   │   ├── losses.py              # 损失函数
│   │   ├── networks.py            # 网络架构
│   │   ├── optimizers.py          # 优化器
│   │   └── evaluate_deep.py       # 深层评估
│   │
│   ├── run_experiment.py          # 单实验运行
│   ├── compare_results.py         # 结果对比
│   └── run_all_experiments.sh     # 批量运行
│
└── results/                        # 实验结果
    ├── baseline_YYYYMMDD_HHMMSS/
    ├── attention_YYYYMMDD_HHMMSS/
    └── comparison/                 # 对比分析
```

## 🎯 改进策略

### 1. **深度加权损失函数** (`depth_weighted_loss.yaml`)
- **原理**：给深层梯度更高的权重
- **适用场景**：深层能量衰减导致梯度过小
- **预期提升**：深层误差降低 20-30%

### 2. **注意力机制** (`attention.yaml`)
- **原理**：网络自动关注深层特征
- **适用场景**：复杂地质结构
- **预期提升**：深层细节恢复提升 15-25%

### 3. **自适应学习率** (`adaptive_lr.yaml`)
- **原理**：深层参数使用更大学习率
- **适用场景**：训练后期收敛缓慢
- **预期提升**：收敛速度提升 30%，深层误差降低 10-20%

### 4. **组合策略** (`combined_best.yaml`)
- **原理**：结合所有有效改进
- **适用场景**：追求最佳性能
- **预期提升**：深层误差降低 40-60%

## 🚀 快速开始

### 运行单个实验

```bash
cd experiments

# 基线实验（对照组）
python run_experiment.py --config configs/baseline.yaml --device cuda:0

# 深度加权损失
python run_experiment.py --config configs/depth_weighted_loss.yaml --device cuda:0

# 注意力机制
python run_experiment.py --config configs/attention.yaml --device cuda:0

# 自适应学习率
python run_experiment.py --config configs/adaptive_lr.yaml --device cuda:0

# 最佳组合
python run_experiment.py --config configs/combined_best.yaml --device cuda:0
```

### 批量运行所有实验

```bash
cd experiments
bash run_all_experiments.sh cuda:0
```

### 对比分析结果

```bash
cd experiments
python compare_results.py --results-dir ../results
```

## 📊 评估指标

### 深层专项指标

1. **深层相对误差** (`deep_relative_error`)
   - 定义：深层区域的平均相对误差
   - 越小越好，目标 < 5%

2. **左下/右下角误差** (`bottom_left_error`, `bottom_right_error`)
   - 定义：角落区域的平均相对误差
   - 越小越好，目标 < 8%

3. **深层 SSIM** (`deep_ssim`)
   - 定义：深层结构相似度
   - 越大越好，目标 > 0.85

4. **深层质量分数** (`deep_quality_score`)
   - 定义：综合质量评分（0-1）
   - 越大越好，目标 > 0.8

5. **梯度保真度** (`deep_gradient_fidelity`)
   - 定义：深层细节恢复能力
   - 越大越好，目标 > 0.75

## 📈 实验结果示例

典型的实验输出：

```
==================================================
EVALUATION RESULTS
==================================================
Deep Layer Relative Error: 3.45%      ← 基线: 8.2%
Deep Layer RMSE: 125.3 m/s            ← 基线: 287.1 m/s
Deep Layer SSIM: 0.892                ← 基线: 0.721
Bottom-Left Error: 4.12%              ← 基线: 12.5%
Bottom-Right Error: 3.89%             ← 基线: 11.8%
Deep Quality Score: 0.847             ← 基线: 0.612
==================================================
```

## 🔧 自定义实验

### 创建新配置

复制现有配置并修改：

```bash
cp configs/baseline.yaml configs/my_experiment.yaml
# 编辑 my_experiment.yaml
python run_experiment.py --config configs/my_experiment.yaml
```

### 可调参数

**损失函数：**
```yaml
loss:
  use_depth_weight: true
  depth_weight_params:
    weight_type: "piecewise"  # exponential, linear, piecewise
    deep_weight: 2.0           # 深层权重倍数
    bottom_weight: 5.0         # 底部权重倍数
```

**注意力机制：**
```yaml
model:
  network_type: "attention"
  attention_hidden: 64
  deep_bias: 2.0  # 深层注意力偏置
```

**学习率：**
```yaml
optimizer:
  optimizer_type: "depth_adaptive"
  learning_rate: 1.0e-4
  lr_deep: 5.0e-4  # 深层学习率（5倍）
```

## 📝 实验建议

### 推荐的实验顺序

1. **先跑基线** (`baseline.yaml`)
   - 建立对照组
   - 预计耗时：2-3小时（4000次迭代）

2. **单独测试各改进** 
   - `depth_weighted_loss.yaml`
   - `attention.yaml`
   - `adaptive_lr.yaml`
   - 确定各自贡献

3. **测试最佳组合** (`combined_best.yaml`)
   - 验证协同效果

### 调优建议

**如果深层误差仍然较大：**
- 增大 `deep_weight` 和 `bottom_weight`
- 增大 `lr_deep`（深层学习率）
- 增加训练迭代数

**如果训练不稳定：**
- 减小学习率
- 启用梯度裁剪
- 减小 `deep_bias`

**如果收敛太慢：**
- 启用学习率调度器
- 增大初始学习率
- 使用 `depth_adaptive` 优化器

## 🐛 常见问题

### Q: CUDA out of memory
**A:** 减小 batch size 或使用更小的网络：
```yaml
model:
  neuron: [2, 128, 128, 128, 1]  # 原来是 [2, 256, 256, 256, 256, 1]
```

### Q: 训练 NaN
**A:** 检查学习率是否过大，启用梯度裁剪：
```yaml
optimizer:
  learning_rate: 5.0e-5  # 减小学习率
```

### Q: 改进不明显
**A:** 
1. 确认深层权重设置足够大
2. 增加训练迭代数到 6000-8000
3. 检查数据质量（噪声水平）

## 📚 进阶主题

### Meta-Learning（计划中）

使用 MAML 等元学习方法，在多个速度模型上预训练，提高泛化能力。

### Transformer 替代 RNN（计划中）

使用 Transformer 进行正演模拟，解决长距离依赖问题。

### 多次波利用（计划中）

联合拟合一次波和多次波，充分利用观测数据。

## 📞 联系

如有问题或建议，请提 Issue 或联系开发者。

## 📄 引用

如果使用此框架，请引用原始 IFWI 论文：
```
Sun et al. (2023). Implicit Seismic Full Waveform Inversion 
With Deep Neural Representation. JGR Solid Earth.
```
