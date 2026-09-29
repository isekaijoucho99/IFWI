# IFWI 深层反演改进实验框架

针对 IFWI（Implicit Full Waveform Inversion）深层地层反演效果的改进实验框架。

## 论文背景

基于论文：**Implicit Seismic Full Waveform Inversion With Deep Neural Representation** (JGR Solid Earth, 2023)

## 改进目标

提升深层（特别是左下和右下角）地层的反演精度。

## 主要改进

1. **深度加权损失函数** - 给深层梯度更高权重
2. **注意力机制网络** - 自动关注深层特征
3. **自适应学习率** - 深层参数使用更大学习率
4. **组合策略** - 结合多种改进

## 快速开始

### 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r experiments/requirements.txt
```

### 运行实验

**单个实验：**
```bash
cd experiments
python run_experiment.py --config configs/baseline.yaml --device cuda:0
```

**批量运行（Linux/Mac）：**
```bash
cd experiments
bash run_all_experiments.sh cuda:0
```

**批量运行（Windows）：**
```cmd
cd experiments
run_all_experiments.bat cuda:0
```

### 对比结果

```bash
cd experiments
python compare_results.py --results-dir ../results
```

## 项目结构

```
IFWI/
├── experiments/
│   ├── configs/              # 实验配置
│   ├── improved_modules/     # 改进模块
│   ├── run_experiment.py     # 主运行脚本
│   ├── compare_results.py    # 结果对比
│   └── README.md             # 详细文档
├── data/                     # 数据文件
├── results/                  # 实验结果
└── test_framework.py         # 测试脚本
```

## 详细文档

请查看 [experiments/README.md](experiments/README.md) 获取完整使用指南。

## 预期效果

| 策略 | 深层误差降低 | 角落误差降低 |
|------|------------|-------------|
| 深度加权损失 | 20-30% | 30-40% |
| 注意力机制 | 15-25% | 20-30% |
| 自适应学习率 | 10-20% | 15-25% |
| **组合策略** | **40-60%** | **50-70%** |

## 系统要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+（推荐）
- 8GB+ GPU 内存

## 引用

如果使用此框架，请引用原始论文：
```
Sun, J., Niu, Z., Innanen, K. A., Li, J., & Trad, D. O. (2023). 
Implicit Seismic Full Waveform Inversion With Deep Neural Representation. 
JGR Solid Earth.
```

## 许可

遵循原项目许可协议。
