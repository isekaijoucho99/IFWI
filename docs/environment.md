# 本次检查使用的环境

2026-09-29 从现有本地环境读取，未安装、升级或更改依赖。

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.18 |
| torch | 2.9.0.dev20250904+cu128 |
| torchvision | 0.24.0.dev20250905+cu128 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scipy | 1.15.3 |
| matplotlib | 3.10.7 |
| Pillow | 11.3.0 |

操作系统为 Windows。PyTorch/torchvision 是既有 nightly 构建，因此没有把该配对硬编码进通用 pip 安装清单。其他平台或新建环境尚未验证；本次打包检查不等同于完整 GPU 训练验证。
