# 验证记录

2026-11-02，项目 Python 环境，CPU、单线程 BLAS、CUDA 不可见：

```
python -B -m pytest verification/rom verification/residual_ifwi -q -p no:cacheprovider
193 passed, 1 warning in 51.81s
```

warning 来自未修改的 `ifwi_modules.py` 旧梯度裁剪代码，提示空 parameters generator。
完整输出保存在 `measurements/cpu_test_output.txt`。

覆盖内容包括加权 POD 数值秩、SPD 条件误差界、Weeks 已知逆变换、真实 MATLAB
导出对照、原 FDTD 黑箱 Schur 补、PML 外边缘、源注入相位、双面/单面填充、
双炮叠加、全基压力/速度梯度、方向有限差分、固定 buffer/checkpoint、非有限值
失败记录、正演子空间包含性质、极点探测与原始状态谱对照、局部反演失败时不执行
下一次 Adam。另运行独立 CPU 小网格 CLI 和完整 Marmousi GPU 基准。

原论文复现在 MATLAB R2020b 实跑，不依赖 Python 测试自己生成参考值。四十个
作者 MATLAB 文件与已验证 Zenodo 归档逐字节一致。

设计和实现经过独立只读复核；修正了非有限数值失败记录、逐案例原子保存和局部
试更新候选状态失败处理。数学条件界与半谱实化伴随记号已单独核对。

科学图检查：Python matplotlib；源静态审计 19 PASS、0 FAIL，唯一 WARN 是技术
报告使用 PNG 预览而非 TIFF 投稿栅格，PDF/SVG 为主要可编辑输出，因此接受该
WARN。PDF 字体实际最小值：误差/成本图 5.6 pt，波场图 7 pt，均超过 5 pt。
逐面板确认了线型/图例、对数轴、阈值线、三次技术计时范围、共同压力色标、独立
误差色标、深度方向及时间标记；无平滑或局部图像调整。

GPU 测试的源码快照/哈希、配置、日志及大型数组在 LOCAL_ARTIFACTS.json 列出的
独立运行目录。三组基准的 `source_hashes_unchanged` 均为 true。局部试更新保存
了实际执行版本的源码；提交版本随后补上数值异常记录，相关 CPU 故障注入测试
包含在上述 193 项中。正式基线文件的保留证据见 PRESERVATION.json。
