# 扩散先验插件验证记录

基线：`feature/fourier-residual-ifwi@5e29d824da2f3f4f8e334fc0a98be5318e33efde`。2026-10-28，在 Python / PyTorch 2.10.0+cpu 环境进行以下验证。原始物理求解器未修改。

## 新增测试

`python -m pytest -q verification/diffusion_prior`：29 passed。

覆盖：DDPM 加噪/干净预测恒等关系；低秩初始化恒等和冻结范围；基础/adapter 权重加载与哈希；未训练权重显式许可；解析高斯/点质量先验的梯度符号与 m/s 缩放；重叠边界覆盖；裁块微批一致性；参数梯度比例与一阶下降界；固定权重在物理梯度为零时的语义；全局 RNG 隔离；噪声退火；插值控制；模型级训练/验证重复拒绝；训练恢复；LoRA 导出；原 FD 接入；每次更新只注入一次；反演续训逐位复现；替换同名先验权重后的恢复拒绝；受控配置一致性。

## 原套件回归

在未修改的独立基线工作树和修改后的工作树分别运行：

`python -m pytest -q verification/residual_ifwi`

两者均为 **271 passed, 6 failed**，失败测试集合完全一致：

1. `test_experiment.py::test_shot_accumulation_preserves_full_objective_and_adam_update`
2. `test_reporting.py::test_constant_offset_reports_physical_rmse_and_mae_without_clipping`
3. `test_reporting.py::test_identical_velocity_fields_have_unit_ssim_and_zero_errors[shape0]`
4. `test_reporting.py::test_identical_velocity_fields_have_unit_ssim_and_zero_errors[shape1]`
5. `test_reporting.py::test_identical_velocity_fields_have_unit_ssim_and_zero_errors[shape2]`
6. `test_stochastic.py::test_selected_microbatches_preserve_full_selected_gradients_and_adam_state`

其中四项报告测试的预期字典未包含原代码已经返回的 `r2` 字段；另两项在当前 CPU 环境对不同炮批划分的数值采用逐位相等断言而失败。未在本功能分支中改写这些原有断言或借此掩盖失败。**不能将本次验证表述为“原仓库全部测试通过”。**

## 没有验证的内容

没有训练可用于正式地质反演的扩散先验；没有全尺寸 Marmousi/Overthrust 反演结果；没有 CUDA 显存、速度或时间—精度优势证据；程序化生成器不是实测地质数据；未测试超级炮、粗网格、function-space 更新或 Muon 的耦合，因此配置入口暂时拒绝这些组合。

小规模训练与反演用于验证代码通路、数值约定和恢复行为，不代表恢复精度或论文复现成功。
