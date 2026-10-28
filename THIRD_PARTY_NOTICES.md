# 第三方来源说明

本包来自本地 `IFWI_clean`。作者原始文件头与内容保持不变。

- `ifwi_modules.py`：文件头署名 Jian Sun，Ocean University of China，Jan 2022。
- `rnn_fd.py`：文件头署名 Jian Sun，并要求引用 Jian Sun, Zhan Niu, Kristopher A. Innanen, Junxiao Li, Daniel O. Trad (2020), “A theory-guided deep-learning formulation and optimization of seismic waveform inversion”, GEOPHYSICS 85: R87–R99。
- `generator.py`、`plot_functions.py` 以及两份 Marmousi CSV：沿用本地原作者 v2 材料（目录编号 `11479806`）；历史 SHA256 在 `source_manifest.json` 中。
- `data/overthrust2d.npz`：本地原作者 v1 材料（目录编号 `7262564`）复制的基准模型。
- `main.py`、`fwi.py`、`pretrain_marmousi.py`：现有本地实验入口与补充实现，不应全部署名为原论文作者的发布代码。
- `weights/ifwi_pretrain_marmousi.pth`：本地现有平滑模型预训练权重；本次没有重新训练。

- `experiments/muon_optimizer.py`：按 Keller Jordan 的 Muon 参考实现（github.com/KellerJordan/Muon，仓库标注 MIT 许可；本次未能下载其 LICENSE 文件逐字核验）重新编写的 Muon 算法（Newton–Schulz 系数、动量与缩放规则）；未复制其辅助 Adam。使用方式依据 McGinnis et al. (2026), “Optimizing Rank for High-Fidelity Implicit Neural Representations”, ICML 2026, PMLR 306。

现有项目说明引用：Sun et al. (2023), “Implicit Seismic Full Waveform Inversion With Deep Neural Representation”, JGR Solid Earth, DOI: `10.1029/2022JB025964`。以上来源来自本地文件标注，本次没有重新进行外部文献核验。

没有在本次来源目录中找到明确的上游许可文件。本文件仅记录出处，不授予新的许可，也不改变原文件和数据的权利归属。

- `experiments/diffusion_prior/author_unet.py`: G²IFWI author architecture matches the provided `G2IFWI_code.zip` `synthetic/core/diffusion/diffusion.py`. Layer implementations are adapted from public `LinXueyuanStdio/PyTorch-DDPM` (`ddpm.py`); upstream GitHub does not advertise a repository-level license, while the user-supplied G²IFWI release describes its software as MIT licensed. This file adds gradient checkpointing and safety checks and is not a redistributable pretrained weight.
