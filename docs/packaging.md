# 源码打包记录

本次整理日期：2026-10-05。GitHub 对照版本：`main` 的 `c5ce59af3c9988dc730a29de2e541c23d1572cd2`。发布检查见 [release-check-20261005.md](release-check-20261005.md)。

## 交付范围

保留原始项目的七个 Python 文件、三份数据和预训练权重，加入单次参数入口 `experiment.py`、原版参数协议与参考日志包装器、独立改进方法框架、预设配置、结果整理与恢复脚本、CPU 回归测试、安装依赖和公开使用文档。原始七个 Python 文件与上述 GitHub `main` 逐字节一致。

Git 忽略规则排除本地研究资产库、浏览器预览配置、虚拟环境、实验结果、检查点序列、日志、临时文件、私有环境变量文件，以及内部开发计划和本地历史验证记录。七份内部文档和两个旧实验专用脚本从公开交付中撤出，本地文件保留。原仓库分发的 `weights/ifwi_pretrain_marmousi.pth` 仍在包内。本次源码包不附带历史训练结果。

`package_manifest.json` 记录本次所有交付文件（除清单自身）的字节数与 SHA256，同时记录对照分支及提交。`source_manifest.json` 保留 2026-09-29 初次整理时的原作者文件身份；原版实验另有不可随打包重算的核心哈希约束。

ZIP 从已提交的 Git 分支生成，只有一个 `IFWI_GitHub/` 顶层目录，使用相对路径。ZIP 的 SHA256 单独保存在本地发布目录；归档中不含 `.git`、环境或训练结果。

## 验证范围

发布检查覆盖原文件身份、Python 与 PowerShell 语法、入口帮助和参数 dry-run、CPU 上的正演/反传/优化器状态/续训/汇总回归，以及解压源码包后的清单和入口检查。完整训练、GPU 性能和全新依赖安装不在本次验证范围内。具体执行结果见发布检查记录。

## 初始包来源

2026-09-29 初始包来自本地 `IFWI_clean`，之后提交为上述 GitHub `main`。原说明存于 [original_notes.md](original_notes.md)，当时环境存于 [environment.md](environment.md)，第三方署名与来源存于 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。这些历史文档中的旧文件数量、输出目录与检查结论应按其日期理解。
