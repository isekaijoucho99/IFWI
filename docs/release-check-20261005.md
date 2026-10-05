# 发布检查：2026-10-05

## 对照范围

通过远程 `git ls-remote` 核对 GitHub `main` 为 `c5ce59af3c9988dc730a29de2e541c23d1572cd2`。本次发布使用独立分支 `feature/experiment-cli-release-20261005`，面向 `main` 提交 PR。

以下原始文件与该提交逐字节一致：`main.py`、`fwi.py`、`pretrain_marmousi.py`、`ifwi_modules.py`、`rnn_fd.py`、`generator.py`、`plot_functions.py`；三份原始数据、原预训练权重和 `source_manifest.json` 也保持一致。文档、依赖、忽略规则和发布清单已更新。

新增 `experiment.py` 是简洁的单次实验入口，默认调用原版随机初始化训练流程。预设修改炮数、隐藏层数、宽度或 omega，显式参数覆盖预设，也支持同时修改多个变量。默认公共设置见 [README.md](../README.md)。用户决定参数组合；做因果对照时建议仅改变一个变量。

批量参数实验在 `experiments/run_parameter_sweep.py` 中保留预注册单变量矩阵。它与独立改进方法框架使用不同配置和输出格式；后者通过 `feature_baseline.yaml` 明确自己的控制组，不与原始随机 IFWI 混用。

## 发布修正

- 根安装清单补全 PyYAML 和 scikit-image，增加测试依赖清单。
- Windows 辅助脚本改为传入结果目录，并可使用已激活的 Python 或显式解释器路径；Bash 入口使用 LF 换行。
- 恢复原版训练时校验保存/best 选择间隔，避免继承不同选择规则下的 best。
- 完成基线导入接受已知的 `backend=reference`、`gradient_clip=null` 字段，并保持对实际协议变化的拒绝。
- 基线源码搬运支持归档包自己的 `author_sources`；改进方法运行与套件汇总使用一致的源码哈希集合。
- 本地研究资产、浏览器配置、结果和环境不进入源码包。

## 验证与限制

在现有 Python 3.10 环境运行 `python -m pytest -q tests`：**148 passed，4 warnings，67.59 s**。四个警告均来自保持原样的作者核心中的耗尽参数生成器裁剪调用；测试核对了其原始行为。

发布白名单包含 **107 个文件，其中 56 个 Python 文件均通过语法解析**。上述 12 个原始文件身份、六项原版实验固定哈希、四份 PowerShell 语法、Bash 入口语法及公开文档相对链接均通过检查。入口帮助、预设列表、默认与四参数组合 dry-run、两套批量 dry-run，以及三个 legacy 入口帮助共九条命令退出码均为 0；dry-run 未创建训练目录。

源码 ZIP 解压到独立目录后，107 个交付文件与清单集合完全一致，106 个被清单覆盖的文件 SHA256 均吻合（清单不包含自身）。该目录未提供外部历史源代码、`.git` 或虚拟环境；使用现有解释器重新运行九条入口检查及完整测试，结果 **148 passed，4 warnings，65.24 s**。最终补充验证记录后，仅文档与相应清单更新，发布 ZIP 再次验证所有文件哈希。

原版核心中的历史梯度裁剪调用保留其原始行为；此次没有重新定义训练、打印、best 选择或保存流程。固定 seed 能控制本次随机数设置，不能保证不同设备或不同 PyTorch 构建逐位一致。

完整 4001 次训练及 25/49 炮显存需求未重新测量。没有从零创建依赖环境，也没有验证全部平台。原始 `main.py` 和 `fwi.py` 的 `fwi_random` 模式依赖未分发的历史随机初始模型文件；保留源码身份，因此没有改写该模式的路径。可移植的随机 IFWI 实验使用 `experiment.py`，传统 FWI 的平滑/噪声入口按原 README 示例使用。

历史改进方法验证文档按其日期保留；它们不是本次修订后重新训练得到的结论。源代码包的范围与哈希定义见 [packaging.md](packaging.md)。
