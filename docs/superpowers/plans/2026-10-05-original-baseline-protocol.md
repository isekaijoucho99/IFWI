# Original Baseline Protocol Implementation Plan

> 历史计划（2026-10-05）：下面的任务、权限边界和验证状态属于当时的工作记录，不是当前执行指令，也不表示本次发布重新运行了测试或训练。预注册的单变量要求适用于严格批量矩阵；当前 `experiment.py` 单次入口允许组合修改炮数、深度、宽度和 omega。当前用法见[实验使用说明](../../../experiments/README.md)。

> **For agentic workers:** Use superpowers:executing-plans for the coordinator and test-driven implementation for the independent runner and guard tasks. The user explicitly requested implementation in this workspace; preserve existing dirty files and historical results, and do not commit, push, or launch full training.

**Goal:** Every formal parameter case uses the original random IFWI baseline preparation, training, logging, best-selection and numbered-checkpoint code; only the registered parameter changes.

**Architecture:** Reuse unchanged author core files and the byte-identical original wrapper as a reference. A shared parameterized entry constructs original IFWI2D and calls its train method. A pinned protocol guard rejects base-setting/source drift before launching, including dry-run and recovery.

**Tech Stack:** Python, PyTorch, NumPy, YAML; Windows PowerShell; project Python env.

**Spec:** User instructions on 2026-10-05: all foundational code and printing must match baseline, only parameters may change. Historical anchor is overnight_20260920_215223/random; spectrum design in docs/wavenumber-evolution-experiment-design-20261005.txt requires matching cadence.

## Global Constraints

- Preserve original ifwi_modules.py, rnn_fd.py, generator.py, plot_functions.py and canonical data bytes.
- Baseline 4x128, omega30,13shots,seed3,Adam1e-4,4001updates,interval100, full-shot/full-record input; keep author actual clipping and numerical handling unchanged.
- Registered treatments: shots25/49,depth6/8,width256/512,omega10/20/50; exactly one factor per configuration.
- Retain author Completed/Epoch/START/Forward modeling/Loading best/COMPLETED/Results output and preupdate-loss/postupdate-state best semantics.
- Original saved checkpoints completed updates1,101,...,4001; no overriding same last.pth instead of numbered model history.
- Modified-feature runners remain distinct; frozen historical suites remain untouched and must not be relabeled as original protocol.

## Review Focus

- Default CLI without completed-baseline flag must still use the verified baseline, not4x256/seed42.
- Reject uniformly drifted optimizer,loss,data,source or batch settings, not merely extra per-case changes.
- Resume original checkpoints with original optimizer/history and numbering; never silently convert schema2.
- Inference and artifact export must not consume training RNG or change printed model-selection semantics.
- Small validation must compare actual original train, all loss columns, stdout, weights, Adam state and retained snapshot files.

### Task 1: Original runner and behavior tests

Files: experiments/baseline_experiment.py; experiments/baseline_reference/ifwi_experiment.py; tests/test_original_baseline_runner.py.

- [x] Write/run failing tiny CPU behavior tests against current missing original runner behavior.
- [x] Preserve byte-identical original wrapper evidence and directly reuse its logging helpers.
- [x] Implement original model/data preparation and direct IFWI2D.train path; add silent metadata/artifact export only.
- [x] Verify stdout,loss,model/Adam state,snapshots and continuation against original train.

### Task 2: Baseline protocol guard

Files: experiments/baseline_protocol.py; tests/test_baseline_protocol.py.

- [x] Write/run failing tests for canonical settings, one-factor limits and pinned source rejection.
- [x] Implement baseline_config, validate_baseline_protocol and validate_original_sources.
- [x] Verify exact registered treatment acceptance and common-setting drift rejection.

### Task 3: Launch, import, recovery and documentation

Files: experiments/run_parameter_sweep.py; parameter_sweep.py; run_experiment.py; import_completed_baseline.py; configs/baseline.yaml and legacy_random_baseline.yaml; scripts/resume_parameter_sweep.py; supporting tests/docs.

- [x] Write/run failing CLI tests for correct defaults and rejection of batching/common-setting drift.
- [x] Route formal launches into the original runner; persist protocol/baseline identities and freeze source evidence.
- [x] Extend original import/recovery without changing historical schema2 execution.
- [x] Explain each formerly missed execution difference and current verification limits in a local audit report.
- [x] Run targeted tests then repository pytest; inspect failures and review diff before reporting completion.

No full-size optimizer training or GPU training is part of this task. Short CPU tests may update tiny synthetic models to verify actual behavior.
