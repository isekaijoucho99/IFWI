# Five IFWI Research Ideas Implementation Plan

> Execution: inline, with test-first numerical checks and a final self-review.

**Goal:** Five isolated, runnable method cores with shared original physics and experiment recipes.
**Architecture:** Additive package; no changes to original scientific kernels or baseline protocols.
**Tech Stack:** Python, PyTorch, NumPy, existing repository dependencies, pytest.
**Spec:** `docs/superpowers/specs/2026-10-30-five-ideas-design.md`

## Global Constraints
- Physical velocity in m/s, grid/interface width in m; gradients must be explicitly in velocity or parameter space.
- No test-truth-based selection, no unreported full-shot scoring cost, no untrained prior silently enabled.
- One method per recipe; baseline old modules remain byte-identical.
- Independent original-operator tiny tests and explicit CPU-only evidence.

## Review Focus
- Zero/NaN gradients, Adam momentum reversing a descent gradient, and exact rollback.
- Odd image grids and changing physical sample spacing.
- Deterministic top-k bias, duplicate sampled shots and full-support probabilities.
- Interface crossing/initialization and second-order solver graph retention.
- Checkpoint incompatibility, stale cache, data leakage and silently dropped config keys.

## Tasks
1. Safety: `research_ideas/safety.py`; tests for conflict projection, cap, real-step Armijo, state rollback.
2. DFM: `dfm.py`, `train_dfm.py`; grouped drift, two-time map, patch sampling, train/val identity.
3. Routing: `routing.py`; compressed descriptors, top-k + corrected stochastic complement, RNG restore.
4. Wavelet: `wavelet.py`; Haar/inverse, odd grids, bounded trainable gain, zero residual invariant.
5. Interfaces: `interfaces.py`; ordered horizons, anchor cancellation, physical derivatives.
6. Gradient attention: `gradient_attention.py`; source-only features, cross-group training, bounded gate.
7. Integration: `physics.py`, `runner.py`; original FD adapter, all costs, independent recipes, restart.
8. Documentation and experiments: complete Chinese protocol, runnable commands, evidence and limits.

Each task: write failing numerical tests; run; implement; run passing tests; record evidence.
Final: full repository test command, unchanged-source hashes, self-review, create new remote branch,
commit only added files and verify remote SHA/file contents. No merge into existing branches.
