# Fixed 52-shot, 500-update experiment

The implementation starts from `2bc95d157637f035d19473be29b30922e6dfd14b` in an isolated worktree. Earlier 13-shot and dense/gated results are retained. This experiment adapts existing modules; it is not an original-author reproduction or a claim that their combination has published validation. See [provenance and dimensional differences](fr-relative-provenance.md).

## Controls and inputs

Each of clean and the same saved -12dB realization has seven cases: full raw, full with existing illumination preconditioning, global B1 with preconditioning, interleaved B2 with preconditioning, interleaved B4 with preconditioning, interleaved B4 with fixed background-relative perturbations (plain 50,049-parameter SIREN) and preconditioning, and interleaved B4 with fixed background-relative perturbations plus existing Fourier reparameterization and preconditioning. All use omega30, zero-output physical v0 initialization and ordinary Adam(lr=1e-4, betas=.9/.999, eps=1e-8). Full raw skips unused illumination observers. Full uses four 13-shot microbatches with the same vectorized solver and global 52 normalization; 13 is a measured feasible cap, not proof of the hardware maximum.

The matched control for isolating source encoding is the preconditioned full case. Plain and `interleaved4_relative` cases have 50,049 trainable entries; `interleaved4_relative_fr` has 3,146,625 with existing author-style FR defaults, isolating Fourier reparameterization from background-relative scaling. [Krebs et al. 2009](https://doi.org/10.1190/1.3230502) and [Schiemenz and Igel 2013](https://doi.org/10.1093/gji/ggt362) are source-encoding references, not evidence of convergence or speed for this local implementation.

The original data52 arrays are hash-checked and reused. Truth/background/wavelet, canonical update-zero checkpoint, input index and reference configuration manifests are bound to the frozen plan. Existing 200-step checkpoints are not silently reused. All new physical work is counted, including discarded warmups. Initial full-shot waveform is evaluated once at exact common v0, saved and reused with provenance for the other 13 cases.

## Bounded execution and accounting

| Stage | Actual planned forward fields | Backward fields | Adam updates |
|---|---:|---:|---:|
| correctness: actual 13 physical shots and all 52-source B4 vs serial rows |34|34|0|
| timing: seven cases, 2 warmups + 10 measured updates |1428|1428|84|
| formal: fourteen cases, 2 discarded warmups + 500 fresh updates |123168|119476|7028|
| total new work, excluding already-created data |124630|120938|7112|

Formal costs include 119000 training F/B, 476 warmup F/B and 3692 evaluation F. Evaluations comprise the shared initial 52F and 70 distinct trained-state full evaluations. Every 10 updates saves velocity/RMSE/SSIM; every 100 saves full waveforms and complete model/Adam/RNG checkpoints. Descriptive RMSE thresholds 500/475/450/425/400/385 m/s use first observed crossings and never govern stopping, tuning or checkpoint selection. Missing crossings are NOT_REACHED; no interpolation or promised equal-accuracy speedup.

The supervisor executes preflight, timing and formal sequentially only after matching successful receipts. It owns one immutable fresh output identity, child process handles, an exclusive GPU lock and exact child/controller receipt chains. It does not enumerate processes, kill stale PIDs, retry scientific steps or expand to1000. The separate explicit1000 entry requires a complete matching500 run and strict state/runtime/source recovery. A failed update records completed physical work and distinguishes a known Adam commit from an uncertain optimizer operation; unsafe states are never retried or checkpointed. A safe user stop saves only validated state. After completed GPU phases, one CPU-only acceptance process validates saved arrays and ledgers, with separate launch/exit evidence; it does not run NN/PDE/GPU work.

## CPU verification

On2026-11-13 the complete relevant27-module CPU regression passed351 tests with2 retained legacy warnings in93.59s. CUDA visibility was disabled and OMP/MKL threads set to1; workspace basetemp was used. Tests include existing author compatibility, vectorized source arithmetic, illumination, normalization, model/Adam/RNG recovery, new relative-scale and shift tests, all twelve mocked500-step cases and result-array acceptance. The final supervisor identity integration test exposed and fixed a duplicate event-time key before any GPU launch. Subsequent focused regression covers that fix and collector receipt compatibility; its exact output is preserved with the experiment records.

Run the focused entry/acceptance regression with:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
& 'F:\project\IFWI\.venvs\g2-paper\Scripts\python.exe' -B -m pytest verification/test_fr_relative_pipeline.py verification/test_fr_relative_suite.py verification/test_fr_relative_orchestration.py verification/test_fr_relative_analysis.py -q -p no:cacheprovider --basetemp '<fresh workspace directory>'
```

Prepare a frozen plan with `python -m experiments.fr_relative_suite --stage prepare` and explicit output/index/reference/canonical/dense-data paths. `python -m experiments.fr_relative_pipeline --help` describes the bounded queue. An explicit `--execute` is required; no training is launched merely by importing or reading these modules. GPU arithmetic, memory, timing, checkpoint recovery and convergence remain measured acceptance gates/outcomes, not conclusions of CPU tests.

## Outputs and limitations

The pipeline root contains the immutable request/plan, source/data digests, LAUNCH/EXIT and child handle receipts, synchronized timing estimate, per-phase ledgers and per-case status/arrays/checkpoints. The [CPU collector](fr-relative-analysis.md) produces cases, curves, timing, updates, stage and phase costs, threshold times, JSON summary and a Markdown report in fresh `analysis500`. It independently recomputes every saved model curve and initial/final waveform metrics, while explicitly retaining intermediate waveform scores as producer-recorded. Training time, evaluation time, per-case elapsed and complete pipeline wall are distinct quantities. There is one seed, one model and one acquisition; clean-reference errors on the noisy run are not held-out-acquisition evidence.
