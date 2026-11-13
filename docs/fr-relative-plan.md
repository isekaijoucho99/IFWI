# 52-shot encoding and Fourier reparameterization experiment

Spec: user attachment `libfile_cba7fbb3c4788191960e90e5b8167252`, read in full on 2026-11-13. Base commit: `2bc95d157637f035d19473be29b30922e6dfd14b`. This is an adaptation experiment, not an author-code reproduction or a claim of novelty.

## Fixed decisions

Run 500 measured Adam updates per case, with an explicit 1000-update upper-limit/resume entry that does not run automatically. Fourteen cases: clean/noisy crossed with `full_raw`, `full_preconditioned`, `global1`, `interleaved2`, `interleaved4`, `interleaved4_relative`, `interleaved4_relative_fr`. All encoding arms use the existing illumination rule, so `full_preconditioned` is the matched encoding control. `interleaved4_relative` isolates fixed background-relative velocity scaling (`v0/rms(v0)`) on the plain 50,049-parameter SIREN, and `interleaved4_relative_fr` isolates Fourier reparameterization on top of `interleaved4_relative`.

Reuse the completed data52 clean/noisy arrays after manifest/hash/geometry checks; do not regenerate data. Plain and `interleaved4_relative` controls use the canonical update-zero SIREN weights and ordinary Adam. `interleaved4_relative_fr` uses the existing FR-INR-style author initialization (default Fourier settings), omega30, zero output, ordinary Adam, and fixed v0/rms(v0) residual scaling with sv=1000m/s. No gates, curriculum, static anchors or truth-driven settings.

Every formal case starts at update zero unless a complete compatibility audit proves exact continuation; the default decision is fresh runs, with all work counted. Interleaved groups use `core_speed.SourceEncoder(..., grouping='interleaved')` directly without redundant row-shift wrappers.

Record model RMSE/SSIM every 10 updates without PDE; full clean/noisy waveform metrics at updates 0,100,200,300,400,500. The exact common v0 initial waveform is evaluated once and copied with provenance to all fourteen cases; the other 70 full-shot evaluations are distinct trained states. Save model/Adam/RNG every 100 updates and at stage boundaries. Predeclare descriptive RMSE thresholds 500,475,450,425,400,385m/s; these are reporting thresholds, never stopping or selection rules. Report first observed crossing without interpolation and mark unreached values. Include synchronized training time and end-to-end time including evaluation, warmup, construction and checkpoint I/O separately. No held-out-acquisition claim.

## Tasks

1. Add and CPU-test a background-relative wrapper around the existing velocity model, preserving the existing Fourier implementation byte-for-byte. Audit author sources, exact trainable and buffer counts, zero-output initial velocity and parameter gradients. Owned files: `experiments/relative_velocity.py`, its tests and provenance notes.
2. Add an explicit illumination-preconditioning switch to the dense training step, preserving the default, and reuse `core_speed.SourceEncoder(..., grouping='interleaved')` directly without redundant row-shift wrappers. Owned files: `experiments/dense_encoding_physics.py`, focused tests.
3. Implement a separate plan/preflight/timing/formal/analysis runner using existing batched physics, strict checkpoint and suite-control infrastructure. Freeze source/data/config hashes, identity, exact budgets, synchronized timings, per-update progress, memory and terminal receipts. Prepare the 1000-update restore entry without automatic expansion.
4. Run relevant CPU regressions and independent code review. Then a small actual GPU full-micro13 and B4 oracle/direction check, 2 discarded warmup +10 measured timing updates for all six arms, and exact model/Adam/RNG resume check. If all gates pass, freeze the measured ETA/budget and launch one sequential formal queue; stop on an uncertain or failed update and preserve cost.
5. Deliver all twelve 500-update curves and final arrays, or a precise live state and read-only monitoring/acceptance arrangement. Keep all old results and failures. Do not push new commits.

## Review focus

Test shift relabeling versus changed membership; full raw gradient versus preconditioned direction; velocity-scale chain rule and immutable v0; exact ordinary-Adam checkpoint restore for FR; global N52 loss normalization across microbatches; no ledger/update replay or duplicate queue after uncertain execution; accurate F/B and initialization/evaluation costs.
