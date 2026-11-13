# Acceptance and interpretation of the 500-update experiment

Run after the pipeline has matching successful terminal receipts:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'F:\project\IFWI\.venvs\g2-paper\Scripts\python.exe' -m experiments.fr_relative_analysis --pipeline-root '<completed pipeline root>'
```

The collector reads `EXPERIMENT_REQUEST.json` for `dense_data_directory`; `--dense-data` is an explicit alternative whose contents must still match the frozen manifest. `--source-root` can point to the frozen worktree if collection runs from another checkout. No inferred download path or model reconstruction is used.

Acceptance requires three unique child launch identities for preflight/timing/formal, matching child and pipeline receipts, unchanged source/plan/input/data hashes, complete physical attempt/complete ledgers, returned Adam-update sequences and all fourteen complete 500-update cases (or resumed 1000-update cases via `--formal-dir`). The single shared initial evaluation costs 52F; later full-shot evaluations cost 3640F per 500 updates. All failures, stop receipts, extra updates, missing/extra metrics and mismatching costs fail acceptance. Existing `analysis500` / `analysis1000` is never overwritten. Validation of a still-running or partial experiment is read-only and creates no analysis output.

NumPy/SSIM recompute every saved velocity curve (0,10,...,target), the one shared initial prediction, and all fourteen final predictions. Initial waveform copies must be byte-identical. Intermediate waveform values at 100/200/300/400 are producer-recorded; their raw arrays remain available but this collector does not perform unnecessary repeated large-array scoring. Snapshot hashes are checked against their publication ledger without loading or executing models. The collector performs 0 PDE F/B, 0 neural evaluations, 0 GPU work and 0 Adam updates.

Outputs are `cases.csv`, `curves.csv`, `timing.csv`, `updates.csv`, `stage_costs.csv`, `cost_by_phase.csv`, `threshold_times.csv`, `results.md`, `summary.json` and collection start/exit receipts. Curves carry update, cumulative synchronized training time, case elapsed time and cumulative evaluation time. Stage and pipeline receipts retain total preparation/condition-construction overhead separately. Collection CPU wall is additional and explicitly recorded.

Use `full_preconditioned` to compare encoding strategies with the same plain SIREN and illumination rule. `full_raw` versus `full_preconditioned` isolates that rule. `interleaved4_relative` isolates fixed background-relative perturbation scaling (`v0/rms(v0)`) on the plain 50,049-parameter SIREN, and `interleaved4_relative_fr` isolates Fourier reparameterization on top of `interleaved4_relative`. Parameter counts, optimizer bytes and peak allocation/reservation remain in the output.

Report every condition/method row, including deterioration. Clean and noisy waveform targets share the training acquisition and are not a held-out-shot evaluation. The predeclared 500/475/450/425/400/385 m/s RMSE thresholds are descriptive: report the first recorded crossing on the 10-update grid without interpolation. If absent, report `NOT_REACHED` and leave time fields empty. No same-accuracy speed ratio follows from an unreached threshold, and a fixed-update time ratio is a different comparison. Neither these thresholds nor truth metrics control training, checkpoint selection, extra iterations or parameters.

Background-relative velocity scaling is a fixed perturbation parameterization, not exact logarithmic velocity or a complete Hessian correction. The experiment is an adaptation of existing components and has no assumed novelty or performance guarantee.
