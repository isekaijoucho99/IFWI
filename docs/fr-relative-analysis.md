# Acceptance and interpretation of the 500-update experiment

Run after the pipeline has matching successful terminal receipts:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'F:\project\IFWI\.venvs\g2-paper\Scripts\python.exe' -m experiments.fr_relative_analysis --pipeline-root '<completed pipeline root>'
```

The collector reads `EXPERIMENT_REQUEST.json` for `dense_data_directory`; `--dense-data` is an explicit alternative whose contents must still match the frozen manifest. `--source-root` can point to the frozen worktree if collection runs from another checkout. No inferred download path or model reconstruction is used.

Acceptance requires three unique child launch identities for preflight/timing/formal, matching child and pipeline receipts, unchanged source/plan/input/data hashes, complete physical attempt/complete ledgers, returned Adam-update sequences and all twelve complete500 cases. The single shared initial evaluation costs52F; later full-shot evaluations cost3120F. All failures, stop receipts, extra updates, missing/extra metrics and mismatching costs fail acceptance. Existing `analysis500` is never overwritten. Validation of a still-running or partial experiment is read-only and creates no analysis output.

NumPy/SSIM recompute every saved velocity curve (0,10,...,500), the one shared initial prediction, and all twelve final predictions. Initial waveform copies must be byte-identical. Intermediate waveform values at100/200/300/400 are producer-recorded; their raw arrays remain available but this collector does not perform unnecessary repeated large-array scoring. Snapshot hashes are checked against their publication ledger without loading or executing models. The collector performs0 PDE F/B,0 neural evaluations,0 GPU work and0 Adam updates.

Outputs are `cases.csv`, `curves.csv`, `timing.csv`, `updates.csv`, `stage_costs.csv`, `cost_by_phase.csv`, `threshold_times.csv`, `results.md`, `summary.json` and collection start/exit receipts. Curves carry update, cumulative synchronized training time, case elapsed time and cumulative evaluation time. Stage and pipeline receipts retain total preparation/condition-construction overhead separately. Collection CPU wall is additional and explicitly recorded.

Use `full_preconditioned` to compare encoding strategies with the same plain SIREN and illumination rule. `full_raw` versus `full_preconditioned` isolates that rule. The relative-FR arm changes parameterization and representation together; a positive or negative result cannot isolate either component. Parameter counts, optimizer bytes and peak allocation/reservation remain in the output.

Report every condition/method row, including deterioration. Clean and noisy waveform targets share the training acquisition and are not a held-out-shot evaluation. The predeclared500/475/450/425m/s RMSE thresholds are descriptive: report the first recorded crossing on the10-update grid without interpolation. If absent, report `NOT_REACHED` and leave time fields empty. No same-accuracy speed ratio follows from an unreached threshold, and a fixed500-update time ratio is a different comparison. Neither these thresholds nor truth metrics control training, checkpoint selection, extra iterations or parameters.

Cyclic shifts of the index-modulo-B partition only relabel groups for N52/B2 or4; they do not randomize group membership. Background-relative velocity scaling is a fixed perturbation parameterization, not exact logarithmic velocity or a complete Hessian correction. The experiment is an adaptation of existing components and has no assumed novelty or performance guarantee.
