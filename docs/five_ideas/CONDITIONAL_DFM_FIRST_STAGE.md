# Conditional DFM: first-stage usefulness and coupling checks

This adds an explicit **conditional** entry point. The older `ifwi_research.run`
DFM option still loads the legacy smooth-condition prior; it is not silently
reinterpreted as the masked conditional model.

The new entry answers two separate questions:

1. Are fixed conditional single-draw candidates useful relative to the current
   data-only estimate? Report physical objective, invalid-draw rate, candidate
   displacement, strong-anchor displacement, and optional all/weak/strong-region
   truth RMSE. No best-of-N or truth-based candidate selection is performed.
2. Does the prior actually change the inversion? Start guarded physics-only,
   conditional DFM, and optional oracle branches from identical model parameters,
   Adam state, RNG, observations, frequency stage and fixed update budgets. Report
   parameter differences, velocity differences, final physical objective, truth
   metrics, all guard outcomes and solve costs.

A nonzero gradient, a network call, or an accepted guarded step is **not** evidence
of geological improvement. Even a truth-fed oracle can disappear in the network
VJP, Adam preconditioning, FP32 rounding, or physical-only guard recovery. The
oracle is an explicitly truth-fed diagnostic control, not a trained prior or a
promised improvement bound.

## Run without inventing weights

```bash
python -m ifwi_research.validate_conditional_dfm \
  --synthetic-smoke --oracle --steps 3 --out outputs/conditional_plumbing
```

This uses a 10 x 18 synthetic grid and the repository's real RNN-FD. It does not
create or load any DFM checkpoint. `candidates.status` is
`not_run_missing_checkpoint`, there is no DFM branch, and the oracle is separately
labeled. The result only checks plumbing and failure interpretation.

## Use a persisted data-only baseline snapshot

First obtain a baseline `checkpoint.pt` with `ifwi_research.run`, using its
`00_baseline.json` research configuration and the exact baseline spec saved with
that run. Then:

```bash
python -m ifwi_research.validate_conditional_dfm \
  --baseline path/to/small_independent_baseline.json \
  --snapshot path/to/baseline_run/checkpoint.pt \
  --diagnostic-checkpoint path/to/conditional_training/best.pt \
  --steps 3 --draws 4 --nfe 1 --seed 3 --ratio 0.025 \
  --out outputs/conditional_first_stage
```

- `--diagnostic-checkpoint` is explicitly evaluation-only. It permits toy or
  unqualified exports while retaining their original qualification metadata.
  It does not qualify them or make them acceptable to the production loader.
- `--checkpoint` instead invokes the existing strict qualified-declaration
  loader. Neither mode verifies the scientific truth of a qualification claim.
- No checkpoint is required for a physics-only / optional oracle diagnostic.
- Add `--oracle` only when a separate truth-fed control is wanted. Ordinary
  candidates and updates never receive truth. The existing benchmark loader
  creates synthetic observations from the dataset truth, as before.
- This entry requires an independently chosen `background.kind=linear_depth`
  baseline initialization. Gaussian-smoothed truth, `smooth400`, and
  `slowness_gaussian` initializations are rejected rather than exported as
  data-only training conditions. Provenance declarations cannot prove that a
  user-chosen linear profile was never tuned using test truth.
- This is a stage-local CPU prototype: at most 4096 grid cells, 16 shots, 256 time
  samples, plus the existing sensitivity graph/work caps. Large production
  Marmousi grids are not supported by this entry. Oversized basic dimensions are
  checked before constructing the forward problem. GPU/production scaling
  remains separate work.
- The baseline spec, initial velocity hash and observation hash must match the
  snapshot. Snapshot model and optimizer states are restored before branching.
  This is a new diagnostic experiment, not an exact source-version resume.
- Comparisons must remain within one frequency stage. The learning rate is the
  fixed base learning rate in every branch; schedule multipliers are deliberately
  not replayed. The effective policy is saved in the report.

## Match training and inference conditions

At the branch point, the default condition is:

- background: a detached copy of the snapshot estimate;
- anchor: that same detached estimate, never test truth;
- binary anchor mask: high relative-sensitivity pixels at the snapshot;
- weak-region weights: the declared `SoftPriorConfig` sensitivity ramp.

An external, independently derived H x W background can be supplied with
`--condition-background file.npy --condition-provenance "recipe and source"`.
The script records hashes and exports the exact condition arrays as
`background.npy`, `anchor.npy`, `mask.npy`, and `illumination.npy`. Use these
**same recipes** for paired training examples; having compatible tensor shapes
alone does not establish a matching conditional distribution. Ordinary generated
samples use the Gaussian-start training time convention, not a corrupted-current-
estimate shortcut.

The numerical operator, source, acquisition and frequency are bound to the
sensitivity context. The existing benchmark operator's PML bound is inherited
unchanged. This is not a new estimate of physical boundary parameters or a
calibrated uncertainty map. Train/validation exact model hashes are checked
against the evaluation model, but family/crop leakage still requires independent
dataset governance.

## Outputs and interpretation

`summary.json` includes:

- fixed sample IDs, every valid/invalid candidate, valid-only mean physical
  objective, invalid fraction and diagnostic generation/physical solve counts;
- original checkpoint metadata, checkpoint hash, sampler seed/NFE/identity,
  complete effective soft-prior config, condition hashes and provenance;
- complete per-step histories for every branch, accepted/rejected state,
  physical-only recovery, actual gradient ratios and final model differences;
- source hashes, snapshot hash, sensitivity costs and the fixed LR policy;
- explicit `scientific_qualification=false` and `qualification_changed=false`.

Each branch also writes `<branch>_velocity.npy` and `<branch>_state.pt`. State
files are labeled diagnostic-only, not production-run checkpoints. The oracle
has its own schema, `oracle_truth_diagnostic_not_dfm`.

Read evidence in order:

1. If candidates are invalid, inspect reasons rather than averaging away failed
   draws. If valid candidates worsen weak-region error or data consistency, fix
   training/conditioning before turning up coupling.
2. Inspect `raw_proposal_norm`, `projected_proposal_norm`,
   `applied_proposal_norm`, `scale`, and actual `ratio`. Projection removes only
   the opposing part of the prior. Scaling caps its norm and never amplifies a
   tiny proposal up to the cap.
3. Inspect `step.update_kind`: `physical_only`, `prior_applied`,
   `physical_recovery`, or `rejected`. `prior_applied` means the composed-gradient
   optimizer branch was accepted, not that Adam produced a measurably different
   velocity. Compare `actual_model_difference`, velocity RMSE and parameter L2.
4. Only then compare final physical and evaluation-only regional errors against
   guarded physics-only. Improvement on one synthetic fixture is not geological
   generalization or scientific qualification.

Invalid generated values, physical-range violations and RMS-budget violations
raise a typed `InvalidProposalError`. The default engine policy remains
`prior_invalid_policy="raise"`; this diagnostic explicitly chooses
`"physical_only"` to continue under the physical guard and report the reason.
Contract errors such as wrong shape, changed operator, stage, or source identity
still stop the run. Cached targets are revalidated; failed refreshes do not
silently reuse stale proposals.

## Verification environment

Use `python -m pytest -q` for the full repository suite. On the available
PyTorch 2.10.0 CPU environment, the unmodified cbf7732 baseline already has 12
shot/code batching equivalence failures from MKLDNN shape-dependent rounding.
Disabling MKLDNN for the verification process makes the 21 affected tests pass;
no solver code or tolerances were changed for this work:

```bash
python -c "import torch, pytest; torch.backends.mkldnn.enabled=False; raise SystemExit(pytest.main(['-q']))"
```

Report the default-backend failures separately from that controlled-backend pass.
No independent geological training corpus, qualified weights, production-grid
validation or GPU benchmark is supplied by this change.

## Oracle and optimizer diagnosis

See [the measured Adam/precision/guard diagnosis](ORACLE_COUPLING_DIAGNOSIS.md), including fixed 5/10-update data-only snapshots. It separates cold-start Adam normalization, FP32 background addition, guard recovery, and actual coupled influence. No optimizer or ratio default was changed. Reproducible diagnostic drivers are included under `verification/five_ideas/`.
