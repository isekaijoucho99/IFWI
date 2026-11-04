# Conditional DFM training and qualification boundary

## What is implemented

`ifwi_research/train_conditional_dfm.py` trains the new
`MaskedTwoTimeTransport` baseline or optional `MaskedMultiScaleTransport` architecture from paired geological models and externally
supplied conditions. It does not retrain during IFWI and does not modify the old
`train_dfm.py`/`FrozenDFMPrior` interface. The output `state_dict` contains EMA
weights for the new `FrozenConditionalDFM` adapter. Compact `last_inference.pt`
and `best_inference.pt` export only metadata plus EMA weights; the full `last.pt`
and `best.pt` additionally retain optimizer/raw-model/RNG state for resuming. Every generated checkpoint
remains **unqualified**, including after a long run or a good validation score.
No pretrained geological model is supplied.

A condition has three channels shared by all samples within one group:

1. `(background_mps - training_mean_mps) / training_scale_mps`;
2. `((estimated_anchor_mps - training_mean_mps) / training_scale_mps) * mask`;
3. a binary estimated-anchor mask.

Anchors are estimates, not known-correct pixels. They are not hard copied into
output. High sensitivity is not proof of accuracy. Weak-region generation is a
soft geological proposal and must still pass independent physical consistency
checks in the inversion engine. Training here contains **no seismic PDE loss**.

## Mathematical contract and deviations from the reference

Reference inspected on 2026-11-04 at upstream commit
[`619663f0e7cbd17e58d98501daaf389d2810077b`](https://github.com/merry7cherry/drift-flow-matching/commit/619663f0e7cbd17e58d98501daaf389d2810077b):
[official grouped DFM implementation](https://github.com/merry7cherry/drift-flow-matching/blob/619663f0e7cbd17e58d98501daaf389d2810077b/image_generation/src/driftfm/methods/drift_flow_matching.py).
This is a new small geological adaptation of the algorithm, not the paper's
image architecture or a reproduction of its image-generation results.

For one geological endpoint `v`, immutable condition `c`, scalar ordered times
`t < r`, and `B` independent Gaussian source samples `z`:

- `x_t = (1-t) z + t v`, `x_r = (1-r) z + r v`;
- `y = T_theta(x_t,t,r,c) = x_t + (r-t) u_theta(x_t,t,r,c)`;
- log-space Sinkhorn independently balances uniform row and column masses for
  `(stopgrad(y), x_r)` and `(stopgrad(y), stopgrad(y))`;
- row-normalized plans form barycentric attraction and self-inclusive repulsion;
- `d = attraction - repulsion`;
- regress `y` against `stopgrad(y+d)` using half pixel-mean squared error.

No correspondence matching loss or gradient through the Sinkhorn target is used.
Group axes remain explicit. Separate groups never interact in pairwise matrices.
Groups accumulate gradients serially before one optimizer step, reducing peak
memory without reducing the number of samples inside each transport group.

Differences from the reference are explicit: spatial conditional CNN rather than
class-conditioned image-latent MLP; RMS-pixel rather than Euclidean distances
(equivalent kernel temperature conversion is `temperature * sqrt(pixel_count)`);
pixel-mean regression rather than coordinate-summed/adaptive weighted regression;
20% exact `0 -> 1` pairs and otherwise sorted sigmoid-standard-normal pairs;
one geological endpoint repeated per condition rather than multiple class
endpoints. The last difference is important: noise draws are not independent
geological realizations. Generalization and multimodal calibration are unproven.
At `r=1`, all target points in that group coincide with its single endpoint.

## Data preparation and leakage checks

Supply a JSON manifest with schema `ifwi_conditional_dataset_v1` and a `records`
list. Each record contains:

```json
{
  "model_id": "basin_a_realization_001",
  "family_id": "basin_a_parent_model",
  "split": "train",
  "spacing_m": 20.0,
  "condition_source": "data_only_reconstruction",
  "condition_provenance": "data-only IFWI recipe, acquisition ID, reconstruction commit and iteration",
  "clean_path": "models/basin_a.npy",
  "background_path": "conditions/basin_a_background.npy",
  "anchor_path": "conditions/basin_a_estimate.npy",
  "mask_path": "conditions/basin_a_anchor_mask.npy"
}
```

Paths are relative to the manifest. Arrays are finite `H x W` NPYs on the SAME
physical grid. Velocity units are m/s in `[100,15000]`; masks are exactly 0/1.
At least one full train and one validation model are required. Optional test
records are checked for overlap but never sampled, scored or normalized.

Split complete parent geological families BEFORE cropping. Every crop, rotated
version, perturbation or resampling of one parent belongs to the same family and
split. The loader rejects overlapping family IDs, repeated model IDs and duplicate
clean arrays using canonical float32 content hashes, including across test data.
All four input arrays are hashed and preserved in the run manifest. These checks
cannot infer false family declarations or detect every transformed duplicate;
provenance review remains necessary. The training/validation corpus itself must
also be independent of final FWI test models and parent geometries.

Use data-only reconstructions to produce backgrounds, estimated anchors and masks,
with acquisition, noise, solver, iteration and mask recipes matching inference.
Do not smooth hidden test truth to create an inference condition. A clean-derived
training degradation alone creates a train/inference distribution mismatch.
`condition_source="synthetic_smoke"` permits explicit synthetic fixtures only and
sets `training.toy=true`. Changing a string does not establish trustworthy data.
There is no automatic condition builder from clean arrays in the real trainer.

Normalization is a fixed pixel-weighted mean/population standard deviation fitted
ONLY on full training models, with a 1 m/s standard-deviation floor. It is reused
for all conditions, validation, inference and resume. No per-patch or per-inversion
renormalization occurs. Do not use image resizing as a physical scale conversion.

## Running and resuming

From the repository root with its Python dependencies installed:

```bash
python -m ifwi_research.train_conditional_dfm \
  --manifest /path/to/manifest.json --out /path/to/new_run \
  --spacing-m 20 --patch-size 64 --width 32 --group-size 16 \
  --groups-per-step 2 --steps 20000 --device cpu
```

The baseline architecture remains `masked_two_time_cnn_v1`. Select
`--architecture masked_two_time_unet_v1` for the optional compact three-resolution
U-Net with residual blocks and skip connections. It has the SAME condition, time,
normalization and checkpoint contracts. Its wider receptive field is a design
capability, not demonstrated geological improvement. Resume requires the same
architecture. At width 128 its approximately 7.94M FP32 parameters occupy about
30.3 MiB in the compact inference export; use the compact export for the loader
64 MiB file cap, not the much larger optimizer checkpoint.

CPU is the default. CUDA is supported by the code but was not run in this work.
For authorized GPU work, start with a short memory probe using the exact target
shape and hardware; record `peak_gpu_allocated_bytes`, then add reserved memory
and the IFWI solver's simultaneous allocations when sizing the combined workflow.
The trainer does not prove an entire IFWI+DFM pipeline fits a 16GB GPU.

Each step holds one group, one condition and one model patch on device. With the
64 x 64, group-16, width-32 defaults, one FP32 width activation has 2,097,152
values, 8 MiB. Several activations, gradients, convolution workspaces, EMA and
optimizer state are additional. Sinkhorn matrices are only 16 x 16 per group.
This is a design estimate, NOT a measured peak or a 16GB guarantee. Reducing groups
per step reduces work but does not reduce single-group peak; reduce patch size,
width or group size for peak memory, while recognizing the statistical tradeoffs.
No mixed precision is enabled; pairwise distance/Sinkhorn work is FP32.

Resume into a NEW output directory (avoids accidental checkpoint/log overwrite):

```bash
python -m ifwi_research.train_conditional_dfm \
  --manifest /path/to/manifest.json --out /path/to/resumed_run \
  --spacing-m 20 --resume /path/to/new_run/last.pt --steps 40000
```

Keep optimization/architecture/data options unchanged. Resume rejects changes to that configuration,
normalization, spacing or any input content/provenance fingerprint. Raw model,
optimizer, EMA and dedicated RNG state are restored. A CPU test establishes
bitwise equality of a 2+2-step resume and an uninterrupted 4-step run; this is not
a cross-device/CUDA determinism guarantee.

Checkpoint writes use a temporary file followed by rename. `last.pt` is the most
recent validation checkpoint; `best.pt` is the best checkpoint written in the
current invocation. On resume, the loaded initial EMA is re-scored with the current
selection metric/NFE and immediately saved as a complete `best.pt` plus compact
`best_inference.pt`. Subsequent validation can replace it. This remains portable
when only `last.pt` was moved: no previous directory or historical-best file is
required. `best_scope` explicitly says `current_invocation_including_resume_initial`;
it does NOT claim the best of the entire training history. The previous best score
and metric are retained only in labeled `resume_history` diagnostics; corresponding
historical weights are not carried. Keep previous run artifacts to compare the
whole history. Fresh runs select only post-update validation checkpoints.
Training crashes can lose steps since the last checkpoint. Changing only the
validation selection statistic or NFE is permitted on resume; the initial EMA is
always re-scored, never mixed with a historical score. Legacy checkpoints used the
two-draw mean at NFE 1; resuming with the new single-draw default records this
change explicitly. No automatic qualification flag is ever set.

## Validation that is less misleading

Validation uses fixed seeds, held-out model center crops, two Gaussian draws, EMA
weights and the SAME externally supplied condition, at NFE 1 and 4. It starts
from pure noise at t=0, not 80% hidden truth. Diagnostics include weak-mask RMSE,
background weak-mask RMSE, explicit single-draw and ensemble-mean RMSE, finite-difference boundary error in
m/s per cell, sample diversity, and fraction outside the physical velocity range.
Invalid outputs are not clamped to make the metric look better. Weak means the
complement of the supplied mask, not verified true low illumination. Checkpoint
selection now defaults to `--selection-statistic single_draw --selection-nfe 1`,
matching the frozen adapter's single generated proposal. Per-draw squared errors
are summed and divided by draw/pixel count before the square root. The optional
`ensemble_mean` statistic is only for an explicitly ensemble-mean deployment.
`sample_weak_rmse_mps` and `sample_rmse_mps` are the single-draw metrics;
`ensemble_mean_weak_rmse_mps` and `ensemble_mean_rmse_mps` are mean metrics.
Historical `weak_rmse_mps`/`rmse_mps` fields remain ensemble-mean aliases, explicitly
labeled in the result, so old reports are not silently reinterpreted. When no
weak pixels exist, the corresponding global RMSE is used. `rejected_draws` and
`total_draws` report whole proposals failing the physical-range check.

`initial_validation.json` and the run summary expose initialization, final raw
weights, final EMA weights and comparison against the background. These are
learning diagnostics, not an automatic convergence or qualification certificate.

Two draws, one crop/model and ensemble-mean RMSE cannot establish useful individual
samples, full-model quality, fault preservation, coverage or a calibrated posterior.
The baseline three-layer 3x3 CNN has only a 7x7 single-step receptive field.
The optional U-Net addresses this architecture limitation but its full geological
completion performance still requires independent qualification.

## Formal prior qualification still missing

Do not change qualification metadata just to load the model. Required independent
research work includes:

1. A licensed, geologically diverse, parent-family-independent corpus, matched
   physical scales and acquisition/condition provenance, with untouched test set.
2. Full-model inference and patch-boundary checks; faults, layer continuity, edge
   position/contrast and velocity bounds; out-of-distribution and bad-anchor tests.
3. Data-only vs DFM vs ordinary smoothness/TV vs unconditional prior controls,
   multiple seeds, matched wall time/forward-solve budgets, multiple noise levels.
4. True physical data residuals and held-out shot residuals, not only encoded
   training residual or image RMSE. Accept proposals only under the inversion
   engine's non-degradation/trust-region guard; include a zero-prior fallback.
5. Low/high-illumination stratification, biased-anchor robustness, NFE and strength
   ablations, and explicit rejection of geological hallucinations despite lower
   waveform residual. Sensitivity is not posterior certainty.
6. Ensemble calibration on independent realizations and conditions before any
   uncertainty claim. Single-endpoint-per-condition training alone is insufficient.
7. Measured peak VRAM and runtime on the target 16GB hardware, including IFWI;
   document concurrent/sequential allocation behavior and safe memory margins.
8. A review report with immutable reference and SHA256, exact checkpoint/data/code
   hashes and explicit scope. The loader checks this declaration, not its validity.

## Reproducible synthetic smoke

```bash
python tools/smoke_conditional_dfm.py --out /tmp/conditional_dfm_smoke --steps 20
python tools/smoke_conditional_dfm.py --out /tmp/conditional_dfm_unet_smoke \
  --steps 20 --architecture masked_two_time_unet_v1
python -m pytest -q verification/five_ideas/test_conditional_training.py
```

This fixture uses analytic arrays and deliberately imperfect estimated anchors.
It creates no licensed geological dataset, no seismic experiment, no qualified
prior, and no evidence of field-data improvement. All smoke checkpoints are
rejected by the production frozen adapter. No user computer, GPU or remote push
is required for the CPU smoke.

## Investigating a poor smoke result

A 20-step smoke checks execution, not pretraining convergence. Run the bounded
CPU diagnostic before attributing its error to the DFM objective or adaptation:

```bash
python tools/diagnose_conditional_dfm.py --out /tmp/dfm_diagnosis \
  --steps 1000 --seeds 3 7 19
```

See `CONDITIONAL_DFM_DIAGNOSIS.md` and its JSON result for the measured
2026-11-04 investigation, including the corrected per-draw metric and failures
under biased background conditions. The diagnostic still creates only synthetic,
unqualified weights.
