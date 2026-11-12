# Dense acquisition and hard-channel gating: validation and use

This branch starts from fa2cdc57966947eca00413679f249339e8841dc5. It adds
isolated dense entry points; the13-shot/dual-network suite and its checkpoint
schema remain available. No new GPU experiment was launched for this change.
The existing gpu_train_01 was neither stopped nor modified.

## Implemented changes

- Dense48 sources: x-indices26..261 step5; dense52:16..271 step5, on the
  existing nx288/dx15m grid. Spacing75m, spans3.525/3.825km. All sources honor
  a10-cell margin. A52-source request with20-cell margins is rejected.
- Full-shot batches use the same vectorized FD adapter as encoded B1/B2.
  Loss, raw velocity gradient, source energy S and adjoint energy A sum across
  chunks with globalN normalization before one illumination direction, one
  network VJP and one optimizer update. Full-source chunks use only their own
  K sources, avoiding redundant dense zero-weight injection work.
- A single50049-parameter SIREN uses fixed first-layer frequencies4..30,
  hard0/1 channels and a zero residual head. The first32 channels start open;
  `32+floor(96*clip(k/200,0,1))` are active after k completed updates.32/200
  are declared development defaults, not values selected from Marmousi truth.
- MaskedAdam uses ordinary Adam equations with integer elementwise ages.
  Locked first-layer rows/bias and next-layer input columns preserve values,
  moments and ages exactly, including weight decay. New channels start their
  own bias-correction clocks. Fixed-frequency and progress buffers, optimizer
  state and Python/NumPy/CPU/CUDA RNG are checkpointed under a separate schema.
- Stratified grouping redraws an independent group-label permutation inside
  each consecutive B-shot block and independent signs per update. Global,
  fixed interleaved and identity modes remain. Each column has one +/-1;
  unequal groups still use the original-shot mean. Adjacent shots across
  block boundaries can belong to the same group.

## CPU evidence

Combined relevant regression: **263 passed,2 warnings,94.44s,exit0**.
CUDA visibility was disabled; OMP/MKL threads were1. Temporary files were
confined to a verified fresh workspace directory. No CPU test launched GPU
training, timing, data generation or a real CUDA RNG generator.

The suite includes the153 fa2cdc5 regressions,16 stratified tests,13 gated
model/optimizer tests,26 dense geometry/physical tests,42 checkpoint tests and
13 entry-point/budget tests. Real small-grid FD compares all-at-once against
unequal microbatches for loss, raw gradient/S/A/H, parameter gradients and one
Adam update. Separate scalar Adam oracles verify local ages and weight decay.
Save/restore tests replay RNG and gated unlocking exactly. GPU orchestration
tests use real CPU models/Adam/control IO and replace physical GPU work only;
they verify2 excluded warmups,10 timing samples and exactly200 fresh updates.

The two warnings come from retained original regressions: an empty original
gradient-clipping generator and deprecated TF32 introspection in an old runtime
report helper. They are not failed assertions; the original source was kept.

Independent spec and code review passed after fixing two findings: CUDA RNG
format validation now occurs on independent generators before restoration,
and synthetic truth is explicitly float32 with shape/finite checks before
publication. A separate overflow regression prevents noisy float32 infinities
from being saved. No claimed GPU performance or accuracy follows from these
CPU checks. See [the scientific claims audit](dense-gated-evidence.md).

Reproduce the same relevant CPU run from the worktree (choose a fresh scratch):

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
$python='F:\project\IFWI\.venvs\g2-paper\Scripts\python.exe'
$cases=@(
 'verification/test_batched_source_encoding.py',
 'verification/residual_ifwi/test_experiment.py',
 'verification/residual_ifwi/test_paper_supplement.py',
 'verification/test_encoding_batch_correctness.py',
 'verification/test_encoding_illumination.py',
 'verification/test_encoding_training_objective.py',
 'verification/test_encoding_suite_data.py',
 'verification/test_encoding_training_step.py',
 'verification/test_progressive_siren.py','verification/test_suite_control.py',
 'verification/test_encoding_checkpoint.py','verification/test_batched_encoding_suite.py',
 'verification/test_stratified_grouping.py','verification/test_gated_siren.py',
 'verification/test_masked_adam.py','verification/test_dense_encoding_physics.py',
 'verification/test_dense_checkpoint.py','verification/test_dense_encoding_suite.py')
& $python -m pytest @cases -q -p no:cacheprovider --basetemp '<fresh workspace directory>'
```

## Prepared GPU checks; not executed

CLI: `python -m experiments.dense_encoding_suite --help`. Without `--execute`
every stage only prepares a plan. Only prepare/synthesize/timing/quick exist;
there is no1000-update mode. Execution checks the nominated previous run's
matching LAUNCH/EXIT identity and terminal status, refuses its existing suite
lock, then acquires that same lock atomically. It never queries or kills PIDs.

Actual CPU-only plans were prepared in the local
`preparations/dense_gated_20261112/plan48` and `plan52` directories, marked
PREPARED_NOT_RUN. They do not contain dense observations. The old13-shot
records cannot be relabeled as48/52-shot data. The separate synthesis stage
will create fixed-seed dense clean/noisy records, hash them and report measured
SNR and all data-generation F costs before any inversion comparison.

Default full batches12/13 reference the prior fa2cdc5 full13 measured allocated
peak10.123GB; dense cache overhead and the largest safe batch remain unmeasured.
`--batch-size` accepts larger values up toN for a future fair capacity/timing
check. These defaults do not establish that four batches are necessary, that
the baseline must be slower, or that encoding gives a4x wall-time speedup.

In a fresh terminal with normal GPU visibility, after the existing job has an
explicit terminal receipt, use three separate fresh output directories:

```powershell
$common=@('--dataset-index','<existing dataset_index.json>',
 '--reference-config','<existing reference config.json>',
 '--initial-checkpoint','<existing canonical update_0000.pt>',
 '--shots','52','--batch-size','13')
python -m experiments.dense_encoding_suite @common --stage synthesize --execute --output '<fresh dense-data directory>'
python -m experiments.dense_encoding_suite @common --stage timing --execute --dense-data '<dense-data directory>' --output '<fresh timing directory>'
python -m experiments.dense_encoding_suite @common --stage quick --execute --dense-data '<dense-data directory>' --output '<fresh quick directory>'
```

Timing uses2 warmups plus10 measured updates per arm. Quick validation discards
the warmup model/optimizer and starts200 updates from the common initial state.
Seven arms compare full,global1,global2,interleaved2,stratified2,stratified2 with
all-open multifrequency features,and stratified2 with hard progression. The
last control distinguishes changed frequencies from progressive gating; all
have50049 parameters. Quick runs clean/noisy conditions with full-shot errors
against both datasets at0/200. Peak allocated/reserved memory, optimizer tensor
bytes, synchronized CUDA/wall samples, phase-separated F/B, warmup/training
optimizer counts, periodic complete checkpoints and terminal receipts are saved.

For52 shots, planned synthesis costs52F/0B; timing756F/756B and84Adam updates;
quick26908F/25452B with28 warmup plus2800 main Adam updates. These are budgets,
not performed work. Actual attempts/completions are authoritative. No claim
of improved accuracy, reduced cycle-skipping or strict spatial low-pass
behavior is made. New checkpoints have a tested restore helper; the driver
does not automatically resume a stopped process or silently change its batch.
