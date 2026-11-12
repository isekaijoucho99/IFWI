# Batched source-encoding comparison

This is a controlled project experiment, not a claim of a new published FWI
method. The original SIREN, Adam, finite-difference stencil/PML, cached13-shot
Marmousi acquisition, initial velocity and observations are held fixed.
The full13-shot control uses the same residual representation and instantaneous
illumination preconditioner as every encoded arm. Historical absolute-model
or unpreconditioned runs are not substituted for this control.

## Objective and physical batching

For physical residuals r_s, N=13 and code matrix C, every update minimizes
`sum((C r)**2)/(models*N*time*receivers)`. Full-shot C is the identity;
global B codes are independent signs divided by sqrt(B); interleaved B groups
contain signs only at source indices s%B=b. Groups have7/6 or5/4/4 sources.
Interleaved codes are not additionally divided by group size or B.
All have E[C.T C]=I. Group residuals are squared separately, removing
cross-group terms while leaving stochastic terms inside groups. The old raw
single-code MSE was larger by N in expectation; it is not reused here.

`SourceEncoder` constructs one solver with B channels. The original solver
uses [model,shot,z,x] with grouped convolution, not [shot,1,z,x]. All B fields
are advanced in one time loop with vectorized source accumulation. An
independent serial oracle uses identical C, sums its1/N objectives, velocity
gradients, raw source energy S and weighted-adjoint energy A. It compares raw
S/A, sqrt(S*A), and the resulting preconditioned direction before Adam can
conceal scale differences. Identity encoding also checks native13-shot FD.
Correctness thresholds are fixed in `encoding_batch_correctness.py`; failures
stop the suite without tolerance adjustment or unrecorded retries.

The relative illumination floor0.1 follows `_precondition` in
[DeepWave-KAUST/ifwi-pub](https://github.com/DeepWave-KAUST/ifwi-pub), local
frozen source SHA256 `f6b5601bc1f306dc2cc117e9c2546547ff5de05518571820688b545db00514fd`.
The energy observer and gradient-norm matching are the existing project
implementation; no EMA, additional gate, clipping or TV is introduced.
The upstream MIT notice is retained in `licenses/KAUST-ifwi-pub-MIT.txt`.
Spatial grouping has prior art, including
[Schiemenz and Igel (2013)](https://doi.org/10.1093/gji/ggt362).

## Frequency adaptation and its limits

The sixth arm is the user's specified two-SIREN adaptation, built from the
existing [SIREN](https://github.com/vsitzmann/siren) representation:
`v0 +1000*std*(low_omega6 + alpha(k)*high_omega30)`.
Both zero residual heads give exactly the same physical v0 as the controls.
The omega30 branch shares their canonical state. Before update k+1,
`alpha(k)=(1-cos(pi*clip(k/200,0,1)))/2`; after the update the progress buffer
is advanced before evaluation/checkpointing. Both branches always execute.

The half-cosine window is adapted from the coarse-to-fine window pattern in
[BARF's author code](https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/main/model/barf.py).
BARF gates input encodings; it does not publish this dual-SIREN FWI pipeline.
The adaptation doubles the trainable parameter count50049→100098 and its
optimizer state. A smaller omega is not a strict physical spatial low-pass
filter. This is a whole-adaptation comparison, not an isolated frequency
ablation, original-author replication or promised noise suppression.

## Fixed execution and evaluation

Use the existing cached clean and sigma4 observations, not regenerated data.
Sigma4 is nominally−12dB; the float32 records actually used measure
−12.036797dB. Raw and used-dtype hashes, source hashes, actual runtime and
verified Git commit are saved before computation. Canonical update0 has
empty Adam state and zero residual heads; the physical v0 is the cached
smoothed Marmousi model, not zero or uniform velocity.
The cache index labels this background `smooth400`; the old reference config
still contains `initialization.sigma=15`. The cached/canonical arrays agree
exactly and are authoritative here. No background is reconstructed from that
legacy sigma field, and the mismatch is retained in the provenance record.

Six settings (full13, global1, global2, interleaved2, interleaved3,
interleaved2+progressive) run on each condition, twelve cases total.
Correctness comes first. Timing uses two warmup and ten measured complete
updates per setting, with explicit CUDA synchronization, wall times,
CUDA-event times and peak allocated/reserved memory. Warmups, failed work,
checkpointing and validation are included in the cost ledger. No theoretical
B-fold speedup is substituted for measured time.

Training is exactly1000 updates per case, resumed in fixed blocks ending
at200,500,1000. Save full model/Adam/Python/NumPy/Torch/CUDA RNG checkpoints
every25 updates and at block ends. One full-shot prediction per case at
200/500/1000 supplies both clean/noisy waveform MSE; all arms share one
initial physical prediction. Report model RMSE/SSIM and both MSEs. These are
the same acquisition and are not independent held-out shots. No truth-based
hyperparameter selection, early stopping or noise-overfitting conclusion is
allowed. Negative outcomes remain in the results.

The CLI is `python -m experiments.batched_encoding_suite --help`. Run stages
`correctness`, `timing`, then `train`, each with a fresh output directory,
the frozen dataset index/reference config/canonical checkpoint, and an honest
code commit label. Training additionally requires `--preflight` pointing to
the two completed preflight directories. The three output directories must
share one parent, since that parent holds their common exclusive lock.
Data and checkpoints remain local and are not bundled in the code repository.
The driver restores between blocks inside the same run; it does not
automatically resume a different process. Following interruption, preserve the
old ledger and explicitly verify any recovery from a full checkpoint, recording
replayed work separately. An old PID or stale progress file is not an exit proof.

If all planned work succeeds, training is46000 forward and46000 adjoint
field propagations; scheduled evaluation adds481 forward fields; correctness
adds48/48 and timing276/276. These are planned quantities, not claims of work
already performed. Actual attempts/completions and timing are authoritative.

Each stage acquires the same exclusive GPU-suite lock and publishes its
actual PID/run ID, append-only accounting and terminal receipt. A matching
`STOP_REQUEST.json` stops cooperatively before the next update; the last safe
checkpoint is saved before `STOP_ACK.json`. Persistent metadata contention
fails clearly; retrying file publication never reruns Adam or ledger append.
No old restricted process queries, security settings or denied sessions are
required. Existing outputs and dirty worktrees remain intact.
