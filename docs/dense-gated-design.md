# Dense acquisition and single-SIREN gating design

User authority: the supplied 2026-11-12 attachment, clarified execution scope:
implement and CPU-test only; prepare GPU checks; do not stop or modify the live
fa2cdc5 gpu_train_01 job, launch GPU work, or push this branch.

The new code is isolated from the old13-shot/dual-network entry point. Source
encoding and normalization reuse its vectorized adapter. Dense48/52 sources
are generated within the actual nx=288,dx15m grid and a declared acquisition
margin.75m is five cells and only used if the requested count fits. Full-shot
microbatches default12/13 and sum original-shot-mean loss, raw gradients and
S/A before applying one common illumination direction and one optimizer step.
Batch limits are configurable, not evidence of a measured memory maximum.

One existing SIREN skeleton retains50049 trainable scalars. Its first128
frequencies are fixed linspace4..30, with a binary channel mask:32 channels
initially, then32+floor(96*clip(completed/200,0,1)). All are open at200.
The first affine rows/bias and next affine input columns use elementwise Adam
ages and moments. Locked elements retain parameters, moments and ages exactly,
including with nonzero weight decay. Unlocking never uses a soft alpha.
This is a project adaptation; it is neither strictly band-limited nor a
published author implementation. A static all-open multifrequency control is
prepared to distinguish gating from the changed first-layer frequencies.

Stratified grouping permutes group labels separately inside each consecutive
B-shot block, then independently draws signs each update. Each shot belongs
to one group, E[C.T C]=I, and residuals are squared per group and divided byN.
Incomplete blocks create unequal groups; neighboring groups across block
boundaries may coincide. Global and fixed interleaved modes remain supported.

CPU oracles cover full versus microbatch loss/gradient/S/A/H/one Adam update,
source coordinates, exact finite-sign normalization, hard freezing, local
bias-correction ages, unlock, weight decay and exact save/restore/RNG. Explicit
GPU entry points will support two warmups plus ten measured updates and a
fixed200-update check, with real F/B, peak memory and synchronized wall times.
The default command only prepares a plan; no1000-update extension is enabled.

Measured facts from fa2cdc5 are retained as facts. The attachment's kernel
timing attribution, variance reduction, speedups and accuracy claims remain
hypotheses; algebra alone does not establish their numerical magnitude.
