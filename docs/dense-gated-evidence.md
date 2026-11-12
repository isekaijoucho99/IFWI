# Dense acquisition and hard gating: evidence audit

Reviewed 2026-11-12. This is a project adaptation of existing methods, not a
claim of a new published method or original-author replication. The design is
[dense-gated-design.md](dense-gated-design.md). No CPU/GPU experiments were run
for this literature and algebra audit; predecessor measurements below are
transcribed from [the recorded preflight](batched-encoding-preflight.md).

## Recorded predecessor facts versus new claims

| Evidence class | Supported statement | Limit |
|---|---|---|
| Recorded fa2cdc5 preflight | Full13 median 3.188 s/update, peak allocated 10.123 GB; Global1/2 and Interleaved2/3 medians 2.983-3.055 s, 1.721-3.122 GB. Two warmups and ten measured updates per arm. | About 4-6% complete-update time reduction in this 13-shot configuration; no dense 48/52-shot timing or mandatory 4x speedup follows. GB is 10^9 allocated bytes. |
| Recorded fa2cdc5 correctness | Six GPU batch/serial/native cases passed; maximum relative velocity-gradient error 3.266e-7, direction error 4.065e-7; predictions matched exactly. | Historical implementation evidence, not verification of the new geometry, gate, or optimizer. |
| Recorded fa2cdc5 architecture | Baseline has 50,049 trainable scalars; the earlier dual-network adaptation has 100,098. | Fixed frequencies, masks and optimizer ages add stored state, but no trainable scalars to the new single-network skeleton. |
| Algebra | Full-shot microbatches can sum original-shot-normalized loss, raw gradient and S/A, then form one common illumination direction and take one update. | Applying nonlinear normalization separately per microbatch or taking multiple Adam steps changes the operation; finite precision can affect equality. |
| Untested hypotheses | Dense acquisition, stratification and progressive hard unlocking may improve time to a given reconstruction quality. | Need matched controls, measured wall time and reconstruction metrics; neither full-shot nor encoded training always wins. |

Source code alone does not establish a 3.5-5 s CPU injection cost or identify
the GPU timing bottleneck. That requires measurements that isolate injection
and synchronize asynchronous GPU work. Field-count savings are not measured
end-to-end speedups.

## Claims requiring qualification

| Claim | Audit finding |
|---|---|
| A soft gate cancels exactly in Adam | Only a fixed positive scaling of the entire gradient history cancels when epsilon is zero and states scale consistently. With epsilon, cancellation is approximate; a forward gate still changes the represented function. |
| Increasing alpha guarantees steps larger than learning rate | False. Updates depend on moment history; a zero gradient history gives zero updates for any alpha schedule. Overshoot can occur but is not guaranteed. |
| Omega 6 learns exactly one fifth as fast as omega 30 | False. The sine derivative contains omega times a cosine, and initialization, activations, downstream Jacobians and adaptive optimization all matter. Omega is not a learning-rate multiplier. |
| Fewer within-group source pairs determine the variance | Pair count alone omits residual/Jacobian correlations. Geometry and model state determine cross-term weights; no numerical reduction factor follows from counting alone. |
| Fixed interleaved groups necessarily bias the raw gradient | False with fresh independent signs and the stated normalization. Fixed grouping changes the noise structure; holding one encoding realization fixed generally changes the objective. |
| Stratified grouping separates every adjacent shot | Distinct labels occur within each consecutive B-shot block. Independently permuted blocks can put the same label on neighboring shots across a boundary (probability 1/B for full blocks). |
| Low first-layer frequencies or gates guarantee low-pass models and prevent cycle skipping | False. Learned affine weights and nonlinear sine compositions do not impose a strict physical wavenumber cutoff. Better convergence is an experimental hypothesis. |

The first two findings follow directly from [Adam, Algorithm 1 and section 2.1](https://arxiv.org/pdf/1412.6980).
For a fixed gradient history scaled by c>0, its update ratio is
`c*m_hat/(c*sqrt(v_hat)+epsilon) = m_hat/(sqrt(v_hat)+epsilon/c)`.
For time-varying alpha, the moments instead contain differently weighted
histories of `alpha_i*g_i` and `alpha_i^2*g_i^2`; a common factor cannot
generally be removed. These are algebraic comparisons at a given history,
not a guarantee that two training trajectories have the same gradients.

[SIREN's authors](https://www.vincentsitzmann.com/siren/) establish periodic
representations and an initialization scheme. Their
[initialization code](https://github.com/vsitzmann/siren/blob/master/modules.py)
distinguishes the first layer from later sine layers. Neither source
establishes the proposed hard unlocking schedule or an omega-to-learning-speed
law. [BARF's author code](https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/main/model/barf.py)
smoothly windows input positional encodings; it is related coarse-to-fine
prior art, not this hard-gated SIREN FWI implementation.

## Encoding expectation and noise

Let each shot occupy one group, `C[b,s] = sign_s * 1[group_s=b]`, with
independent symmetric signs redrawn each update, independently of the current
model and group construction. At a fixed model and grouping, diagonal entries
of `C.T*C` are one and off-diagonal expectations are zero, so `E[C.T*C]=I`.
Therefore `L_C = ||C*r||^2/N` and its raw gradient are unbiased for the full
original-shot mean, assuming a common receiver/time layout, source-linear
propagation and correspondingly encoded observations. Additional common
receiver/time/model normalization does not change this argument.

For real residual vectors at a fixed model, write `a_ij = <r_i,r_j>`. Then

```text
L_C - L_full = (2/N) sum_{i<j, same group} sign_i*sign_j*a_ij
Var_sign(L_C | groups, model) = (4/N^2) sum_{i<j, same group} a_ij^2
```

This audit's derivation explains why removing cross-group terms can help but
pair counts alone cannot specify a variance ratio. The analogous raw-gradient
cross terms involve derivatives of `a_ij`. Unbiased raw gradients do not imply
unbiased nonlinear illumination directions, Adam updates or trajectories:
generally `E[f(X)] != f(E[X])`.

[Schiemenz and Igel (2013), section 4.2 and Fig. 7](https://doi.org/10.1093/gji/ggt362)
compare fixed line-indexed, iteration-randomized, and fixed coarsely spaced
source partitions with random codes. Their setting supports spatial grouping
as prior art and the need to consider crosstalk, illumination and total
optimization cost. Its results do not establish a variance factor or speedup
for this grid, solver, optimizer or proposed stratification.

## Geometry, capacity and required controls

The actual grid has nx=288 and dx=15 m: node coordinates span
`(288-1)*15 = 4305 m` (the image-cell display width can be 4320 m).
With a 10-cell acquisition margin, 75 m spacing is five grid cells and fits:

| Shots | Source span | Example centered x indices | x coordinate endpoints |
|---|---:|---|---|
| 48 | 47*75 = 3525 m | 26,31,...,261 | 390-3915 m |
| 52 | 51*75 = 3825 m | 16,21,...,271 | 240-4065 m |

Both are legal within the margin and need no grid change. In particular,
52 sources at 75 m do **not** span 2.55 km. Coordinates must still be checked
against the selected run's declared margin and source/receiver depths.

For `[2,128,128,128,128,1]` with biases the count is
`(2*128+128) + 3*(128*128+128) + (128+1) = 50049`.
Hard masks preserve this count. Exact freezing must preserve masked parameters,
moments and local ages, including under weight decay; zeroing gradients alone
is insufficient. This is a local optimizer requirement, not a published SIREN
claim.

The new comparison needs the same dense acquisition for full-shot and encoded
arms, a static all-open multifrequency control, and matched initialization,
objective normalization, illumination and update budget. The requested future
two-warmup/ten-measurement timing and 200-update check remain pending GPU work.
No evidence here authorizes a 1000-update extension or establishes an accuracy,
noise-suppression or cycle-skipping advantage.
