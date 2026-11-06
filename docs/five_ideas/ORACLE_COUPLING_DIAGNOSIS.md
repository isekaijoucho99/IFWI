# Why the synthetic oracle has zero final FP32 velocity effect

## Finding

The oracle is connected and changes gradients, parameters, optimizer moments, and rejected trial models. At the default ratio 0.025, its final accepted velocity is nevertheless bitwise equal to the physics-only velocity after three updates. Three distinct mechanisms explain this result:

1. Zero-output initialization restricts both gradients to nine output-head coordinates. On the first Adam step, the protected proposal changes their magnitudes but none of their signs, so Adam nearly cancels the change through its componentwise normalization.
2. The remaining first-step parameter difference is nonzero, but its velocity effect is far below the FP32 resolution after adding the 2000 m/s background.
3. On updates 2 and 3, the proposal does produce observable trial-model differences. Every allowed Adam trial fails the strict physical decrease check. The accepted physical-only recovery omits the proposal and resets optimizer history.

This is a synthetic native RNN-FD diagnostic, not a trained-DFM test, geological result, or qualification. No production code, tests, algorithm defaults, optimizer, guard, or weights were changed.

## Setup and reproducibility

- Existing `synthetic_snapshot()`: 10 x 18 cells, 20 m spacing, 3 shots, 48 time samples at 0.001 s, 30 Hz wavelet, native order-2 FP32 RNN-FD, 3-cell PML.
- Constant 2000 m/s initial model; explicitly truth-fed oracle adds 150 m/s over the synthetic anomaly. The benchmark creates synthetic observations from truth and inherits a shared PML velocity bound of truth.max() = 2150 m/s. These are fixed across compared branches. Only the labeled oracle additionally receives truth as an update target; ordinary conditioning uses the independent/data-only estimate.
- Existing residual SIREN [2, 8, 8, 1], seed 3, zero final layer; Adam learning rate 0.0001 and default epsilon 1e-8.
- Identical initial model, optimizer, and RNG for matched branches. Guard permits four Adam trials, factors 1, 1/2, 1/4, and 1/8, followed by physical-gradient recovery.
- Existing sensitivity map, two probes, seed 3, prior normalization 1000 m/s, and existing velocity/edge penalty. One CPU thread.

From the repository root:

```sh
MPLCONFIGDIR=/tmp/ifwi-mpl XDG_CACHE_HOME=/tmp/ifwi-cache \
  python verification/five_ideas/diagnose_oracle_influence.py
```

Reproduction drivers are in `verification/five_ideas/`; they write full JSON/loggable output into `outputs/oracle-influence-diagnostic/`:

- `diagnose_oracle_influence.py`: self-contained diagnostic driver. Instrumentation wraps the existing guard in memory only.
- `diagnostics.json`: full gradients, replay comparisons, per-trial guard observations, controlled moment experiment, ratio sweep, and production-source hashes.
- `run.log`: concise execution output.

## 1. The gradient is not missing or initially too small to reach its cap

At the initial model:

| Quantity | L2 norm |
| --- | ---: |
| Physical velocity-space gradient | 5.6349407e-7 |
| Physical parameter gradient | 1.6394441e-3 |
| Raw oracle parameter gradient | 3.7868533e-2 |
| Oracle gradient after one-sided projection | 2.7328887e-2 |
| Oracle contribution after cap and FP32 addition | 4.0986123e-5 |

The raw oracle gradient is about 23.1 times the physical parameter gradient. Its cosine with the physical gradient is -0.69223, so projection removes its opposing component. The projected norm is still about 16.67 times the physical norm. The cap multiplies it by 0.0014997355, reaching the requested 0.025 ratio within FP32 arithmetic.

This is deliberately a maximum ratio, not an amplification target. Multiplying the raw proposal by 1000 changes the cap scale to 1.4997354e-6 but leaves the applied norm unchanged at 4.0986123e-5. In contrast, multiplying the raw proposal by 0.001 yields scale 1 and an actual ratio 0.0166696. At multiplier 1e-6, the actual ratio is approximately 1.66708e-5. Thus increasing an already capped penalty magnitude would not solve this case, while a genuinely weak raw prior would not be boosted to 2.5%.

## 2. First-step Adam normalization is the main attenuation

Only 9 of 105 trainable coordinates have a nonzero initial physical gradient; the oracle has exactly the same support. The zero output head blocks hidden-layer gradients. The nonzero physical magnitudes range from 8.07201e-5 to 7.58312e-4, well above Adam epsilon 1e-8. The protected proposal at ratio 0.025 flips no gradient signs and activates no new coordinates.

With initially empty moments, bias-corrected Adam's componentwise update is proportional to `g / (abs(g) + epsilon)`. Consequently the first step is almost sign-only. At the accepted learning rate 1.25e-5:

- Physics-only parameter update L2: 3.7498811e-5.
- Oracle parameter update L2: 3.7498837e-5.
- Between-branch parameter difference L2: 8.6788863e-11.
- Eight parameter coordinates differ. Parameter rounding therefore did **not** erase the entire proposal.

A controlled FP64 optimizer/INR replay, using the same captured FP32 gradients, gives parameter difference L2 8.6458924e-11. FP64 parameter arithmetic alone does not remove Adam's magnitude invariance.

An epsilon-zero sign-limit control, excluding all-zero gradient blocks to avoid undefined 0/0, gives FP64 parameter difference only 5.08220e-21. This isolates Adam normalization from parameter precision. This is not a proposed optimizer setting.

For comparison only, plain SGD at the same learning rate gives parameter difference 5.12327e-10, but its entire physical update is only 2.04931e-8. It is not a matched-progress optimizer comparison and does not justify replacing Adam. An epsilon 1e-6 replay changes three FP32 velocity pixels; it likewise is an arithmetic diagnostic, not a validated configuration recommendation.

## 3. FP32 background addition erases the surviving accepted velocity effect

At 2000 m/s, one FP32 ULP is 0.0001220703125 m/s. The accepted first-step branch differences are:

| Measurement | RMS difference, m/s |
| --- | ---: |
| FP32 network residual before background addition | 4.5601057e-8 |
| Full FP32 velocity after background addition | 0 exactly |
| Same final FP32 parameters decoded in FP64 | 4.5259934e-8 |
| FP64 optimizer and INR replay of fixed FP32 gradients | 4.6017232e-8 |

The maximum pre-background FP32 residual difference is 9.68575e-8 m/s, over 1200 times smaller than one velocity ULP. This direct residual-versus-full-output comparison identifies output quantization, rather than merely assuming that nonzero parameters must change the physical model.

The FP64 replay is **not** a FP64 physical inversion. Passing a double velocity directly into the existing native operator raises `RuntimeError: expected scalar type Double but found Float`. The current `RepositoryObjective.operator_identity` and `NativeSensitivity` contracts explicitly bind FP32/order-2. The underlying solver class exposes a dtype argument, but rebuilding a different operator would require separate validation and is outside this diagnosis. No kernels or PML coefficients were changed.

## 4. Later Adam trials have measurable influence but the guard discards them

Update 1 accepts both branches only at factor 1/8. Their accepted velocities and physical losses match exactly. The oracle nevertheless changes first-moment state by L2 4.0986086e-6 and second-moment state by L2 4.7528820e-11.

Update 2 starts at physical objective 1.07349433e-7. The branches' Adam trial velocities differ by RMS 0.00393246 m/s at the full learning rate and 0.000495685 m/s at factor 1/8. Thus the coupling can produce observable velocities once optimizer history exists. However, even the smallest allowed Adam trial has objective:

- Physics only: 1.15186944e-7.
- Oracle: 1.14761971e-7.

Both exceed the starting objective. The first physical-only SGD recovery is accepted at 1.07324937e-7; both accepted velocities are again identical. Recovery removes all six Adam state entries. The existing tiny parameter difference survives but does not affect the FP32 velocity.

Update 3 similarly rejects all four Adam trials. The factor-1/8 trial has a measurable between-branch velocity RMS difference 3.75144e-5 m/s, but its oracle objective 1.13488286e-7 exceeds the starting 1.07324937e-7. Physical-only recovery reaches 1.07153538e-7 for both branches and again leaves no optimizer history.

The final branch parameter difference is 8.6788863e-11, unchanged from the first step; final velocity difference is exactly zero. An `update_kind=prior_applied` label describes accepted use of a composed gradient, not proof of an observable velocity change. `physical_recovery` correctly identifies steps where the proposal is excluded.

### Isolating current-gradient and previous-moment effects

A four-case experiment holds the accepted first-step physics-only parameters, current physical gradient, and learning rate 1.25e-5 fixed. It varies only the previous moments (physics-only versus oracle) and current direction (physical versus composed).

- Adding the current proposal with physics-only moments changes trial velocity by RMS 0.000356706 m/s.
- Changing only previous moments, while retaining the physical gradient, changes trial velocity by RMS 0.000156802 m/s.
- All four cases fail strict physical decrease. Their physical first-order parameter changes are positive, approximately 9.36e-10 to 1.02e-9, so inherited Adam directions are already locally uphill in this controlled second-step comparison.

This distinguishes actual moment effects from output quantization. A one-sided guarantee about raw parameter gradients does not certify the Adam-preconditioned displacement or the finite physical step; the objective guard is necessary.

## 5. Ratio sweep shows that the oracle path is functional

Three guarded native RNN-FD updates with only the requested ratio changed:

| Ratio | Final velocity RMS vs physics-only, m/s | Oracle final objective | Accepted update kinds |
| ---: | ---: | ---: | --- |
| 0 | 0 | 1.07153538e-7 | physical, recovery, recovery |
| 0.025 | 0 | 1.07153538e-7 | prior, recovery, recovery |
| 0.1 | 0.0538194 | 1.05698328e-7 | prior, recovery, prior |
| 0.25 | 0.0538254 | 1.05706885e-7 | prior, recovery, prior |
| 0.5 | 0.0538324 | 1.05697758e-7 | prior, recovery, prior |
| 0.9 | 0.103283 | 9.95867635e-8 | prior, prior, prior |

At ratios up to 0.5 the initial protected gradient still flips no signs, and the first accepted FP32 velocities remain identical. At ratio 0.9 one initial sign flips and the first accepted velocities already differ by RMS 0.0163087 m/s. At ratio 0.1, visible divergence first survives the guard on update 3.

These are sensitivity controls, not a ratio-selection study. No ratio was chosen using truth metrics, and no default was changed. The small truth-RMSE differences in the JSON are evaluation-only observations on a simple synthetic fixture. They cannot qualify a learned prior or establish useful geological recovery.

## Implications for the main validation report

- Report gradient norms, actual cap ratio, parameter difference, pre-background residual difference, full velocity difference, and accepted-versus-recovery status separately.
- Do not attribute this zero to a missing oracle connection or an under-cap raw proposal; both are ruled out here.
- Do not equate a nonzero protected gradient or `prior_applied` label with useful model influence.
- Keep the FP64 replay labeled as optimizer/decoder arithmetic only, not FP64 native-wavefield validation.
- Keep guard strictness and non-amplification intact. Any future optimizer, epsilon, learning-rate, ratio, or line-search change needs a separately declared experiment rather than silently modifying the method to obtain a positive result.
- A later data-only snapshot with populated hidden-layer gradients and realistic optimizer history would be a different, useful coupling test. It must remain matched across branches and must not use truth to choose the snapshot or accepted updates.

## 6. Actual optimizer-setting and warmed-snapshot controls

The main Adam settings are learning rate 1e-4, betas (0.9, 0.999), epsilon 1e-8, weight decay 0, and AMSGrad disabled. The stage-local multiplier is 1; the guard alone scales individual trials. Raw SGD below has no momentum, weight decay, or adaptive preconditioner. The guard, physical operator, penalty, requested ratio 0.025, initial model, and three-update budget remain matched within each optimizer pair.

`python verification/five_ideas/diagnose_oracle_optimizers.py` generates `warmed_optimizer_controls.json` and its log. These are additional native physical runs, unlike the earlier frozen-gradient arithmetic replay.

| Optimizer setting from zero-head start | Final velocity RMS oracle vs matched physics-only, m/s | Physics-only / oracle final objective | Oracle accepted path |
| --- | ---: | --- | --- |
| Adam, lr 1e-4, epsilon 1e-8 | 0 | 1.07153538e-7 / 1.07153538e-7 | prior, recovery, recovery |
| Adam, lr 1e-4, epsilon 1e-6 | 1.57592e-5 | 1.06937931e-7 / 1.06972569e-7 | prior, rejected, rejected |
| SGD, lr 1e-4 | 4.26761e-5 | 1.28229836e-7 / 1.28230155e-7 | prior, prior, prior |
| SGD, lr 1e-2 | 7.76957e-4 | 1.01609370e-7 / 1.01687164e-7 | prior, prior, prior |

Same-learning-rate SGD has much less physical progress: total baseline velocity RMS change is 0.0007280 m/s, versus 0.0531657 m/s for default Adam. SGD at lr 1e-2 changes the baseline by 0.0405708 m/s. These settings were declared before inspecting results, not tuned or selected using truth. The table demonstrates sensitivity to optimizer and step scale, not optimizer superiority. The oracle has a slightly worse final physical objective than its matched baseline in all nonzero-effect controls in this table. Increasing epsilon can expose a few quantized pixels without making later guarded updates acceptable.

### Fixed data-only warmed snapshots

Two additional snapshots were taken after exactly 5 and 10 ordinary data-only baseline Adam updates. Warmup uses the existing unguarded baseline configuration, not oracle/truth-guided selection or the coupling guard. Its physical objective is not guaranteed monotone. Snapshot budgets were fixed before inspecting results; neither was chosen for favorable performance. Sensitivity and soft-prior spatial weights are recomputed at each frozen data-only estimate, consistent with the validation workflow. Thus these are realistic alternate snapshot controls, not an isolation of moments alone; the preceding four-case factorial is the moment-isolation experiment.

Both snapshots have all 105 physical parameter gradients nonzero, including all 96 hidden coordinates, and six populated Adam state entries. Their retained state step counts are 5 and 10 respectively. Three matched guarded coupling steps at the original ratio 0.025 produce:

| Data-only warmup updates | Final oracle vs baseline velocity RMS, m/s | Physics-only / oracle final objective | Oracle accepted path |
| ---: | ---: | --- | --- |
| 5 | 0.00344297 | 9.15280580e-8 / 9.19229217e-8 | prior, prior, recovery |
| 10 | 0.00384526 | 6.61646116e-8 / 6.68125577e-8 | prior, prior, prior |

The original 2.5% ratio therefore gives an observable, persistent model difference under Adam once the initial zero-head/empty-moment setting is removed. This rules out a general statement that Adam always eliminates this prior. It does not establish useful prior guidance: the oracle's final physical objective is again slightly worse than its matched baseline in both warmed controls, although every accepted individual update satisfies the guard relative to its own starting objective.

No optimizer, epsilon, learning-rate, guard, or ratio default was changed. L-BFGS, momentum SGD, RMSProp, AdamW, and convergence-scale comparisons were not run; the diagnostic supports no ranking among them.
