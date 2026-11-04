# DFM 20-step smoke: root-cause investigation, 2026-11-04

## Finding

The earlier result was not a test of a pretrained geological DFM. It started from
random/identity-initialized weights and ran only 20 optimizer steps. The model
had barely moved from initialization. This explains the large error on that
fixture without establishing that all adaptation or real-world issues are solved.

No change to the loss, time direction, normalization, architecture or optimizer
was needed for the same fixture to learn when given a longer budget. The loss
should not be replaced based only on the 20-step smoke.

## Controlled evidence

CPU, PyTorch 2.10.0+cpu, 16x16 patches, width 16, group size 8, two groups per step,
learning rate 0.0005, EMA decay 0.9. Keep the dataset and all training settings
fixed except budget and the declared initialization seed. Training/validation
are analytic fixtures, not real independent geological populations.

Corrected **single-draw aggregate weak-region RMSE**, m/s, at NFE 1 after 1000 steps:

| Backbone | Seed 3 | Seed 7 | Seed 19 |
|---|---:|---:|---:|
| CNN EMA | 71.50 | 68.28 | 70.37 |
| U-Net EMA | 45.54 | 44.62 | 65.79 |
| Supplied background | 83.94 | 83.94 | 83.94 |

For seed 3 at 20 steps, the corrected single-draw errors were CNN 394.01 and
U-Net 395.99. Earlier reported 365.03/367.07 values measured a two-draw mean,
which is a different, more optimistic quantity. At initialization the old
mean metric was 366.86; CNN raw/EMA at 20 steps were 361.93/365.03. EMA lag explains
only about 3.1 m/s of that CNN gap, not the failure to beat the background.

These 1000-step runs demonstrate that the implementation can learn this small
conditional task. They do not establish real geological performance, broad
convergence, uncertainty calibration, or a useful pretrained prior.

## Numerical checks

An independent audit checked the pinned official grouped DFM implementation:

- Analytic gradient `-drift / (G*B*H*W)` matches autograd within 1.17e-10.
- Observed Sinkhorn uniform-row mass error was 3.73e-8 on the tested case.
- Attraction/repulsion signs, detached target, Gaussian-to-geology time direction,
  normalization and masked-anchor inputs were consistent.
- A two-parameter endpoint-only conditional affine control learned approximately
  `[noise coefficient=0, condition coefficient=1]` in 300 CPU steps, with held-out
  normalized RMSE approximately 0.00029.

The affine control deliberately supplies a directly learnable synthetic condition;
it is not an IFWI or geological reconstruction experiment. No general claim of
bug-freedom follows from these checks.

## Fixed evaluation defect

The frozen production adapter emits one candidate per refresh, but training used
the error of a two-draw average as its only headline selection metric. This can
hide opposing sample errors and select the wrong checkpoint for deployment.

The trainer now:

1. Separately records per-draw aggregate and ensemble-mean metrics. It averages
   squared errors across draws/pixels before taking the square root.
2. Defaults checkpoint selection to the single-draw metric at the deployment NFE.
   The optional mean-based setting is explicit. Historical metric aliases retain
   their old meaning and are labeled, rather than being silently redefined.
3. Counts rejected whole draws as well as out-of-range pixels.
4. Records initialization, raw and EMA scores plus background comparisons.
5. Resets the best score when resuming with different metric/NFE semantics,
   including old mean-scored checkpoints, without resetting model/optimizer/RNG.

No qualification status is changed by these diagnostics.

## Remaining adaptation risk, measured rather than guessed

At 1000 steps, adding 200 m/s to the supplied background condition (not the target)
raised the CNN single-draw error to 238.51-308.12 m/s and the U-Net error to
141.38-208.46 m/s. Adding 200 m/s to estimated anchor values had a smaller effect
on this particular shallow-mask fixture. Removing anchors sometimes helped, and
NFE 4 was not uniformly better than NFE 1.

This is evidence of conditioning-distribution sensitivity, not proof of a code
bug. Arbitrary augmentation is not a demonstrated repair. Real pretraining must
use data-only reconstruction conditions with realistic correlated errors, masks,
physical scales, acquisition and iteration distributions matching deployment.
Biased-background and incorrect-anchor cases must be included in qualification.
The physical acceptance guard remains necessary and may correctly reject these
samples. No physical inversion experiment was run in this diagnostic.

## Reproduce

```bash
python tools/diagnose_conditional_dfm.py --out /tmp/dfm_diagnosis \
  --steps 1000 --seeds 3 7 19
python -m pytest -q verification/five_ideas/test_conditional_training.py
```

`CONDITIONAL_DFM_DIAGNOSIS.json` records the controlled-run settings, checkpoint
hashes, re-scored errors, conditioning ablations and learning curves. The original
1000-step runs used the old mean-based checkpoint selector; the reported table
re-scores their final EMA weights with the corrected single-draw metric. Selection
does not change optimizer updates. Do not compare an old best-score field against
a new single-draw score without stating the metric difference.

All checkpoints remain synthetic and unqualified. No user computer, GPU, field
data, or remote publication was used.
