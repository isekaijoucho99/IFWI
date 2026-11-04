# Drift Flow Matching

Source: https://github.com/merry7cherry/drift-flow-matching

Pinned commit: 619663f0e7cbd17e58d98501daaf389d2810077b

Paper: https://arxiv.org/abs/2605.17244

The six functions from `_expand_marginals` through `_adaptive_matching_loss`
in `image_generation/src/driftfm/methods/drift_flow_matching.py` are copied
without body modifications into `ifwi_research/_vendor/dfm_sinkhorn.py`.
Project-specific imports and classes are omitted. See the accompanying MIT
license. The frozen reference fixture records the original source SHA256.

IFWI spatial conditioning, CNN backbone, validation protocols, proposal gradient
and physical review are adaptations in this repository, not author-provided
geological inversion methods. Old IFWI schema-v1 weights are incompatible.
