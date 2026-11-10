# Validation, 2026-11-10

Base: `isekaijoucho99/IFWI` main `9b0bdf47999e5bb51baacfb90dddc7c3224ae3f9`.
Environment: Python 3.12, PyTorch 2.10 CPU, one PyTorch thread. No large GPU run.

- `python -m pytest verification/natural_gradient -q`: **21 passed**. Includes dense-Jacobian references, batch normalization, tiny original FD end-to-end training, old Adam checkpoint branching, Adam/NG exact CPU model and sampling resume, failure accounting, no-state rejection, finalization-only resume, residual/absolute nonzero-head derivative equivalence.
- Both documented smoke CLI commands completed 2 updates and wrote outputs, metrics, ledger and checkpoints.
- `python -m pytest verification -q`: **184 passed, 2 failed**. Not a fully green repository suite.
- Both failures reproduced separately from an untouched `git archive` of the base commit under the same environment:
  - `verification/residual_ifwi/test_experiment.py::test_shot_accumulation_preserves_full_objective_and_adam_update`
  - `verification/residual_ifwi/test_stochastic.py::test_selected_microbatches_preserve_full_selected_gradients_and_adam_state`
  - Both fail existing strict `torch.equal` comparisons of observed tensors across microbatch layouts. They are not suppressed or tolerance-relaxed in this branch. This is an unresolved qualification limit of the original baseline/environment, not evidence of new algorithmic performance.
- Existing TorchScript deprecation warnings and original empty gradient-clipping-generator warning remain.
- Independent read-only review found two restore edge cases; both fixed and covered: invalid NG-to-Adam branch must not emit a resumable fresh-Adam checkpoint, and final-evaluation recovery must not require an extra training update.
- Independent reviewer reran core tests: 13 passed, no new blocking findings.

No claims about full-model convergence, GPU throughput, memory sufficiency, superiority over Adam, exact reproduction of a paper, or DFM benefits follow from these tests. Tiny sampled-batch losses are not comparable across steps without fixed-shot evaluation.
