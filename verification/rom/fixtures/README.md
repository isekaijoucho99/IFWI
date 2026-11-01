# MATLAB reference provenance

The compact NPZ fixtures were exported from actual MATLAB R2020b executions on
2026-11-02, not generated using the Python functions under test.

Upstream: https://doi.org/10.5281/zenodo.15033159, GitHub tag `v_2`, archive
directory `mhamzakhalid-Model_order_reduction_for_Seismic_Applications-3f4ba89`.
Verified original ZIP MD5: `6ff9e0dfa1275412fc90875c2aa1d817`.
The upstream BSD-3-Clause license is retained in
`experiments/rom/LICENSE_SIAM_MOR`.

- `upstream_matlab_v2_reference.npz`: reduced operators and held-out solves
  exported after unchanged `Figure_8_bc_Plot.m` ran on the supplied 232248-DOF
  mesh; wrapper `experiments/rom/reproduce_upstream.m`.
- `upstream_core_fixture.npz`: a small positive-definite two-DOF problem passed
  directly to upstream `getseismosTD_lam`; wrapper
  `experiments/rom/upstream_core_fixture.m`. Used to isolate Weeks ordering,
  normalization, and Laplace/Galerkin parity without needing MATLAB in CI.

The author's entire archive and paper PDF are not vendored. Large raw MATLAB
outputs and logs remain in the local reproduction directory; the public report
records their hashes and numerical reproduction results.
