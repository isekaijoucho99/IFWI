# Small dynamic-ROM experiment

User-authorized continuation of the fixed-basis SIAM/PML study, starting from
75abbe8. Goal: preserve native IFWI gradients over actual Adam steps and include
all online correction, monitoring, projection, refactorization and fallback
costs. No new network, modified FDTD, loss, learning rate or extra Adam steps.

## Methods and accounting (declared before GPU measurements)

1. Native FDTD, frozen 192-dimensional basis, fresh exact basis each update.
2. Cached sparse LU + first residual correction of primal/adjoint snapshots,
   rebuilding the fixed-size POD space each update. If the correction's
   aggregate PDE residual exceeds 0.01, reanchor the LU cache at current v.
3. The same method with two residual corrections, testing cost versus accuracy.

Online monitoring additionally evaluates final-Galerkin primal/adjoint residuals
at eight held-out retained bins. Growth beyond twice the initial-basis indicator
triggers reanchoring; the fixed reference does not drift after reanchoring.
The pilot and final ROM adjoint sources are compared with exact finite-irfft
weights (DC/Nyquist multiplicity and MSE scale included); above 1% discrepancy,
one additional consistency pass is allowed, then full refresh/native fallback.
Initialization enforces the same source-consistency check. These policies do
not replace independent gradient validation or provide a certified bound.

Default bounded pilot: Marmousi 94x288, central single shot, nt1000, dt0.0019,
8Hz, sigma15, seed3, residual original SIREN, Adam1e-4, five updates per arm.
CPU smoke uses a small crop first. Original acquisition, wavelet, MSE and
gradient accumulation remain unchanged within each paired run. Initial basis
preparation may be excluded from online runtime as requested, but is separately
reported, alongside totals including it. Initial LU storage/host memory is
reported because retaining 64 factorizations is not free.

Every arm uses the same initialized network/optimizer and records actual
velocities/model states. Operator/basis adaptation is frozen while computing
each update's derivative. No future trajectory state or truth wavefield enters
correction. Runtime-selected refreshes use only current v, observations,
approximate waveforms and PDE residuals. Rank192 and protected primal rank96,
64 training frequencies, cutoff32Hz and FFT settings remain matched.

After timing an arm, replay its saved states against native FDTD and compare
waveforms, physical velocity gradients and original SIREN parameter gradients.
This ex-post audit never selects updates/refreshes. Its actual time is reported
separately and also added to a full validation wall-time total. Any online FDTD
check or fallback must stay in the algorithm's online total. Acceptance: all
used update states below 1% waveform and 5% physical/parameter gradient error;
also report the final state. Failed arms retain their times and errors and do
not qualify as same-accuracy acceleration. Include final MSE/RMSE/SSIM versus
the equal-update FDTD arm; five updates cannot establish final convergence.

## Residual snapshot transport

With the original fixed PML, only the diagonal pressure mass/source terms
change: A(v,z)=A0(z)+D(v,z), D=diag((a0+a1*delta)*(vpad^-2-v0pad^-2)).
Cache LU(A0), primal u0=A0^-1 b0 and adjoint l0=A0^-H f0 at selected frequencies.
At current v:

    e1 = A0^-1 [b(v)-b0-D u0];          u1=u0+e1
    d1 = A0^-H [f(v)-f0-D^H l0];       l1=l0+d1
    e(k+1) = -A0^-1 D e(k)
    d(k+1) = -A0^-H D^H d(k)

The finite-iteration residuals are -D e(k) and -D^H d(k); their normalized
aggregate norms drive the refresh policy. They are indicators, not certified
wavefield/gradient bounds, because no uniform PML inf-sup constant is known.
Neumann convergence requires ||A0^-1 D||<1 (or a suitable spectral condition);
it is not assumed for all velocity changes. Full refactorization resets D=0.

First construct the transported primal space and run its small projected model
to obtain the current observation residual. Then transport adjoint snapshots
with the correct finite-FFT real adjoint forcing, preserve the primal space,
and add the adjoint complement. This avoids using the stale waveform as if it
were the updated adjoint source. A rank-deficient or nonfinite basis is a
recorded failure/fallback, never silently padded.

Static K/PML projections can dominate a naive refresh. Cache sparse operators
on the target device; project PML terms using their nonzero rows, and allow
PressureROM to accept these checked preprojected matrices. Default construction
and all old tests remain unchanged. Compare fast projections with dense native
ones; never change the discrete operator to obtain speed.

## Verification and deliverables

CPU tests: exact mass/source differences; one/two correction orders versus fresh
complex solves; conjugate adjoint; residual identity; v==anchor and no NumPy
global RNG consumption; fixed-source/primal subspace preservation; fast sparse
projections and checkpoint buffers; refresh decision/failure bookkeeping.
Then CPU actual-Adam smoke, followed by the bounded GPU pilot. Save configs,
source snapshots/SHA256, per-step timings/events/memory/checkpoints, ex-post
audit tables and an honest speed/accuracy report in independent output dirs.

## Second exploratory pilot (declared after first-run audit, before second GPU run)

First five-arm run: fresh and first-order correction pass all used-state gates,
but take58.44s and57.64s versus16.14s native. Second-order correction takes32.74s
and fails at the third used state (5.96% velocity /6.63% parameter gradient).
Thus none qualifies as same-accuracy acceleration. All raw failures remain.

Test third-order residual correction, using the same rank, eight-bin probe,
2x initial probe threshold, source consistency policy and Adam1e-4. Four
independent frequency workers perform factorization/solves; BLAS is limited to
one thread inside those workers and four for the remaining POD/probe work.
This is an implementation concurrency change plus an explicit correction-order
ablation, not a new physics operator or a claimed theoretical innovation.
CPU tests must first compare serial/parallel fields and third-order residuals.
GPU arms: native FDTD, fresh rebuild and correction3, five updates, same seed3.
No threshold tuning to hide the failed second-order state. If a speed/accuracy
gain appears, a separate held-out network seed should be checked before any
general claim; no long training is authorized by this pilot.
