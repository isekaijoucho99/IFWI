# Local preflight evidence, 2026-11-12 UTC

The fixed protocol is documented in [batched-encoding-protocol.md](batched-encoding-protocol.md).
These are implementation/throughput checks, not1000-update accuracy results.
The original `rnn_fd.py` remains byte-identical to `source_manifest.json`.

## Verified correctness and recovery

-153 relevant CPU regressions passed in46.11s, exit0. This includes the
 original experiment/paper-supplement tests and new batching, objective,
 illumination, progressive SIREN, checkpoint and orchestration tests.
- Six real GPU batch/serial/native cases passed fixed tolerances. Predictions
 matched exactly; maximum relative velocity-gradient difference was3.266e-7,
 and maximum preconditioned-direction difference4.065e-7. Raw source/adjoint
 energy and H differences were below3e-15. Actual cost:48F/48B,0Adam updates.
- Full13 and progressive update12 GPU checkpoints restored exact model/Adam,
 velocity and Python/NumPy/CPU/CUDA random replay. This audit used0PDE solves
 and0Adam updates. No identity/state substitution or unsafe pickle load.
- Earlier failed checks are retained locally: modified-author-source hashes
 triggered13 regression failures, resolved by restoring the original file and
 moving batch injection to a separate adapter; a workspace hard-link operation
 required formal approval. No manifest, access control or security setting was
 changed to suppress those failures.

## Ten measured complete updates after two warmups

NVIDIA GeForce RTX5060Ti16GB, Python3.10.18, Torch2.9.0.dev20250904+cu128,
NumPy2.2.6; intra/inter-op threads1, cuDNN deterministic, benchmark disabled,
matmul/convolution IEEE FP32. Same cached nominal−12dB Marmousi acquisition,
same initial velocity, same Adam and illumination rule. Arms ran sequentially
with no concurrent training or CPU numerical tests during timing.

|Setting|Median wall s/update|Median CUDA ms|Peak allocated GB|Trainable parameters|
|---|---:|---:|---:|---:|
|Full13|3.188|3187.864|10.123|50049|
|Global1|2.986|2986.312|1.721|50049|
|Global2|3.055|3054.838|2.422|50049|
|Interleaved2|2.983|2983.297|2.422|50049|
|Interleaved3|3.015|3014.693|3.122|50049|
|Interleaved2 + progressive|2.983|2982.527|2.540|100098|

GB denotes1e9 bytes of PyTorch allocated memory, not total device memory.
The progressive arm uses two SIRENs. Its timing window covers early alpha
values; both branches execute throughout. It does not isolate frequency from
network capacity. Initialization, serialization and other stage overhead are
outside per-update samples but included in total stage wall time307.970s.
Two warmups plus ten samples per arm cost276F/276B and72Adam updates.

Encoding reduced field counts and allocated memory, but complete-update wall
time fell only about4–6% relative to the efficient batched full-shot control.
These measurements do not establish a large end-to-end speedup or accuracy
advantage. No old serial benchmark is substituted for the full-shot control.

The authoritative local stage artifacts are `gpu_correctness_01` and
`gpu_timing_01` under the dated batch-encoding preparations directory, with
append-only ledger, raw samples, hashes and successful terminal receipts.
Both bind to physical/data/runtime protocol SHA256
`b80c45b1409a20142af0cc12c40d451436f4331dc4fb29856eacfb35554188ad`.
Preflight Git labels honestly record the working tree; a separate verified
commit/source binding accompanies formal training after push.
