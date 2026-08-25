# PERF-30 scale performance matrix

## Ownership boundary

This package profiles scale above S0. It does not import, modify, wrap, or re-execute the W5
`src/twelve_six/s0_profile.py` incumbent. W5 remains the S0 profiling authority.

It also does not edit the active C01/D13 compute-plan files. The generated machine report is a
measured input for that owner to consume after integration rather than a competing compute plan.

## Observed geometries

The observed matrix loads the repository stage configs through the normal `load_stage_config()`
contract:

- S1: 107,856 parameters, vocabulary 512, context 256;
- S2: 1,066,112 parameters, vocabulary 2,048, context 512;
- S3: 10,059,840 parameters, vocabulary 8,192, context 1,024.

Every observed row is bound to the exact checkout SHA and the SHA-256 identities of its stage
config, ModelSpec, and InitSpec.

The default comparison workload is batch size 1, 128 input tokens, fp32 CPU, two Torch threads,
and seed 1337. The fixed 128-token sequence is deliberate: it gives S1/S2/S3 the same workload
instead of confusing model scaling with context scaling. Native maximum context is still recorded
in every row.

## Measurements

Each observed stage runs in a fresh Python subprocess so process RSS high-water observations from
a smaller stage do not contaminate a larger stage. The collector measures:

- model construction;
- no-grad forward and input tokens/second;
- the complete canonical `Trainer.train_microbatch()` transition and optimized tokens/second;
- checkpoint-v1 save size/time and verification time;
- stateless first-party greedy generation and generated tokens/second;
- fp32 parameter bytes;
- optimizer tensor-state bytes after real AdamW updates;
- process RSS high-water before/after the stage workload.

Current Trainer public semantics do not expose a clean timing seam between forward, backward, and
optimizer update. The scale report therefore labels the timed transition
`canonical_train_microbatch_forward_backward_update` and explicitly records
`NOT_EXPOSED_BY_PUBLIC_TRAINER_SEAM`. It does not fabricate subphase timing.

RSS is a process high-water observation, not an exact allocator peak for one operation. The
profile does not infer activation-memory capacity solely from RSS.

Generation on the exact #89 base is the canonical stateless full-prefix path. This package makes
no KV-cache latency or serving-throughput claim and does not collide with the active KV-cache
incumbent.

## Approximately 100M analytical row

Full 100M CPU execution is not required for this profile. The report contains a separate
`P100M_ANALYTICAL` row using an internal analytical-only ModelSpec of 100,017,216 parameters. It is
not a stage config and is explicitly marked
`ANALYTICAL_GEOMETRY_NOT_CANONICAL_STAGE_CONFIG`.

Runtime estimates use a log-log power-law fit over the three actually observed S1/S2/S3 points.
The report retains the fitted exponent and maximum relative fit error so downstream planning can
judge whether an extrapolation is credible. Parameter bytes are exact for the analytical fp32
geometry. Optimizer/checkpoint bytes use observed ratios. Process RSS is deliberately not
extrapolated because runtime baseline and allocator history make that inference unsound.

Observed and extrapolated values live in separate top-level report arrays. A validator rejects a
rehashed report that blurs that boundary.

## CPU decision boundary

The machine report identifies the earliest observed stage that meets either engineering threshold:

- median canonical train step at least 1.0 second; or
- optimized training throughput below 2,000 tokens/second.

If no observed S1/S2/S3 row crosses the threshold, the same decision is evaluated against the
100M analytical estimate and is labeled extrapolated rather than observed.

These are iteration-speed engineering thresholds for deciding when CPU-only training stops being
a sensible development path. They are not SLA, capacity, quality, or promotion criteria.

A CPU boundary can make accelerator/GPU engineering mandatory for serious training. It does not,
by itself, prove distributed training mandatory. Distributed necessity depends on accelerator
memory including activations/precision and on measured single-device throughput. The report keeps
that claim fail-closed.

## Reproducibility and evidence

The focused workflow:

1. checks out the exact PR head;
2. verifies the existing D08 Linux x86-64 lock without running repository-wide tests;
3. installs the exact hash-locked toolchain/runtime/dev environment;
4. runs Ruff and pytest only for the new profile package;
5. executes S1, S2, and S3 in isolated subprocesses;
6. validates the self-hashed matrix report;
7. retains the matrix JSON and locked-environment evidence.

Pure timing work does not add a second broad test suite. Existing repository CI may still run by
its own policy, but this profiler's workflow is intentionally focused.

Timing observations never enter model, optimizer, checkpoint, or deterministic training-state
fingerprints.

Authority remains `LOCAL_FREE_CPU_SCALE_PROFILE_NOT_CAPACITY_OR_PROMOTION`. No paid-compute,
GPU/CUDA, distributed-runtime, MFU, production-capacity, CANDIDATE/STABLE, audit, or quality claim
is created by this package.
