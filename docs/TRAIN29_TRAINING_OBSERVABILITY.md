# TRAIN-29 — next-scale training observability

## Scope and incumbent boundary

TRAIN-29 does not replace or copy the active W5 S0 CPU profiler in PR #142. W5 remains the
phase/resource profile for the exact 10,140-parameter S0 path. This package adds a reusable
training-observability layer and exercises it on the current 107,856-parameter S1 engineering
ModelSpec from the D02 S1 preflight incumbent.

The S1 probe is `LOCAL_FREE` CPU engineering evidence only. It reuses the controlled S0
fixture because the current S1 corpus/tokenizer are not frozen. It does not claim S1 quality,
S1 data readiness, target-GPU capacity, distributed scaling, or paid-compute authorization.

## What is measured

`TrainingObserver` records structured, bounded observations beside an exact run identity:

- optimized tokens and train tokens/second;
- blocking iterator/data-wait time;
- whole `Trainer.train_microbatch()` wall time, p50 and p95;
- loss, learning rate, and optimizer-step gradient norm;
- process RSS high-water mark where the platform exposes it;
- CUDA allocated/reserved and allocator peak bytes when CUDA is active;
- optional `torch.cuda.utilization()` samples when the runtime/provider exposes that hook;
- checkpoint save/verify duration through the real D05 checkpoint path;
- evaluation duration;
- rank, local rank, world size, process-group initialization state, and backend;
- optional forward/backward/update timings supplied by future backends that own trustworthy
  phase seams.

The current D02 Trainer does **not** expose public forward/backward/update timing seams. TRAIN-29
does not copy `train_microbatch()` or monkey-patch private internals merely to manufacture those
numbers. Its current S1 evidence therefore emits each phase as
`UNAVAILABLE_NOT_RECORDED`. A future backend or Trainer API can pass explicit `PhaseTimings`
without changing the observer schema.

## Low-overhead design

There is no logging-framework dependency and no file write in the hot step path. Aggregate
counters cover every observed step while retained step samples are bounded. When the sample cap
is exceeded, the collector deterministically increases its sampling stride and keeps aggregate
totals exact.

CUDA utilization is sampled on a configurable cadence. End-of-run distributed aggregation uses
one compact `all_gather_object` over summaries; there are no observability collectives on every
training step.

For GPU runs, per-step CUDA synchronization is opt-in. With synchronization disabled, the
summary marks step timing as `CUDA_HOST_ENQUEUE_WALL`; with it enabled, it marks
`CUDA_SYNCHRONIZED_WALL`. Checkpoint/evaluation regions synchronize by default because they are
low-frequency decision measurements.

## Determinism boundary

The run identity is copied and SHA-256 hashed before observations begin. Identity validation
rejects timing/resource keys such as `*_seconds`, duration, throughput, GPU-utilization, memory
peak, and timestamps. Telemetry is exported next to `run_identity_sha256`; it is never inserted
into `Trainer.state_dict()`, checkpoint deterministic state, or a deterministic training
fingerprint.

The unit contract proves that two runs with identical identity and radically different timing
measurements produce the same `run_identity_sha256`.

## Bottleneck diagnosis

The summary reports raw fractions as well as a conservative heuristic classification:

- `CHECKPOINT_BOUND` when checkpoint save/verify consumes at least 15% of observed
  train/eval/checkpoint wall time;
- `DATA_BOUND` when blocking data wait consumes at least 20% of measured training wall;
- `COMPUTE_BOUND_GPU_SATURATED` when data wait is low and sampled CUDA utilization averages at
  least 70%;
- `COMPUTE_BOUND_OR_RUNTIME_BOUND` when train-step work consumes at least 80% of measured
  training wall but GPU saturation is not established;
- `MIXED_OR_INCONCLUSIVE` otherwise.

These labels are diagnostics, not capacity claims. The raw shares, p50/p95 values, memory peaks,
and utilization samples remain authoritative if thresholds need to change.

## Distributed aggregation semantics

Every rank keeps its own summary. `aggregate_rank_summaries()` requires one summary for every
contiguous rank and one common run identity. Global optimized throughput is computed as:

`sum(rank optimized tokens) / max(rank observed training wall)`

This uses the slowest rank as the synchronous critical path. The aggregate also reports rank-time
skew, maximum rank data-wait time, and maximum rank step time. `gather_distributed_summary()`
performs the gather only after a process group already exists; observability never initializes
one itself.

## €2k / €10k decision use

The output deliberately separates **measurement** from **authorization**.

A CPU-only S1 probe is useful to verify instrumentation, identify obvious data/checkpoint
pathologies, and establish model/checkpoint memory mechanics. It is not enough to price a GPU
training run. Therefore the current probe must emit:

- €2k gate: `BLOCKED_PENDING_TARGET_GPU_CALIBRATION`;
- €10k gate: `BLOCKED_PENDING_TARGET_GPU_AND_DISTRIBUTED_CALIBRATION`.

Once the same observer is run on the intended GPU type, the measured global tokens/second can be
combined with an exact target training-token budget and an actual provider price:

`projected_cost_eur = target_training_tokens / measured_global_tokens_per_second / 3600 * measured_or_quoted_eur_per_gpu_hour * gpu_count`

For a €10k decision, a single-GPU measurement is still insufficient: the project also needs
multi-rank throughput, rank skew, peak memory/headroom, checkpoint overhead, and stability at the
planned topology. The observer exposes those seams but never authorizes spend by itself.

## Exact S1 probe

The workflow `.github/workflows/train29-s1-observability.yml` checks out the exact PR head,
verifies the D08 locked environment, runs repository checks, creates the hash-locked Python
3.11.16 environment, runs the observer unit contract, then executes 12 optimizer steps on the
current S1 engineering ModelSpec (`107,856` parameters) with the real D02 Trainer.

It measures four held-out/full-split evaluation regions and two checkpoint regions
(`save_trainer_checkpoint`, `verify_checkpoint`) and uploads:

- `locked-environment-linux-x86_64.json`;
- `s1-training-observability.json`;
- `s1-training-observability.jsonl`.

The JSONL artifact has explicit `run_identity`, `step`, `region`, and `summary` record types. It is
written after execution rather than synchronously inside the training hot path.

## Current truth boundary

Until a target-GPU run is actually executed, no GPU utilization, GPU tokens/second, distributed
scaling, provider-cost, or paid-run claim is PASS. Until S1 data/tokenizer and stage contracts are
frozen, this remains mechanics/observability evidence rather than S1 stage evidence.
