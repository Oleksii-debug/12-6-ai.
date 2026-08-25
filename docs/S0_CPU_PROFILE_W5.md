# W5 S0 CPU phase profiling

This package measures the current exact 10,140-parameter S0 implementation on one bounded LOCAL_FREE CPU host. It is performance evidence for engineering and scale planning, not a capacity benchmark, SLA, release decision, audit verdict, or stage-promotion claim.

## Scope

The profiler composes existing first-party contracts without changing them:

- D01 scratch model construction;
- D03/D04 exact controlled split, byte tokenizer, and packing;
- D02 canonical training step and exact real-S0 training evidence;
- D05 checkpoint save, verification, and fresh-object load;
- D07 first-party greedy generation.

The report is bound to the exact Git checkout SHA and fails closed if the declared source SHA does not equal `git rev-parse HEAD`. It also binds ModelSpec, InitSpec, dataset, train/validation files, tokenizer config/vocabulary, packing config, and the committed D08 environment-lock index.

## Measured phases

The machine report contains repeated wall-clock and process-CPU samples for:

1. seed application plus scratch model construction;
2. train-split read, tokenization, and packing;
3. no-grad forward evaluation on a representative packed batch;
4. one complete canonical `Trainer.train_microbatch()` transition, including forward, backward, gradient normalization/clipping, and optimizer update;
5. checkpoint save;
6. checkpoint integrity verification;
7. checkpoint load into a fresh model and Trainer;
8. first-party greedy generation.

The Trainer phase is intentionally reported as one end-to-end transition. The current public Trainer contract does not expose a clean timing seam between backward and optimizer-update internals. This package therefore does not invent separate backward/update numbers.

The profiler also invokes the existing exact S0 training-evidence runner for a bounded multi-step run and records optimized-token throughput plus the retained training-evidence hash. Validation remains held out with zero optimized validation tokens.

## Memory and storage observations

The report records deterministic byte counts for model parameters, model buffers, and optimizer tensor state. It records checkpoint directory bytes from the actual D05 checkpoint.

Python `tracemalloc` peak/current bytes and process RSS high-water values are separate observations. `tracemalloc` does not measure all native PyTorch allocations, while RSS high-water is process-global and monotonic on supported Unix hosts. Neither is represented as an exact per-operation allocator peak.

## Truth boundary

The profile does not claim or authorize:

- materially paid compute;
- GPU/CUDA performance;
- distributed execution performance;
- MFU;
- cluster, service, or production capacity;
- cross-machine reproducibility of timing;
- foreign pretrained Base weights;
- instruction/alignment/refusal/personality/domain-specialization behavior;
- AUDIT-A/B PASS;
- CANDIDATE, AUDITED_CANDIDATE, or STABLE promotion.

Timing and throughput are host observations and should be used as inputs to C01/D08 planning with the exact host/source identities attached. They must not be extrapolated into paid-compute budgets without an explicit model of hardware, utilization, parallel efficiency, data pipeline, precision, and stage-specific architecture.

## Commands

Run a full bounded profile from an exact checkout:

```bash
python -m twelve_six.s0_profile run \
  --source-sha "$(git rev-parse HEAD)" \
  --output s0-cpu-profile.json \
  --training-steps 40 \
  --repetitions 5
```

Validate retained evidence against an expected exact head:

```bash
python -m twelve_six.s0_profile validate \
  s0-cpu-profile.json \
  --source-sha "$(git rev-parse HEAD)"
```
