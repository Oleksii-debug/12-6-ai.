# DIST-19 TorchTitan framework-adoption seam

Date: 2026-08-25

## Decision

Do not make TorchTitan the 12-6 training driver yet. Keep proven S0 unchanged and use the
existing D12 PyTorch-native seam for the next scale steps.

TorchTitan is directionally valuable for later 100M+ training because it already composes
maintained PyTorch FSDP2/HSDP, tensor parallel, pipeline parallel, context parallel,
activation checkpointing, compile, distributed checkpointing including async save,
checkpointable distributed data loading, and distributed metrics/profiling.

The blocker is not weight provenance. 12-6 can keep scratch initialization and its own
ModelSpec/InitSpec identities. The blocker is the current model protocol boundary.

## Concrete current mismatch

Current `TwelveSixDecoder` is a normal `torch.nn.Module` tree. Construction eagerly creates
and initializes parameters and the RoPE buffer. It has no TorchTitan nested `Config`,
`init_states`, or recursive `parallelize` protocol.

Current TorchTitan model integration expects its `BaseModel`/`Module` configuration and
state-initialization protocol. Converting 12-6 directly is therefore a model-structure
refactor rather than a thin registration adapter. That refactor should not be mixed into
S0 or into the active D12 FSDP2/TP/checkpoint work.

TorchTitan also documents its extension points as subject to change, and its current
`ModelSpec` implementation contains an upstream TODO to deprecate that abstraction. 12-6
must not rename or replace its stable semantic `ModelSpec` around that moving framework
surface.

## Implemented seam

`src/twelve_six/distributed/framework_adapter.py` adds one additive framework boundary.
It does not modify D12 `runtime.py`, checkpoint-v1, Trainer, data, tokenizer, or model code.

The seam maps:

- model registration: 12-6 `ModelSpec` and `InitSpec` remain authoritative; scratch random
  initialization is explicit;
- dataset boundary: existing `Mapping[str, Tensor]` input/target/loss-mask contract remains
  authoritative, with data ownership on effective DP and TP/PP/CP peers sharing that batch;
- optimizer: existing `TrainerConfig` AdamW values are the source of truth and can map to
  TorchTitan `OptimizersContainer` later without accepting different defaults;
- distributed plan: reuses D12 `build_torch_native_plan()` instead of adding topology code;
- checkpoint: D05/D18 retain semantic/topology identity, while large checkpoint storage is
  delegated to PyTorch Distributed Checkpoint or later TorchTitan CheckpointManager over DCP;
- logging: existing D02 `StepMetrics` maps to a stable structured event and can later feed
  TorchTitan MetricsProcessor.

`backend="auto"` stays PyTorch-native until the direct TorchTitan adoption gate passes.
`backend="torchtitan"` fails closed while the concrete model-protocol blockers remain.

## TorchTitan complexity worth adopting later

Once a scale experiment justifies the model-protocol migration, TorchTitan can remove a
large amount of orchestration that 12-6 should not reproduce itself:

- FSDP2/HSDP/TP/PP/CP composition and mesh handling;
- meta-device construction/materialization;
- activation checkpointing and compile ordering;
- DCP save/load and async checkpoint orchestration;
- checkpointable distributed dataloader state;
- distributed metrics, profiling, and memory monitoring.

12-6 should retain architecture/data/checkpoint semantic identities and scratch-trained
weights even after those runtime services are delegated.

## LOCAL_FREE probe

`tools/run_scale_framework_probe.py` runs one real CPU optimizer transition using an
existing stage config and the canonical `TwelveSixDecoder`:

1. construct from 12-6 ModelSpec + InitSpec with scratch random initialization;
2. produce synthetic in-vocabulary token IDs only for the mechanical probe;
3. run canonical model forward and D02 causal LM loss;
4. backward and use the existing D02 AdamW builder;
5. require model-state SHA-256 to change;
6. prove framework probing did not initialize `torch.distributed`;
7. record installed PyTorch-native FSDP2/DTensor/TP/DCP capabilities;
8. record TorchTitan availability and exact direct-adoption blockers.

Authority is `LOCAL_FREE_CPU_MECHANICS_ONLY_NOT_SCALE_PERFORMANCE`. It is not GPU,
multi-process, throughput, convergence, or 100M performance evidence.

The current committed stage registry ends at S3 ~10M. No synthetic 100M stage config is
added by this package. The same seam is intended for a future approved 100M+ ModelSpec.

## Adoption gate

Move from PyTorch-native to TorchTitan only when all of these are true:

1. a 100M+ stage needs at least two TorchTitan-owned services beyond what the D12 native
   runner already proves in real execution;
2. the 12-6 model has a dedicated TorchTitan-compatible construction/init/parallelization
   adapter without changing ModelSpec/InitSpec or importing foreign weights;
3. exact scratch initialization and single-rank logits are verified against the canonical
   12-6 model before distributed execution;
4. D05/D18 semantic checkpoint identity wraps the framework checkpoint bytes/state;
5. D03/D04 data/tokenizer/packing identity remains externally bound;
6. an exact hash-locked TorchTitan + PyTorch environment is owned by D08;
7. real GPU evidence is run only after explicit compute authorization.

Until then, TorchTitan is a future orchestration backend, not the canonical 12-6 semantic
layer.
