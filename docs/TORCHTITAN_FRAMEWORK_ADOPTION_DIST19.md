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
ModelSpec/InitSpec identities. The blocker is the current model/runtime protocol boundary.

## Stable 0.2.2 versus current upstream main

The latest PyPI stable release inspected on 2026-08-25 is TorchTitan 0.2.2, released on
2026-02-20. That release uses `TrainSpec`, `BaseModelArgs`, and `ModelProtocol`. Its Trainer
constructs the model on the meta device, applies a model-specific `parallelize_fn`, then
calls `init_weights()` after `to_empty()`.

Current upstream `main` has already moved to a different `BaseModel`/`Module`/nested
`Config` protocol with recursive `init_states()` and `parallelize()`. Its current
`ModelSpec` source also contains a TODO to deprecate that abstraction. TorchTitan documents
extension points as subject to change.

A 0.2.2-only adapter would therefore target an API generation already replaced upstream.
It would also still require 12-6-specific `init_weights` and `parallelize_fn` code, including
the same module-aware TP/FSDP composition currently owned by active D16/D17/D12 work. That
would be throwaway duplication rather than a reduction in custom complexity.

## Concrete current mismatch

Current `TwelveSixDecoder` is a normal `torch.nn.Module` tree. Construction eagerly creates
and initializes parameters and the RoPE buffer. It has no TorchTitan 0.2.2 `init_weights`
contract and no current-main nested `Config`, `init_states`, or recursive `parallelize`
protocol.

Converting 12-6 directly to current-main TorchTitan is therefore a model-structure refactor
rather than a thin registration adapter. Building against stable 0.2.2 instead would avoid
part of that refactor but would require a model-specific parallelization implementation
against the older TrainSpec API. Neither path should be mixed into S0 or the active D12
FSDP2/TP/checkpoint work.

The project `ModelSpec` remains the semantic source of truth and must not be renamed or
replaced around TorchTitan's changing framework `ModelSpec` surface.

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

`backend="auto"` stays PyTorch-native. `backend="torchtitan"` fails closed because this PR
does not claim to implement a TorchTitan training driver. The explicit
`torchtitan_training_driver_adapter_not_implemented` blocker must be removed only by a
future package that actually builds and executes the chosen TorchTitan API generation.

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
2. an exact TorchTitan API generation is selected and locked by D08 rather than coding to
   both the 0.2.2 TrainSpec API and current-main protocol simultaneously;
3. the 12-6 model has a dedicated compatible construction/init/parallelization adapter
   without changing ModelSpec/InitSpec or importing foreign weights;
4. exact scratch initialization and single-rank logits are verified against the canonical
   12-6 model before distributed execution;
5. D05/D18 semantic checkpoint identity wraps the framework checkpoint bytes/state;
6. D03/D04 data/tokenizer/packing identity remains externally bound;
7. real GPU evidence is run only after explicit compute authorization.

Until then, TorchTitan is a future orchestration backend, not the canonical 12-6 semantic
layer.
