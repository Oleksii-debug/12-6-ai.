# D08 — DISTRIBUTED SCALE READINESS

Status: S0 planning/simulation surface. No paid compute was launched.

## S0 rule

S0 remains a single-device baseline. The distributed package does not import PyTorch and does not change the D01 model or D02 training loop. It provides contracts that later backends can translate without forcing FSDP/TP/PP/CP/EP into the tiny baseline.

## Current interfaces

- `HardwareProfile`: nodes, accelerators per node, optional device-memory capacity.
- `ModelScaleSpec`: minimal size/shape inputs used only for planning.
- `ParallelPlan`: DP/TP/PP/CP physical dimensions, EP subgroup size, and an explicit data-parallel model-state-sharding flag.
- `validate_topology`: fails closed on world-size mismatch and baseline divisibility errors.
- `estimate_training_memory`: transparent first-order dense per-rank parameter/gradient/optimizer/master-weight/activation estimate.
- `build_torchrun_command`: constructs an exact command tuple but never executes it.

## Expert-parallel topology correction

The first D08 draft treated EP as an independent multiplier of physical world size and of the whole-model state-shard factor. That is unsafe for the target Megatron-style MoE path. Megatron Core describes expert model parallelism as distributing experts across a sub-dimension of data parallelism. The corrected contract therefore:

- computes physical world size from DP × TP × PP × CP;
- requires `expert_parallel` to divide `data_parallel`;
- exposes expert data parallel size as `DP / EP`;
- never applies EP to all dense model parameters;
- fails closed in the generic memory estimator for `EP > 1` until a MoE-aware model spec separates dense/shared parameters from expert parameters.

This correction does not affect S0 because dense S0 remains EP=1.

## Memory-estimator limits

The default state model approximates bf16/fp16 parameters and gradients, fp32 master weights, and two fp32 Adam moments. Activation memory uses an explicit `B*S*H*L` multiplier. This is planning evidence only. It does not include allocator fragmentation, kernels/workspaces, communication buffers, KV cache, dataloader memory, checkpoint staging, or backend-specific implementation details. Capacity claims require measured profiler evidence from D02/D08 on the exact candidate SHA and hardware.

The current estimator is dense-only. It intentionally rejects EP>1 rather than pretending that a single total-parameter count is enough to estimate MoE sharding.

## Scale path

1. S0: single-device canonical training; use these validators only as tests/planning.
2. After D01/D02/D05 contracts stabilize: integrate `torchrun` + FSDP2 behind the existing plan surface.
3. Add HSDP/TP/PP/CP only when measured model size/context/throughput requires them.
4. For the long-term sparse MoE path, add EP through a backend that has real MoE support and a model spec that separates shared/dense and expert parameter counts; do not set EP>1 on dense S0.
5. Evaluate TorchTitan/OLMo-core first for transparent PyTorch scaling and Megatron Core where large TP/PP/CP/EP scale justifies it. DeepSpeed remains optional where it provides a measured advantage.

## Authorization

`build_torchrun_command` is a constructor, not a launcher. Materially paid GPU/cloud runs remain prohibited until an exact run is covered by `COMPUTE_AUTHORIZED` with the required Git/model/tokenizer/data/config/cost/output metadata.

## NOT TESTED

- real multi-process `torchrun` execution;
- FSDP2/HSDP wrapping;
- TP/PP/CP collectives;
- MoE/EP runtime and a MoE-aware memory estimator;
- distributed checkpointing;
- NCCL/Gloo behavior;
- throughput or MFU;
- GPU memory accuracy;
- TorchTitan/OLMo-core/Megatron Core integration;
- cloud topology or cost.

Those are intentionally deferred until the canonical model/trainer/checkpoint surfaces exist and appropriate hardware/authorization is available.
