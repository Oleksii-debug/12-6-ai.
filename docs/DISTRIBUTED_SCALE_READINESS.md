# D08 — DISTRIBUTED SCALE READINESS

Status: S0 planning/simulation surface. No paid compute was launched.

## S0 rule

S0 remains a single-device baseline. The distributed package does not import PyTorch and does not change the D01 model or D02 training loop. It provides contracts that later backends can translate without forcing FSDP/TP/PP/CP/EP into the tiny baseline.

## Current interfaces

- `HardwareProfile`: nodes, accelerators per node, optional device-memory capacity.
- `ModelScaleSpec`: minimal size/shape inputs used only for planning.
- `ParallelPlan`: DP/TP/PP/CP/EP factors plus an explicit data-parallel model-state-sharding flag.
- `validate_topology`: fails closed on world-size mismatch and baseline divisibility errors.
- `estimate_training_memory`: transparent first-order per-rank parameter/gradient/optimizer/master-weight/activation estimate.
- `build_torchrun_command`: constructs an exact command tuple but never executes it.

## Memory-estimator limits

The default state model approximates bf16/fp16 parameters and gradients, fp32 master weights, and two fp32 Adam moments. Activation memory uses an explicit `B*S*H*L` multiplier. This is planning evidence only. It does not include allocator fragmentation, kernels/workspaces, communication buffers, KV cache, dataloader memory, checkpoint staging, or backend-specific implementation details. Capacity claims require measured profiler evidence from D02/D08 on the exact candidate SHA and hardware.

## Scale path

1. S0: single-device canonical training; use these validators only as tests/planning.
2. After D01/D02/D05 contracts stabilize: integrate `torchrun` + FSDP2 behind the existing plan surface.
3. Add HSDP/TP/PP/CP only when measured model size/context/throughput requires them.
4. For the long-term sparse MoE path, add EP through a backend that has real MoE support; do not set EP>1 on dense S0.
5. Evaluate TorchTitan/OLMo-core first for transparent PyTorch scaling and Megatron Core where large TP/PP/CP/EP scale justifies it. DeepSpeed remains optional where it provides a measured advantage.

## Authorization

`build_torchrun_command` is a constructor, not a launcher. Materially paid GPU/cloud runs remain prohibited until an exact run is covered by `COMPUTE_AUTHORIZED` with the required Git/model/tokenizer/data/config/cost/output metadata.

## NOT TESTED

- real multi-process `torchrun` execution;
- FSDP2/HSDP wrapping;
- TP/PP/CP collectives;
- distributed checkpointing;
- NCCL/Gloo behavior;
- throughput or MFU;
- GPU memory accuracy;
- TorchTitan/OLMo-core/Megatron Core integration;
- cloud topology or cost.

Those are intentionally deferred until the canonical model/trainer/checkpoint surfaces exist and appropriate hardware/authorization is available.
