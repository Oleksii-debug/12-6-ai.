# D12 tensor-parallel seam for future 12-6 stages

Status: engineering execution seam. This does not change S0, freeze a later stage, authorize paid
compute, or claim GPU/NCCL/multi-node evidence.

## Incumbent lineage

This package is stacked on D12 PR #76 exact head
`8cf9fea1c5262b6003a5b2751ab53eec06aa20c1`, whose CI run `32747977475` is terminal SUCCESS.
PR #76 carries the current #74 runtime/DeviceMesh lineage and its finite topology verification.
PR #71 remains the earlier green distributed authority candidate, but #71 and #74 both own broad
`distributed/runtime.py` surfaces and must not be merged wholesale. This TP package is additive and
does not modify either incumbent runtime implementation.

The existing `TorchMeshSpec.tensor_parallel_mesh()` seam already returns the one-dimensional `tp`
submesh required by current PyTorch `torch.distributed.tensor.parallel.parallelize_module`.

## First native TP policy

The v1 seam keeps the residual stream replicated between sublayers and shards only the dense block
matmuls. That gives one collective at the end of attention and one at the end of the SwiGLU MLP,
without introducing a custom distributed framework.

| Canonical parameter | Global shape | TP style | Local shape at degree T |
| --- | --- | --- | --- |
| `attn.q_proj.weight` | `[Q, D]` | column-wise, dim 0 | `[Q/T, D]` |
| `attn.k_proj.weight` | `[KV, D]` | column-wise, dim 0 | `[KV/T, D]` |
| `attn.v_proj.weight` | `[KV, D]` | column-wise, dim 0 | `[KV/T, D]` |
| `attn.out_proj.weight` | `[D, Q]` | row-wise, dim 1 | `[D, Q/T]` |
| `mlp.gate_proj.weight` | `[F, D]` | column-wise, dim 0 | `[F/T, D]` |
| `mlp.up_proj.weight` | `[F, D]` | column-wise, dim 0 | `[F/T, D]` |
| `mlp.down_proj.weight` | `[D, F]` | row-wise, dim 1 | `[D, F/T]` |

`D=d_model`, `Q=n_heads*head_dim`, `KV=n_kv_heads*head_dim`, and `F=d_ff`.
Token embedding, final norm, and LM head remain replicated in v1. Vocabulary/loss parallelism and
sequence parallelism are deliberately separate follow-ons; they are not required to prove the first
credible block-level TP seam.

## GQA invariant

The current attention implementation reshapes projection outputs into complete heads before SDPA and
uses `repeat_interleave` to expand grouped K/V heads. The safest first TP contract therefore refuses
any degree that would split a query head, KV head, or SwiGLU channel partition:

- `T` must divide `n_heads`;
- `T` must divide `n_kv_heads`;
- `T` must divide `d_ff`.

This preserves the global GQA ratio independently on every TP rank. No KV head is replicated or split
inside a head in v1.

Against the non-frozen D01 PR #37 planning candidates:

- S5 ~400M: GQA `16/4`, `D=1024`, `F=5120`. Head-aligned TP2 and TP4 are valid. TP8 is rejected
  because four KV heads cannot be split into eight whole-head shards.
- S6 ~1B: GQA `32/8`, `D=2048`, `F=6720`. TP2, TP4, and TP8 are valid. TP16 is rejected for the same
  KV-head reason.

Those S5/S6 ModelSpecs are engineering candidates, not frozen architecture authority. The TP planner
binds the exact `ModelSpec.identity_sha256()` it receives and must be regenerated if D01 changes the
candidate geometry.

## PyTorch execution seam

`TensorParallelPlan` is torch-free and deterministic. `parallelize_decoder_tp()` lazily imports
PyTorch and applies the native plan to each canonical transformer block:

- Q/K/V and gate/up use `ColwiseParallel`;
- attention output and MLP down projections use `RowwiseParallel`;
- the TP mesh must be one-dimensional;
- the canonical global ModelSpec remains attached to the decoder;
- only derived runtime attention values (`n_heads`, `n_kv_heads`, `q_dim`, `kv_dim`) are localized
  after projection sharding so the existing reshape and GQA code sees local whole-head geometry.

The model boundary stays replicated, so no S0 trainer, tokenizer, loss, inference, or checkpoint code
is automatically routed through TP.

PyTorch 2.13 documents TP as DTensor-based and still experimental. Its current API explicitly
recommends composing `ColwiseParallel` and `RowwiseParallel` for Transformer attention/MLP modules and
requires slicing an N-D DeviceMesh to a one-dimensional TP submesh before `parallelize_module`.

Primary references:

- https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html
- https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html

## Checkpoint identity

TP changes physical storage, not the semantic Base model. The contract therefore separates two
identities:

1. canonical D05 parent identity: unchanged ModelSpec, canonical FQNs, and canonical global shapes;
2. TP physical layout identity: `TensorParallelPlan.identity_sha256` plus
   `checkpoint_layout_sha256`, which changes when TP degree or sharding policy changes.

The real CPU probe requires the sharded model state dict to retain exactly the same FQN set and global
tensor shapes as the unsharded model. Physical sharded persistence should use
`torch.distributed.checkpoint` and the existing D12 distributed checkpoint envelope. Directly
serializing rank-local shards as if they were an ordinary D05 single-device checkpoint is outside the
contract and should fail integration review.

A topology-changing load may preserve canonical model semantics while changing physical shard
identity. Existing D12 reshard/RNG rules continue to govern whether an exact trajectory claim is
allowed.

## Executable evidence

`tests/test_tensor_parallel.py` provides two distinct proof levels:

- deterministic fake decomposition: manually slices canonical attention and SwiGLU tensors by TP rank,
  computes local head/FFN work, sums row-wise partials, and compares with the ordinary full forward;
- real LOCAL_FREE multiprocess proof: `run_cpu_tensor_parallel_probe(2)` starts two Gloo ranks, creates
  a CPU DeviceMesh, runs actual PyTorch DTensor `parallelize_module`, checks rank-local parameter
  slices against the canonical tensor, verifies canonical state-dict FQNs/global shapes, and compares
  full decoder logits against the unsharded reference.

The probe is intentionally bounded to four local CPU ranks. CPU/Gloo evidence validates distributed
mechanics only; it is not GPU throughput or production-scale evidence.

## GPU/NCCL extension path

For the first authorized accelerator pilot, keep the same semantic plan and replace only the physical
backend:

1. launch one process per GPU with `torchrun` and initialize NCCL;
2. build the existing D12 N-D DeviceMesh, then select `mesh["tp"]`;
3. construct the canonical candidate model from its exact ModelSpec/InitSpec;
4. call `parallelize_decoder_tp(model, tp_mesh)` before optimizer creation;
5. for pure TP, train with replicated data on each TP group and ensure DP sampling is defined only
   across independent data-parallel groups;
6. when DP sharding is also needed, apply block-level FSDP2 after TP using the D12 DP mesh dimensions;
7. checkpoint with DCP, retaining canonical D05 semantic parent identity plus the topology-specific
   distributed envelope;
8. record per-rank memory, collective timing, forward/backward parity tolerances, optimizer update,
   and shutdown/error behavior before making any performance claim.

Practical first topology targets from the current non-frozen geometry are S5 TP4 and S6 TP8 on one
node, subject to actual device memory and the separately authorized compute plan.

## Megatron Core boundary

Native PyTorch remains the default for this first dense block-level TP seam. Megatron Core becomes a
serious evaluation target when the model needs a combined NVIDIA-oriented stack such as TP+PP,
context/sequence parallelism, MoE/expert parallelism, Transformer Engine/FP8 kernels, or mature
large-scale overlap/scheduling that would otherwise require custom orchestration.

`build_megatron_core_tp_adapter()` records the dependency-free semantic translation boundary:

- `tensor_model_parallel_size <- T`;
- `hidden_size <- d_model`;
- `num_attention_heads <- n_heads`;
- `num_query_groups <- n_kv_heads`;
- `kv_channels <- head_dim`;
- `ffn_hidden_size <- d_ff`;
- canonical separate Q/K/V tensors must be fused at the backend adapter boundary if the selected
  Megatron module uses fused QKV storage;
- canonical gate/up tensors must likewise be mapped explicitly to the selected fused SwiGLU FC1
  representation;
- canonical checkpoint identity remains above the backend-specific physical format.

This adapter is a plan, not Megatron runtime evidence. No Megatron dependency, model clone, or
backend-specific checkpoint writer is introduced here.

## Truth boundary

This package proves or prepares only TP mechanics. It does not claim CUDA/NCCL execution,
multi-node behavior, TP training convergence, optimizer-state sharding, FSDP2+TP composition runtime,
sequence/vocabulary parallelism, GPU memory savings, MFU/throughput, Megatron runtime parity, paid
compute authorization, candidate promotion, or audit authority.
