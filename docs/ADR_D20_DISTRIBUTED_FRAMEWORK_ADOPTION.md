# ADR D20 — stage-triggered distributed framework adoption

Date: 2026-08-25
Status: proposed, executable adapter included
Base: D12 PR #71 (`b76fbc616cf7a6df3d0499168fc3678cbe78ce7f`)

## Decision

Keep **native PyTorch/FSDP2** as the incumbent multi-GPU path. Keep **Megatron Core** as the
measured scale escape hatch. Do not add TorchTitan or OLMo-core as runtime dependencies now.

This is not a permanent framework bet. The switch is triggered by measured fit, throughput, or
parallelism requirements rather than parameter count alone.

The implementation is additive. It does not modify D12 `distributed/runtime.py`, topology, mesh,
or checkpoint-layout code. S0 remains unchanged.

## Why these two

| Dimension | Native PyTorch | TorchTitan | OLMo-core | Megatron Core |
| --- | --- | --- | --- | --- |
| 12-6 model | wraps `TwelveSixDecoder` | model/config port | TrainModule/model lifecycle adoption | config + state-dict adapter |
| GQA | explicit TP plan required | shared GQA sharding helpers | supported in model stack | `num_query_groups` |
| 12-6 RoPE | exact model code retained | exact only after model port | exact only after model port | `rotary_interleaved=True` preserves adjacent-pair basis |
| data | D03/D04 stays owner | trainer dataloader integration | composable OLMo data becomes a major lifecycle | project data can stay at boundary |
| checkpoint | D05 identity + DCP shards | strong DCP manager + seed checkpoint | integrated distributed checkpoint | `torch_dist` + resharding; bridge D05 identity |
| FSDP/TP | FSDP2 + DTensor TP | strong | strong | strong, plus distributed optimizer |
| PP/CP | available but PP alpha and CP prototype | integrated composition | integrated | mature scale path |
| MoE | topology not yet implemented | EP/TP active | integrated EP/MoE | strong EP/EDP; project topology v2 required |
| fault recovery | torchrun elastic + DCP resume | checkpoint manager + launcher | trainer/checkpointer | checkpoint restart + optional NVIDIA resiliency |
| logging | keep project metrics; PyTorch logging/profiler | rich built-ins | callbacks/metrics | mature training/profiling ecosystem |
| dependency cost | lowest | medium/high | medium/high | high and NVIDIA-oriented |
| maintenance cost for 12-6 | lowest now | duplicates D12 orchestration | collides with D02/D03/D05/D12 lifecycle | justified only after measured native limit |
| selected | **yes** | no, reference | no | **yes, conditional** |

All four are permissively licensed in the relevant upstream projects. License is not the deciding
factor; lifecycle ownership and dependency/runtime burden are.

## Important compatibility findings

### Native PyTorch

FSDP2 `fully_shard()` operates in-place on an existing `nn.Module`, so the project does not need to
replace `ModelSpec`, `InitSpec`, or `TwelveSixDecoder`. DTensor TP accepts module-FQN sharding plans.
DCP supports parallel save/load and load-time resharding. `torchrun` elastic supplies worker restart
semantics. This matches the existing D12 seam directly.

The unstable edge is multi-axis composition: current PyTorch TP is explicitly experimental, PP is
alpha, and DTensor CP is prototype. Therefore the project should add each axis only with numerical,
checkpoint, and failure evidence rather than treating API presence as production proof.

### TorchTitan

TorchTitan is technically strong and remains the best reference for composing current PyTorch
features. Its current custom-model path asks for a TorchTitan model/config registry, declarative
sharding, parallelize/pipeline hooks, Trainer.Config, and optionally a state-dict adapter. That is a
real port, not a thin wrapper around the existing 12-6 lifecycle.

TorchTitan also documents the initialization issue that matters here: topology-dependent meta/sharded
initialization may not reproduce a single-device initialization. Its recommended seed checkpoint is
created on one CPU and loaded on arbitrary GPU counts through DCP resharding. D20 adopts that design
principle without adopting the whole framework: project random initialization remains authoritative,
and topology comparisons should start from one project-bound seed checkpoint.

TorchTitan stable releases currently pin nightly `torch`/`torchao`, and the project is under active
interface evolution. Adding it now would create another framework layer over D12 for capabilities we
are already implementing with the same PyTorch primitives.

### OLMo-core

OLMo-core provides a complete, credible training stack. That completeness is the mismatch: its
`TrainModule`, model configuration, composable data loader, optimizer, distributed parallelism, and
checkpointer collectively overlap D02, D03/D04, D05, and D12. Adopting it would be a lifecycle
migration rather than a scale adapter. No current measured problem justifies that migration.

### Megatron Core

Megatron Core maps unusually well to current `ModelSpec` semantics:

- `num_query_groups` maps `n_kv_heads` for GQA;
- `kv_channels` maps `head_dim`;
- `gated_linear_unit` plus SiLU maps SwiGLU;
- RMSNorm and epsilon are configurable;
- `rotary_base` and `rotary_percent` map full or partial RoPE;
- `rotary_interleaved=True` rotates even/odd adjacent pairs, matching 12-6 v1 rather than Llama's
  half-split basis;
- default attention/MLP output initialization is `std / sqrt(2 * num_layers)`, matching `InitSpec`
  `sqrt_2_layers`.

This makes a dense state-dict/config adapter realistic. It does **not** make immediate adoption safe.
Current 12-6 package metadata requires Python `>=3.11,<3.12`; current Megatron-LM main package metadata
requires Python `>=3.12`. A Megatron runtime therefore needs a separately locked D08 Python/CUDA
profile before validation. The current Megatron install guide is looser than its package metadata;
for this project the package metadata is the gate.

The second blocker is topology. D12 models expert parallelism as a subgroup of project DP and excludes
EP from physical world-size multiplication. Megatron can model expert parallelism differently. D20's
adapter deliberately rejects `expert_parallel > 1`; future MoE adoption requires an explicit project
topology-v2 contract rather than silently reinterpreting ranks.

## Stage-triggered route

### Through first ~100M multi-GPU stages

Use the canonical model and Trainer boundaries. Start with native FSDP2 + DCP. Add ordinary data
parallelism first. Add TP only when parameter/optimizer or layer width requires it. Preserve D03/D04
batch order and D05 logical identity.

No framework migration is justified merely because the model reaches 100M parameters.

### ~400M to ~1B

Continue native FSDP2. Add DTensor TP where the GQA/MLP divisibility gate is satisfied. Benchmark the
Megatron adapter only when at least one of these becomes true:

1. native model state no longer fits at the intended batch/context;
2. a required runtime feature is incomplete or unstable in the native path;
3. PP or CP becomes operationally necessary;
4. an NVIDIA target cluster exists and a locked Megatron runtime has passed parity/resume tests.

### Multi-billion dense

Run the same checkpoint and deterministic batch stream through native and Megatron candidates. Do not
migrate on feature count. The default code gate requires a validated Megatron runtime and at least
`1.15x` measured step-throughput improvement, or a validated hard fit/runtime failure in native. The
1.15 threshold is a project maintenance-cost default, not a universal performance law, and is
configurable.

### MoE

Do not reuse D12 EP semantics. First define topology v2 with explicit physical/logical expert axes,
checkpoint identity, optimizer-state reshard rules, and rank mapping. Then benchmark Megatron Core as
the leading candidate. TorchTitan and OLMo-core remain comparison references.

## Executable adapters and experiments

`src/twelve_six/framework_adoption.py` contains:

- a machine-readable four-framework assessment;
- a native adapter that preserves ModelSpec/InitSpec hashes and delegates topology to D12;
- a dependency-free Megatron dense config prototype;
- exact GQA/RoPE/init mappings;
- a fail-closed MoE topology gate;
- a measured adoption gate that refuses migration before runtime validation.

`tests/test_framework_adoption_d20.py` exercises the real S3 stage identity plus a synthetic GQA,
partial-RoPE model. The local interface harness passed 7/7 tests before publication. Exact repository
CI on the PR head is the authoritative result; the local harness is not a GPU or Megatron-runtime
claim.

`tools/plan_framework_adoption.py` emits a JSON plan. Example:

```bash
python tools/plan_framework_adoption.py configs/stages/s3_10m.json \
  --dp 2 --tp 2 --shard-model-state
```

A future NVIDIA A/B experiment should hold constant:

- exact ModelSpec and InitSpec identity;
- one random-init seed checkpoint;
- exact D03/D04 sample/token order;
- optimizer and LR schedule;
- precision mode and global batch/tokens per update;
- checkpoint interruption step and resume target.

Measure at minimum: tokens/s, p50/p95 step time, peak allocated/reserved memory, loss and selected-logit
parity, checkpoint save/load time, exact logical checkpoint identity, restart success, and rank-failure
behavior. No paid GPU run is authorized by this ADR.

## Upstream primary references checked 2026-08-25

- PyTorch FSDP2: https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html
- PyTorch DTensor TP: https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html
- PyTorch CP: https://docs.pytorch.org/docs/stable/distributed.tensor.html
- PyTorch PP: https://docs.pytorch.org/docs/stable/distributed.pipelining.html
- PyTorch DCP: https://docs.pytorch.org/docs/main/distributed.checkpoint.html
- PyTorch elastic: https://docs.pytorch.org/docs/stable/elastic/quickstart.html
- TorchTitan model integration: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/README.md
- TorchTitan checkpointing: https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md
- OLMo-core architecture: https://github.com/allenai/OLMo-core/blob/main/AGENTS.md
- Megatron parallelism guide: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html
- Megatron TransformerConfig: https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html
- Megatron distributed checkpointing: https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/core/dist_checkpointing.html
- Megatron package metadata: https://github.com/NVIDIA/Megatron-LM/blob/main/pyproject.toml
