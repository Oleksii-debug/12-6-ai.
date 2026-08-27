# SCALE-TP-01 — Tensor Parallel + FSDP2 + DCP scale spine

Status: engineering integration candidate. LOCAL_FREE only. No accelerator, paid-compute, corpus-freeze, model-quality, stage-promotion, or trained-1B claim.

## Why this package exists

The Product lineage already had a merged FSDP2 + Distributed Checkpoint path, while D12 had an independently terminal-green native Tensor Parallel seam. The remaining explicit gap was their composition. A billion-parameter dense model should not require a second model implementation or a premature framework migration merely to combine orthogonal model and data sharding.

This package composes the incumbents instead:

1. canonical `TwelveSixDecoder` and global `ModelSpec` remain semantic authority;
2. native Tensor Parallel is applied first on the TP submesh;
3. FSDP2 is then applied on the orthogonal DP shard submesh;
4. optimizer construction happens only after both sharding transforms;
5. training token normalization is reduced only across the DP process group, because TP ranks are model shards rather than independent data replicas;
6. DCP persists the full `ParallelPlan`, logical ranks, semantic model identity, and physical sharded state.

This ordering follows the maintained PyTorch 2D TP + FSDP pattern: TP inside a fast model-parallel group and FSDP across the orthogonal data-parallel group.

## Runtime boundary

`src/twelve_six/distributed/hybrid_tp_fsdp2.py` adds an additive dense-v1 composition path. It deliberately rejects PP, CP, EP, HSDP replication, single-DP, and single-TP plans. Those axes must be added only with separate execution evidence; they are not silently accepted.

The incumbent `parallelize_decoder_tp()` still owns TP semantics. The incumbent `apply_fsdp2()` still owns FSDP2 wrapping. The incumbent D18 DCP adapter still owns checkpoint persistence. This package only binds them in the required order and gives the Trainer the exact DP communicator used by FSDP2.

## 1B planning identity

The non-frozen D01 S6 planning candidate remains:

- parameters: `999,106,560`;
- vocab: `32,768`;
- context: `4,096`;
- hidden size: `2,048`;
- layers: `18`;
- query heads / KV heads: `32 / 8`;
- head dim: `64`;
- SwiGLU FFN: `6,720`;
- ModelSpec SHA-256: `cc64cbe94a461c364f063652098e55bdcf640d4be756ee1e743a23dda3de7261`.

Its GQA geometry supports TP2, TP4 and TP8. A DP2 × TP8 engineering topology has world size 16 and a first-order model-state shard factor of 16 before any PP/CP/MoE decision. This is topology/readiness evidence only; it is not a fit, throughput, convergence, or cost claim.

## LOCAL_FREE execution proof

`tests/test_hybrid_tp_fsdp2_dcp.py` uses the canonical S1 107,856-parameter model as a size-cheap analogue on four CPU/Gloo ranks with DP2 × TP2. The test is required to prove all of the following on one exact source head:

- TP is active and bound to the unchanged global ModelSpec;
- FSDP2 is active on the orthogonal DP mesh;
- optimizer construction occurs after sharding;
- one real forward/backward/AdamW update completes;
- gradient normalization sees two DP replicas, not four DP×TP ranks;
- DCP saves a topology that records DP2 and TP2;
- a fresh model/optimizer/Trainer stack exact-topology loads the checkpoint;
- local checkpoint shards match exactly after reload;
- the resumed next update and loss match the uninterrupted control exactly.

A queued or failed GitHub Actions run is not PASS. CPU/Gloo success, if obtained, establishes only composition mechanics. It does not establish CUDA/NCCL performance or 1B hardware feasibility.

## Scale decision boundary

This package removes one software-composition blocker on the path to ~1B. It does not remove the larger data gate. RESEARCH-251 remains authoritative that meaningful learning above ~10M is data-limited on the current corpus. A 100M or 1B parameter count without enough unique, rights-cleared, decontaminated, no-replay training data is mechanics, not a meaningful trained-model milestone.

Before any serious 100M/400M/1B paid run require, at minimum:

- terminal frozen training-corpus identity and unique-loss ledger;
- selected tokenizer and packing identities;
- held-out evaluation freeze and decontamination proof;
- exact GPU/NCCL smoke on the intended topology;
- measured VRAM, throughput, checkpoint pause and recovery;
- exact run manifest and explicit compute authorization.
