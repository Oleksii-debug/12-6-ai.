# D18 distributed checkpoint successor

Status: **EXPERIMENTAL / LOCAL_FREE**. This is the scale checkpoint path for future 100M/400M/1B training. It does not replace D05 `checkpoint-v1`, authorize paid compute, or claim GPU/NCCL/multi-node/object-store durability.

## Incumbent audit and ownership

D05 `checkpoint-v1` remains the correct S0 format. PR #85 explicitly documents that its verified load snapshot retains serialized payload bytes in memory and that this is not the final distributed checkpoint design. Current D05 follow-ons own atomic-path, fsync/crash-durability, schema, portability, retained-artifact, and export work. D18 therefore does not edit checkpoint-v1 core or add another S0 filesystem wrapper.

D12 PR #71 is the first terminal-green distributed authority candidate (`b76fbc616cf7a6df3d0499168fc3678cbe78ce7f`, CI `32747001147` SUCCESS). It provides `ParallelPlan`, logical rank identity, D05-vs-layout identity separation, shard checksum contracts, and direct-vs-reshard planning. PR #74/#76 add useful exact-topology/RNG semantics, but #74 explicitly says #71 should remain the first green authority and that their overlapping runtime surfaces must not be merged wholesale. D18 is stacked directly on #71 and adds only the real maintained DCP data plane, one focused multiprocess test, and this document.

## Why PyTorch Distributed Checkpoint

The repository lock is PyTorch 2.13. Current PyTorch documentation states that `torch.distributed.checkpoint` saves/loads from multiple ranks in parallel and performs load-time resharding between cluster topologies. `torch.distributed.checkpoint.state_dict.get_state_dict()` / `set_state_dict()` are the maintained model+optimizer state seam and handle PyTorch FSDP/`fully_shard`, DDP/replicate, tensor parallel, and combinations while preserving canonical FQN mappings.

Primary references:
- https://docs.pytorch.org/docs/main/distributed.checkpoint.html
- https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html

D18 keeps all Torch imports lazy. Merely importing the D12 contract package still does not initialize `torch.distributed`.

## Format and identities

Schema: `12-6.distributed-dcp-checkpoint.v1`.

The DCP data plane owns large model and optimizer tensors. D18 adds a small control plane:

- `ScaleCheckpointIdentity`: exact Git SHA, ModelSpec, InitSpec, tokenizer config, tokenizer vocabulary, data manifest, packing, training config, environment lock, seed, optimizer step, and token count;
- canonical logical state-schema SHA-256 over state keys plus global tensor shape/dtype, independent of physical shard ownership;
- topology SHA-256 over the complete D12 `ParallelPlan`, world size, and ordered logical rank identities;
- streaming SHA-256 and exact byte size for every physical DCP artifact and optional rank-state sidecar;
- order-stable artifact-set SHA-256;
- metadata SHA-256;
- aggregate checkpoint SHA-256 binding semantic identity, state schema, topology, physical artifact set, metadata, schema, and DCP backend.

The aggregate identity is intentionally topology-specific. A legitimate topology-changing reshard preserves the semantic/training identity but does not pretend that the source physical layout identity remains valid after the next save.

## Save publication and atomicity truth boundary

The implemented LOCAL_FREE backend is local filesystem only:

1. rank 0 creates a unique same-parent staging generation;
2. all ranks write optional caller-owned rank state;
3. all ranks call `dcp.save()` for model+optimizer state;
4. only after DCP returns and a collective barrier completes does rank 0 stream-hash the complete payload inventory;
5. rank 0 writes the manifest, manifest hash, and `COMMITTED` marker;
6. rank 0 publishes the generation with same-parent `os.rename()`;
7. all ranks observe publication through the control-plane broadcast and final barrier.

This gives an atomic **namespace publication** contract only under a cooperative single-writer, same-filesystem local-directory precondition. It deliberately does not claim:

- file or directory fsync / power-loss durability;
- atomic no-replace against a hostile concurrent writer;
- NFS/SMB/FUSE/container-host durability;
- object-store rename semantics.

Those are separate storage guarantees. D05 PR #124 already owns local-POSIX fsync semantics for checkpoint-v1; D18 does not clone that work.

A hard rank death may block until the process group's configured timeout. If DCP or a participating rank fails before publication, no final checkpoint generation is published; an uncommitted staging generation may remain and is garbage-collectable. If rank 0 dies after the rename but before the final broadcast, a fully committed checkpoint may exist even though the training job reports failure; restart logic must verify `COMMITTED` + manifest + all artifact hashes before accepting it.

## Object-store adapter boundary

No fake S3/GCS implementation is introduced. PyTorch DCP already exposes `StorageWriter` / `StorageReader` extension points. The object-store successor should preserve this same logical manifest while replacing local rename publication with:

- a unique immutable generation prefix;
- DCP object writes through a maintained storage adapter;
- per-object size/checksum inventory where the store exposes trustworthy checksums, otherwise client-streamed SHA-256;
- a manifest object written after all payload objects exist;
- one small commit object written last, containing the manifest digest;
- readers accepting only generations whose commit object, manifest digest, exact inventory, and aggregate identity verify.

Object stores must never emulate filesystem atomicity by assuming rename. A concrete provider adapter needs provider-specific multipart/retry/consistency evidence before support is claimed.

## Exact-topology resume vs topology-changing reshard

`ResumeMode.EXACT_TOPOLOGY` requires the target topology hash to match the saved topology. DCP restores model+optimizer state. If every rank supplied a rank-state sidecar and the caller supplies `restore_rank_state`, the same logical rank receives that state and `exact_trajectory_claim_allowed=True`. The caller owns the semantics of that state (RNG, sampler cursor, data cursor, etc.) and must restore it before the next stochastic operation.

`ResumeMode.RESHARD` permits a different topology only when the canonical model/optimizer state schema is unchanged. DCP loads into the target preallocated sharding. Saved rank-local state is not mapped to a new rank cardinality. The returned policy is therefore `reseed-new-rank-streams-from-global-seed-step; no-bitwise-trajectory-claim`, and `exact_trajectory_claim_allowed=False`.

Topology-changing resume means semantic training continuation, not a claim that rank-local RNG streams, data assignment, floating-point reduction order, or the subsequent trajectory remain bitwise identical.

## LOCAL_FREE execution proof

The focused test `tests/test_distributed_dcp_checkpoint.py` executes the real maintained APIs on CPU/Gloo:

- 2-rank DCP save of a trained `nn.Linear` model plus AdamW state;
- exact 2-rank -> 2-rank load with exact model weights and optimizer step restoration;
- exact logical-rank sidecar restoration through the caller callback;
- topology-changing 2-rank -> 1-rank DCP load with exact model weights and optimizer steps for this controlled fixture;
- explicit rejection of a one-byte physical `.distcp` corruption through streaming SHA-256.

Local execution environment for this worker: Torch `2.10.0+cpu`; result: **1 passed in 14.57 s** for the repository-shaped focused test. A separate direct probe produced the same 2->2 and 2->1 result. This is engineering evidence only. The repository's locked PyTorch 2.13 GitHub Actions on the exact D18 head are the authority for integration.

No CUDA/NCCL, FSDP2-wrapped canonical D01 model, TP-sharded model, multi-node process group, rank kill injection, or remote object store was executed here.

## 100M / 400M / 1B checkpoint sizing

D01 engineering candidates are S4 `100,384,512`, S5 `400,598,016`, and S6 `999,106,560` parameters. The table is first-order payload arithmetic; metadata, allocator, filesystem block size, compression, padding, and backend framing are excluded.

| Candidate | BF16 model only, 2 B/param | FP32 params + AdamW m/v, 12 B/param | Conservative BF16 model + FP32 master + m/v, 14 B/param |
| --- | ---: | ---: | ---: |
| S4 ~100M | 0.201 GB | 1.205 GB | 1.405 GB |
| S5 ~400M | 0.801 GB | 4.807 GB | 5.608 GB |
| S6 ~1B | 1.998 GB | 11.989 GB | 13.987 GB |

The 12 B/parameter column is the useful baseline for a full AdamW resume when the live parameter representation is FP32. If the training stack retains a separate FP32 master copy in addition to BF16 model weights, use the 14 B/parameter column until the exact optimizer/precision implementation is measured.

Approximate write time for the 12 B/parameter payload at sustained aggregate storage bandwidth:

| Candidate | 0.5 GB/s | 1.0 GB/s | 2.0 GB/s |
| --- | ---: | ---: | ---: |
| S4 ~100M | 2.41 s | 1.20 s | 0.60 s |
| S5 ~400M | 9.61 s | 4.81 s | 2.40 s |
| S6 ~1B | 23.98 s | 11.99 s | 5.99 s |

Formula: `checkpoint_seconds ~= payload_bytes / sustained_aggregate_write_bytes_per_second + metadata/collective latency`.

These are not performance claims. Before a paid 400M/1B run, measure sustained DCP throughput on the exact storage path, topology, precision, optimizer, and shard count. Async DCP may later reduce exposed training stall, but the synchronous path should remain the correctness baseline first.

## Next integration boundary

D18 is ready to be consumed by the actual FSDP2 worker once the canonical model is `fully_shard`-wrapped. That integration should pass the existing model and optimizer directly through this DCP seam rather than convert back to an unsharded D05 in-memory snapshot. Single-device S0 continues using checkpoint-v1 unchanged.
