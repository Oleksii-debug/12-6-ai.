# SCALE-144 FSDP2 sharding/reshard policy

Status: engineering evidence for the single-host, 2-rank CPU/Gloo class only.

This work keeps the existing PyTorch FSDP2 + DCP runtime. It does not add a sharding implementation. The canonical bottom-up 12-6 grouping remains authoritative: tied token embedding/lm-head group, each decoder block, then the decoder root.

## Maintained policy set

| Policy | Non-root `reshard_after_forward` | Root `reshard_after_forward` | Purpose |
| --- | ---: | ---: | --- |
| `full_shard` | `true` | `true` | Current 12-6 behavior and minimum retained unsharded parameter residency. |
| `root_keep_unsharded` | `true` | `false` | PyTorch-maintained root-retention variant while preserving layer FULL_SHARD behavior. |
| `shard_grad_op` | `false` | `false` | Retain unsharded parameters after forward to avoid backward parameter re-all-gathers. |

The root-retention variant is implemented only by calling the maintained FSDP2 `set_reshard_after_forward(..., recurse=False)` API after the canonical `apply_fsdp2` grouping. No grouping logic is duplicated.

## Evidence classes

Two evidence classes are deliberately separated.

1. `tools/run_fsdp2_shard_policy.py` is the repository-integrated proof. It executes the current 10,000,640-parameter S3 configuration, existing `FSDP2Trainer`, existing DCP save/load path and exact continuation checks under the exact locked runtime in `.github/workflows/scale-144-fsdp2-shard-policy.yml`.
2. A LOCAL_FREE exact-geometry calibration was executed in the available sandbox with PyTorch 2.10.0+cpu/Gloo. It used the exact S3 and S4 parameter geometries and the same maintained bottom-up FSDP2 group structure, tied embedding/head alias, AdamW and PyTorch DCP APIs. This calibration is useful for policy direction, but it is not a substitute for the repository-integrated locked-runtime workflow.

No compatible CUDA/NCCL hardware was available. There is no CUDA/NCCL or multi-node performance claim.

## LOCAL_FREE 10M calibration

Exact model size: 10,000,640 parameters. World size: 2. Device/backend: CPU/Gloo. Precision: fp32. Timed microbatch: one sequence of 8 tokens per rank.

All three policies produced the same first-step loss (`5.617318153381348`) on both ranks. After DCP save, object destruction, fresh model/optimizer construction and exact-topology reload, all three reproduced the control second-step loss exactly (`5.892124652862549`) and reproduced the final local parameter shard byte-for-byte. The tied embedding/lm-head alias was preserved before sharding, after sharding and after the reload rebuild.

Per-rank persistent state was policy-independent after the first AdamW step:

- parameter state: 20,001,280 bytes (19.075 MiB);
- gradient state before optimizer step: 20,001,280 bytes (19.075 MiB);
- optimizer state: 40,003,000 bytes (38.150 MiB);
- parameter + gradient + optimizer state: 80,005,560 bytes (76.299 MiB).

| Policy | Max-rank step time | Min-rank tokens/s | Max sampled RSS | Logical transfer proxy/rank |
| --- | ---: | ---: | ---: | ---: |
| `full_shard` | 0.6097 s | 13.12 | 476.92 MiB | 57.224 MiB |
| `root_keep_unsharded` | 0.5846 s | 13.69 | 477.25 MiB | 57.224 MiB |
| `shard_grad_op` | 0.4345 s | 18.41 | 485.04 MiB | 38.149 MiB |

The communication number is an algorithm-independent logical tensor-payload proxy, not measured network wire bytes. For world size 2 it accounts for scheduled forward all-gather, backward re-all-gather when applicable, and gradient reduce-scatter payload.

`shard_grad_op` was about 28.7% lower step time than `full_shard` in this CPU/Gloo calibration while increasing sampled peak RSS by about 1.7%. This is not projected to NCCL.

## LOCAL_FREE ~100M materialization boundary

Exact model size: 99,897,600 parameters. World size: 2. Device/backend: CPU/Gloo. This boundary is real materialization plus a no-grad two-token forward. It is **not** 100M training, backward, optimizer-state or tokens/sec evidence.

Each rank held 199,795,200 bytes (190.540 MiB) of sharded fp32 parameter state before forward.

| Policy | Max-rank forward time | Max sampled RSS | Max forward RSS delta | Training transfer proxy/rank, if backward were executed |
| --- | ---: | ---: | ---: | ---: |
| `full_shard` | 0.4118 s | 763.77 MiB | 212.73 MiB | 571.619 MiB |
| `root_keep_unsharded` | 0.4427 s | 763.49 MiB | 212.66 MiB | 571.617 MiB |
| `shard_grad_op` | 1.1684 s | 1056.00 MiB | 505.30 MiB | 381.079 MiB |

At this boundary, retaining all unsharded parameters after forward raised sampled peak RSS by about 38.3% versus `full_shard` and increased this CPU forward time by about 2.84x. That is sufficient to reject an unconditional `shard_grad_op` default for the tested 10M→100M CPU/Gloo class even though it won the 10M step-time calibration.

## Root-group finding

Because the existing decoder is already sharded bottom-up, the root FSDP group owns only the final RMSNorm parameter after the tied group and decoder blocks have claimed their parameters. The root-owned fp32 payload is only:

- 1,024 bytes at 10M (`d_model=256`);
- 3,072 bytes at ~100M (`d_model=768`).

Therefore `root_keep_unsharded` is intentionally a very small delta from `full_shard` for this decoder. It removes a needless root re-all-gather without creating meaningful memory pressure, but its timing difference should be treated as noise-scale rather than as a speed claim.

## Engineering default

For the tested **single-host, 2-rank CPU/Gloo** hardware/topology class across the current 10M training point and ~100M materialization boundary, use `root_keep_unsharded` as the engineering default:

- non-root decoder groups keep FULL_SHARD-style `reshard_after_forward=true`;
- the root uses `reshard_after_forward=false`;
- tied embedding/lm-head grouping remains explicit and shared;
- DCP exact-topology continuation must remain a gate.

Do not promote `shard_grad_op` globally from the 10M CPU result: its 100M forward/materialization memory penalty is already material. Do not promote this CPU/Gloo default to CUDA/NCCL or multi-node. A CUDA/NCCL default requires real compatible free hardware and the same correctness/checkpoint gates plus measured GPU peak memory and communication/timing evidence.

## Decision rule for this decoder

1. Keep `root_keep_unsharded` as the scale-safe 2-rank CPU/Gloo baseline because the root-owned payload is negligible.
2. Consider `shard_grad_op` only when the target model/context has measured peak-memory headroom after retaining unsharded non-root groups and a topology-specific timing win large enough to justify that residency.
3. Fall back to `full_shard` when memory headroom is the limiting resource or when `shard_grad_op` has not been measured at the target scale/hardware.
4. Never infer multi-node or CUDA/NCCL performance from CPU/Gloo.
