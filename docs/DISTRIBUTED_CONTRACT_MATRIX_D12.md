# D12 distributed contract verification matrix

Status: **EXPERIMENTAL / LOCAL_FREE**. This is a stacked verification package over PR #74 and does not create a second distributed runtime implementation.

Parent authority at branch creation:

- PR #74 branch: `d12/distributed-runtime-contracts-20260824`
- parent head: `3a6c2aab85f0adbac82341bf5d7b3c0f119aa8f2`
- parent CI run `32747138599`: arm64 SUCCESS; x86 FAILURE at Ruff only
- parent x86 Ruff blockers: two UP035 import-location findings and two TRY004 exception-type findings

The stacked package carries only those mechanical Ruff corrections in parent-owned Product files. No topology, checkpoint-resume, backend-selection, model, trainer, tokenizer, data, Base-behavior, or promotion semantics are changed.

## Exhaustive local-contract matrix

`tests/test_distributed_contract_matrix.py` intentionally expands the verification surface without adding a new backend abstraction.

The finite matrix enumerates:

- project DP: `1, 2, 4, 6, 8`;
- TP, PP, CP: `1, 2` each;
- EP: every positive divisor of project DP.

This yields **112 valid project topologies**. Across them the test visits **1,971 logical ranks** and checks:

1. rank -> coordinate -> rank bijection;
2. exact DP/TP/PP/CP group cardinality and partition coverage;
3. EP and expert-DP groups as two complementary partitions of each project DP group;
4. dense shared-weight synchronization groups of cardinality `DP * CP` with TP/PP fixed;
5. logical layout identity stability for every valid HSDP split;
6. every positive divisor of project DP as `dp_replicate * dp_shard`, totaling **368 mesh factorizations**;
7. current project-to-Megatron translation `TP * PP * CP * EP * expert_DP == project_world_size` for every case;
8. invalid HSDP shard factors fail closed.

## Checkpoint and resume adversarial matrix

The follow-on also verifies that the PR #74 distributed checkpoint envelope keeps identity layers distinct:

- physical shard order is canonicalized and does not change artifact identity;
- changing writer-rank ownership changes physical artifact and envelope identities;
- changing rank-local RNG digest changes aggregate envelope identity but not shard-byte identity;
- the D05 semantic parent identity remains unchanged across physical mutations;
- exact-topology resume requires the identical logical layout;
- topology-changing reshard can be allowed when state-dict schema is identical, but it cannot claim an exact rank-local RNG trajectory;
- state-dict schema drift blocks resume;
- duplicate shard paths and rank-RNG cardinality drift fail closed.

## Backend trigger totality

All 64 boolean combinations of the current backend-adoption trigger function are checked against its documented precedence:

1. MoE, or NVIDIA-only PP -> Megatron Core evaluation first;
2. CP or Float8 -> TorchTitan evaluation first;
3. DCP reshard need -> native PyTorch FSDP2/DCP;
4. PP without higher-priority triggers -> OLMo-core or native PyTorch benchmark;
5. otherwise -> stay single-device/native and simple.

This is a policy-contract test only. It does not benchmark or select a production backend.

## CPU safety bound

The LOCAL_FREE Gloo probe is checked to reject `world_size > 8` before spawning processes. PR #74 already owns the real bounded CPU/Gloo collective smoke test; this follow-on does not duplicate it.

## Evidence boundary

Passing this matrix proves deterministic contract algebra under small synthetic topologies. It does not prove NCCL, multi-node scheduling, network fabric, GPU memory capacity, actual FSDP2/TP/PP/CP/EP model numerics, DCP I/O durability, elastic node replacement, Float8 quality, throughput/MFU, or paid-scale readiness. It does not authorize CANDIDATE/STABLE or an audit verdict.
