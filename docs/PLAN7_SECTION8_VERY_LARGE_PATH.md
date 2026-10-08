# Plan 7 / Section 8 — 30B, 70B, 100B multi-node engineering path

## Plan acceptance 8.1 — scale, economics, admission
- Canonical Plan-7-owned `very_large_scale_paths.py` uses real accepted ModelSpec arithmetic and S7 typed capacity/backend evidence; no duplicate training/checkpoint/runtime authority.
- Versioned dense 30B/70B/100B architecture with 10% parameter tolerance and 24x-parameter training-state, 4x-parameter checkpoint dual-copy, BF16 weights planning *floors*. They are arithmetic projections, not peak RAM or physical benchmark.
- Sparse mode is `SPARSE_CONTRACT_ONLY`, explicitly blocked for expert parallelism until Plan7 Section9 produces and verifies the genuine MoE contract; it is not a sparse implementation.
- Multi-node worker/rank placements, exact SHA-256 placement map, distinct node/worker counts, throughput/network/checkpoint/recovery/wallclock and USD forecast all bounded by explicit limits. Economic budget failure returns NO_GO.

## Plan acceptance 8.2 — failure/recovery, artifact transport
- Reuses existing S6 ordered, physically hashed tiny checkpoint shard manifest verifier. A corrupt/missing/reordered/duplicate shard set fails before any acceptance. Shards are LOCAL_FREE fixtures, not published real 30B–100B checkpoints.
- Same run SHA on simulated node-loss/restart, exact recipe and data-order SHA on resumed run. Stale run, changed recipe, changed data order, forged placement SHA or unknown node fail closed.
- Explicit artifact ownership, atomic publisher, real sharded distributed optimizer recovery and physical multi-node transport remain externally produced contracts; no fabricated receipt or promotion.
- Backend matrix consumes S7 versioned LargeAdapter evidence; none of these receipts grants launch or paid/training authorization.

## Qualification and truth boundary
- Candidate branch: `plan7/section8-multinode-20261008`.
- Exact source Git blob SHA1 `4b6d16931512a240d4e3ddc4287eb2a02fd7d155`; test blob `849f5f732b229f33afa690cde75802c3621fed1e`.
- Independent LOCAL_FREE execution on exact S7+S8 code/test Git blobs with *narrow public-contract mirrors* of accepted S6 shard verifier and canonical ModelSpec: **34 pytest passed, 0 failed**, including 17 dedicated S8 scenarios, compileall PASS and source/test static width/whitespace PASS.
- These are component fixture tests, not a real multi-node test or production serving/evaluation/accelerator qualification. Hosted workflow is an independent check and must not be called PASS if queued/unavailable.
- No paid compute/training/GPU/cloud calls; no weights, real production optimizer/recovery, Plan9 campaign or Plan10 full-product release evidence.
