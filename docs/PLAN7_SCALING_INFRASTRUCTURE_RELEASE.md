# Plan 7 — Scaling Infrastructure release candidate (LOCAL_FREE v1)

## Contract and acceptance boundary

This is **Plan 7 Section 12** terminal component qualification and integration,
not an assertion that training at 200M/1B/70B/1T has occurred. Its scope is
the independent scaling/distributed *mechanisms* only. Plan 9 owns future
training campaigns; Plan 10 owns final whole-product integration and
Windows/NVDA/physical/cloud acceptance.

The already accepted canonical implementations are reused without forks:

- Section 2: `accelerated_scaling.py` model growth, parity and budget authority.
- Section 5: `product_scale_recipes.py` resource/learned-proof admission authority.
- Section 6: `billion_systems_gate.py` shard hash/capacity contracts.
- Section 9: `extreme_moe_paths.py` sparse expert-parallel contracts.
- Section 10: `distributed_control.py` logical exposure, adapter/rank,
  checkpoint commit and same-run node-loss recovery.
- Section 11: `distributed_transport.py` transfer receipts, durable local
  fixture publisher and telemetry/readback.

No second scheduler, checkpoint publisher, optimizer, model implementation,
training approval or promotion authority is introduced by Section 12.

## Terminal acceptance matrix

`tests/test_plan7_terminal_scaling_qualification.py` executes the cross-owner
integration at tiny CPU scale:

1. An actual random-init tiny decoder grows function-preservingly under
   joint parent+descendant resource bounds, with a new descendant model identity.
2. The resulting actual `state_dict` bytes are serialized, physically split
   into eight bounded worker shards, hash-verified, committed as one logical step
   and received with independent transfer timings/receipts.
3. The staged checkpoint is published/read back on a local filesystem.
   Physically reassembled bytes reload through the real decoder and give
   bit-for-bit equal logits; no hash-only/mock weights are substituted.
4. Simulated epoch advancement preserves run/recipe/data-order/committed cursor;
   duplicate exposure after recovery is rejected.
5. Partial, reordered, corrupt, replayed and identity-forged shards/receipts
   fail closed; interrupted/uncommitted steps do not advance logical exposure.
6. Worker-memory, checkpoint-storage, interconnect and paid-compute admission
   denials are explicit; even a passing proxy remains non-launch authority.
7. The 200M admission recipe refuses missing real 20M proof and insufficient
   memory/checkpoint/throughput; unbound/forged proxy and 300M bridge reuse
   do not receive authorization.

The dedicated workflow `plan7-terminal-qualification.yml` also runs the
incumbent S2/S5/S6/S10/S11 tests (plus Ruff and compileall) on the same PR
candidate. A green or queued workflow on a different SHA does not qualify
a moved candidate. Exact-head readback and accepted-main readback are required.

## Truth and resource boundary

- Execution class: **LOCAL_FREE**, CPU fixture/simulation.
- Trained Base/champion weights: **NOT produced**.
- Physical multi-node/DCP/cloud execution: **NOT performed**.
- Learned 20M proof: **NOT produced by this plan**.
- No downloaded proprietary weights, purchased cloud/GPU compute, real training
  lease, launch/promotion approval or external serving.
- Physical network throughput, storage capacity and recovery times:
  **unmeasured**; transport figures are observed on local deterministic fixtures.
- Final status must only be called `DONE` after applicable runnable tests,
  exact integration and durable canonical GitHub/Drive readback.

## Integration and release identity

The accepted Git commit, component Git blob SHA identities, precise executed
test results, and explicit hosted-CI limitations are recorded in the
Plan-7 Section-12 row of `MULTI_PLAN_CLOSURE_STATE.md` after integration.
This document records the contract and release inventory, and is deliberately
not itself a training/production deployment manifest.
