# Plan 7 / Section 8 — 30B, 70B, 100B very-large path

## Component acceptance
- `very_large_paths.py` reuses accepted Plan 7 S6 physical shard verifier,
  S7 typed capacity/limits/adapter evidence and canonical ModelSpec. There is no
  parallel checkpoint authority, no model construction and no training dispatch.
- Exact ModelSpec parameter arithmetic: 30B / 70B / 100B dense GQA/RoPE/SwiGLU
  reference paths. Dense and conditional sparse alternatives have versioned
  independent receipts. Sparse remains NO_GO until Section 9 expert contracts;
  it does not receive speculative memory discounts.
- Conservative 24x parameter training-state and 4x dual-checkpoint byte floors,
  constrained worker/node placement, storage, throughput, interconnect,
  checkpoint transfer time, recovery SLO and wallclock.
- Typed dollar/hour and total budget evidence is checked conservatively.
  No receipt grants paid compute, training, launch or canonical publication.
- Multi-node deterministic primary/replica transport *simulation*: bound exact
  scientific run, ordered physical tiny-shard hashes and manifest, worker losses
  and survivor placement. Missing/corrupted/duplicate/reordered fragments or
  both owners lost never count as a canonical checkpoint.
- Producer boundaries: Plan 2 data, Plan 4 evaluation, Plan 9 campaigns and
  Plan 10 real distributed/serving integration are separate authorities.

## Bounded qualification command
`python -m pytest -q tests/test_plan7_billion_systems_gate.py
tests/test_plan7_large_scale_paths.py tests/test_plan7_very_large_paths.py`

The Section 8 standalone, no-paid-compute fixture suite has 16 parametrized
cases including deterministic 30B/70B/100B recipe identity, loss/restart, SHA
fencing, missing/corrupt/reordered/duplicate shard, admission insufficiency,
invalid/forged types, cost ceiling, spare capacity and sparse fail-closed
checks. Verify exact candidate HEAD and the CI result separately; queued CI is
not green.

## Evidence boundaries
Arithmetic parameter/state/transport estimates are NOT measured peak memory,
real cluster throughput or hardware cost. `inspect_transport` verifies bounded
bytes locally and simulates remote placement; it does not move bytes among
machines or issue a production sharded checkpoint. No 30B/70B/100B weights,
multi-node training, production resume, serving, actual compute authorization,
GPU charges, Plan 9 campaign or Plan 10 release has been executed.
