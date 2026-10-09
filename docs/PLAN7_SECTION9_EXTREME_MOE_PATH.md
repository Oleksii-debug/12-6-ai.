# Plan 7 / Section 9 — 300B/1T and sparse MoE engineering path

## Accepted engineering boundary
- Reuses ModelSpec dense GQA/SwiGLU/RoPE arithmetic, Plan 7 Section 8 node
  placement and transport simulation, Sections 6–7 ordered SHA-verified physical
  tiny shard inspection, typed resources, backend contracts and fail-closed
  admission; does not implement duplicate checkpoint or data-run authority.
- 300B and 1T **dense reference** specifications are arithmetic-only with
  deterministic canonical ModelSpec hashes and 10% target tolerance.
- Optional expert architecture exposes total parameter count, active parameter
  count, dense active ratio, per-expert half FFN width, conservative full expert
  training/checkpoint floors and explicit no-implicit-paid-compute markers.
  The sparse total can be much larger than the named dense target.
- Seed-locked hash top-k tiny synthetic router: deterministic token→expert
  assignments, bounded expert count/top-k/capacity, per-expert load/imbalance
  receipt and explicit NO_GO on overload. No silent dropped assignments.
- Expert-parallel group divisibility, worker count, deterministic two-owner
  expert placement and simulated node-loss failover; both owners unavailable
  forbids recovery. Transport receipt tied to run, placement, loss, manifest
  and SHA-256-bound ordered physical bounded shard bytes.
- Signed-hash-shaped independent serving and evaluation contract identifiers
  are **bindings**, not evidence that those producer services ran.
- Conservative full-parameter memory, dual checkpoint space, worker/node
  topology, throughput, transport, interconnect, wallclock, recovery SLA,
  hypothetical hourly USD/budget checks; real backend absent means NO_GO.

## Bounded qualification
`python -m pytest -q tests/test_plan7_billion_systems_gate.py
tests/test_plan7_large_scale_paths.py tests/test_plan7_very_large_paths.py
tests/test_plan7_extreme_moe_paths.py`

LOCAL_FREE isolated source-compatible mirrored-context test: **29 passed /
0 failed**, Python compileall PASS, 100-column and whitespace static PASS.
Includes both dense profiles, two MoE scale recipes, routing replay, overload
and zero silent loss, node loss/same-run SHA fencing, corruption/partial/
duplicate/reordered checkpoints, forged state, invalid policy/tokens, eight
resource denials and economic budget denial. The standalone mirror is not an
assertion of a full live repository CI PASS; use the dedicated hosted exact-head
workflow when available, and report QUEUED accurately.

## No overclaim
No 300B/1T tensors, actual MoE expert execution, real multi-node failover,
measured peak GPU RAM, real bandwidth or checkpoint publisher, paid cloud/GPU,
authorized training, run exposure, serving, Plan 9 campaigns, whole-product
release or production cross-plan integration was performed. Component DONE
means qualified LOCAL_FREE reference controls and frozen producer boundaries.
