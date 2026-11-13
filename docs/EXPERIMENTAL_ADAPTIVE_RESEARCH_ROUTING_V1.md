# Experimental adaptive research routing v1 (2026-10-14)

**Status: independent LOCAL_FREE reference component; NOT a complete agent or product.**
Owner requirements: `docs/OWNER_AGENT_FIRST_RESEARCH_REQUIREMENTS_2026-10-13.md`; tracking: #3230.

## Problem and composition boundary

The 12-6 system already has:
- Plan-4 Section-8 multi-model serving / bounded provider role slots;
- Plan-5 TaskStore/Scheduler/ToolRegistry for task-state, leases and authenticated effects;
- Plan-6 teacher council for bounded independent candidate evidence.

What was missing from the owner-requested cognitive behavior was an independent, **dynamic graph route-planning policy**. This component is `src/twelve_six_agent_runtime/adaptive_research_routing.py`. It never dispatches a model, opens a browser, reads a secret, executes code, creates a checkpoint or promotes a training candidate. It must be embedded by a trusted host that independently controls model lifecycle, authentication, actual memory reservations and side effects.

## API

- `Specialist`: pinned model identity, local/external/deterministic provider, declared capability set, memory/CPU demand, maximum charged microunits, priority and allowed successor IDs. Can represent one role served by a shared model, rather than requiring physically distinct checkpoints.
- `RoutePolicy`: bounded call count, per-specialist revisit count, allowed no-progress streak, cost ceiling and explicit authorization for external/paid providers. All are trusted host inputs, **not model-created grants**.
- `ResourceReading`: current trusted capacity reading in MiB and abstract CPU units, supplied before every route decision. This reading is a declared admission signal, not reservation or evidence of process memory release.
- `RouteState`: task identity, required next capability, history, independently admitted receipts, cumulative cost, stagnation and revision. Canonical JSON snapshot plus SHA-256 checksum supports deterministic serialization/restore **but provides integrity only, not authentication or standalone crash-safe persistence**. Host must persist and bind it to the existing Plan-5 TaskStore control epoch and verify effect receipts.
- `select_next`: deterministic, side-effect-free single-model selector. It can skip specialists, backtrack via allowed successor graph, choose selectively based on capability, previous visits, priority, resource readings and costs. Returns `SELECTED`, `WAIT_RESOURCES`, `BUDGET_EXHAUSTED`, `STOP_NO_PROGRESS` or `NO_ELIGIBLE_ROUTE`. No infinite wait or unbounded model dialogue.
- `accept_outcome`: host-owned step outcome admission. Recomputes the selection under the **current** policy/capacity, checks task/revision/node/receipt/charge and requires an independently supplied verifier callback. A model response cannot validate itself. The verifier receipt must bind to exact external effect/model/inputs in the integrating host; this reference alone does not verify claims, source truth or actual cost.

V1 intentionally selects one model at a time; parallel execution, actual unload/load telemetry, async receipt reconciliation, promotion/weight training and live provider adapters are future integration work.

## Demonstration scenario

A trusted host might register nodes with paths
`sensory -> logic OR causal -> verifier -> sensory`.
Task state requests capability `sensory`, then an independently verified intermediate result changes next capability to `causal`. The next route picks `causal` rather than walking every node. A later verifier result selects `sensory` for another iteration. Rejected/no-progress routes stop at a fixed budget; unavailable RAM produces `WAIT_RESOURCES` instead of a false success.

This is **not** a guarantee that repeated model passes improve reasoning quality. Compare against a single-model baseline and fixed-route policy using quality, wall time, token counts, disk model loads and RAM/cost measurements.

## LOCAL_FREE qualification

`PYTHONPATH=src python -m pytest -q tests/test_adaptive_research_routing.py`

`python -m ruff check --select E4,E7,E9,F src/twelve_six_agent_runtime/adaptive_research_routing.py tests/test_adaptive_research_routing.py`

`python -m compileall -q src/twelve_six_agent_runtime/adaptive_research_routing.py tests/test_adaptive_research_routing.py`

Tests cover non-linear cycles, fail-closed resource admission, external+paid flags, over-budget and stagnating runs, stale/forged/replayed receipts, snapshot tampering, catalog integrity and deterministic selection. A full deployed agent still requires Plan-4/5/6/10 integration, real model execution, physical RAM measurement, integration acceptance and NVDA operator qualification; do not label the product DONE based on these tests.
