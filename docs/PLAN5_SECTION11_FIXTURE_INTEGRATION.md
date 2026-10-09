# Plan 5 / Section 11 — Agent runtime fixture integration v1

**Owner:** Plan 5; conflict keys `agent-runtime` and `tool-contracts`.
This is an independent LOCAL_FREE integration fixture, not Nika Core,
a production ModelGateway, a real computer/tool executor or final Plan 10 release.

## Canonical composition (no duplicate authorities)

- `fixture_integration.py` is an integration **driver** only.
- S1 `ContextEntry/build_context` selects bounded source-aware model inputs.
- S2 `TaskStore` alone persists task/control epoch/effect identity and receipts.
- S3 `MemoryStore` alone persists verified host-result memory and correction history;
  retrieval remains `memory_only` until independent live verification.
- S4 `WorldModel` holds deterministic, evidence-marked observed fixture state.
- S7 `ToolRegistry` alone describes tools, validates exact schemas/permissions and
  accepts host-verified typed effect receipts.
- S10 `prepare_secured_tool` checks host-issued exact grants, payload safety,
  authorization, expiry and scoped permissions **before** the S7 preparation.
- `EchoGatewayFixture` implements the replaceable `ModelGatewayFixture` proposal
  protocol; **a proposal is never a grant or an instruction with host authority**.
- `FakeHostTool` is the only effect adapter. It is injected and has no remote
  dispatch. A task effect is first transitioned to `unknown` durably; then a
  fake tool result is verified and resolved. Never dispatch again after unknown.

## Durable deterministic scenario

`tests/fixtures/plan5_agent_runtime_scenario_v1.json` pins the full canonical
world observation, goal and task identity. The SHA-256 scenario digest is bound
to `TaskStore.checkpoint_id` and the reserved `effect_id`. Replacing even one
world byte makes an existing task incompatible and fails closed.

A cold restart does not rerun the mock model after reservation. A pending effect
may be resumed with a new control epoch and newly host-authorized tool call.
An issued `unknown` effect **never** automatically retries; recovery requires
an independently verified exact external receipt. A resolved effect can complete
durable memory/checkpoint bookkeeping after restart without a second dispatch.
Repeated completed calls return the same TaskStore snapshot.

The reference host adapter does not persist a real executor ledger. The
`reconcile_unknown` verifier is injected by the trusted test host; checking
receipt identity alone is not proof of a production effect. No real providers,
credentials, model training, Nika application state or paid compute are used.

## Qualification

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_agent_fixture_integration_section11.py \
  tests/test_agent_security_section10.py \
  tests/test_agent_memory_section3.py \
  tests/test_agent_world_section4.py \
  tests/test_agent_tools_section7.py \
  tests/test_agent_scheduler_section6.py \
  tests/test_agent_task_state_section2.py
PYTHONPATH=src python -m compileall -q src/twelve_six_agent_runtime
ruff check src/twelve_six_agent_runtime/fixture_integration.py \
  tests/test_agent_fixture_integration_section11.py
```

The fixture tests cover: positive model → observed world → context → tool grant →
durable effect/receipt → memory, model replacement, idempotence, stop/resume,
process restart, never-retry unknown effect, independently verified reconciliation,
recovery after effect resolution but before memory, memory correction, scheduler
resource envelope, forged model proposals/permissions, denied host grants, forged
tool results, immutable scenario identity and invalid schema. Hosted workflows
require exact-head completion before their result can be called PASS.

**Acceptance scope:** 12-6-owned agent-facing *reference component* only.
Plan 10 owns real model/provider/OS/Windows/NVDA and product integration.
