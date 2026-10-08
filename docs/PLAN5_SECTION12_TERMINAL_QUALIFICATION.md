# Plan 5 / Section 12 — terminal agent reference-runtime qualification

**Owner:** Plan 5, conflict keys `agent-runtime` / `tool-contracts`.
**Canonical source:** "5. П’ятий план" in the repository's indexed Google Drive plan folder.

## Acceptance boundary

This is the terminal **12-6-owned agent-facing reference component**, not Nika
Core or any production browser, computer, credential, model, device or OS runtime.
Plan 10 independently owns whole-product, physical Windows/NVDA and integration
acceptance. No production model, learned champion, external vendor or paid
GPU/cloud compute is exercised or authorized here.

Canonical authorities are unchanged:
- S1 context compaction/provenance;
- S2 durable TaskStore, effects and epoch fencing;
- S3 evidence-qualified MemoryStore and corrections;
- S4 observed/unknown/stale WorldModel;
- S5 evidence-only self-model;
- S6 resource-aware scheduler;
- S7 typed ToolRegistry;
- S8 semantic browser/computer DOM/AX/UIA reference adapter;
- S9 typed multimodal artifacts and privacy/locality/capability preflight;
- S10 trust/secret/effect-grant boundaries;
- S11 deterministic ModelGateway/fake-host E2E fixture.

S12 adds no second runtime or control authority. The versioned shared scenario
`tests/fixtures/plan5_agent_runtime_scenario_v1.json` remains the reproducible
fixture, with no external network, vendor access, training or production effects.

## Terminal acceptance matrix

| Gate | Evidence |
| --- | --- |
| Long-horizon and restart | 32 independent durable tasks: reserve, cold restart, epoch renewal, finish, no double tool/model call |
| Model replacement | Fresh deterministic fixture vs restarted alternative produce identical effect receipts, checkpoint IDs, memory and world semantics |
| Memory correction | Corrected memory is durable but remains `memory_only`; observed world cannot be overwritten by memory |
| Browser/computer | DOM, AX and UIA ambiguous targets deny issue; stale target generations cannot produce effects |
| Multimodal | Image, audio, ASR, TTS and vision retain source artifact identities across model IDs; remote/private transfer, unverifiable provenance/content and missing capability deny |
| Security | Model-prompt-injection content remains untrusted; forged owner tool request cannot create grants, tool effects or memory |
| Existing cross-component gates | All checked-in `test_agent_*.py` tests for S1–S11 (context, task/restart, memory, world, self-model, scheduler, tools, UI, media, security and fixture integration) remain passing |

## Executed LOCAL_FREE qualification

Exact source: Github branch candidate (see S12 PR and `MULTI_PLAN_CLOSURE_STATE.md`
for frozen SHA and accepted main SHA). S12 test blob:
`1bd214a04307779545ae0b393c31a08bd2e89acf`.

Run inside an independently reconstructed exact Git-blob checkout using only
Python 3, local SQLite, Pytest and deterministic fixtures:

```sh
PYTHONPATH=src python -m pytest -q tests/test_agent_*.py
PYTHONPATH=src python -m compileall -q src/twelve_six_agent_runtime tests/test_agent_terminal_qualification_section12.py
```

Result at qualification: **125 passed, zero failed**, Python compileall PASS.
The new S12 test source Git blob was independently recomputed and matched;
its maximum line width was 91, with zero tabs, trailing whitespace and conflict
markers. Existing legacy Plan 5 formatting debt is not a reason to reopen
already-terminal Sections 1–11. Ruff locally unavailable.

Hosted CI: if workflow runners remain queued/unavailable, **do not describe them
as PASS**. The local component qualification supports Simplified Section
Closure Protocol v3, not a claim of whole-repository green CI or production
evidence. Known actual test failures, if later discovered, require repair.

## Terminal candidate rule

The exact S12 test/workflow/document source, with the already merged S1–S11
runtime, forms the Plan 5 terminal reference candidate. After guarded merge
to `main`, the accepted integration SHA and exact Git blob readbacks must be
recorded in `MULTI_PLAN_CLOSURE_STATE.md` and the assigned Drive Plan 5
Section 12 must be marked DONE. Compatible future real ModelGateway arrival
does not automatically reopen this plan. Demonstrated regression, changed
acceptance contract or breaking integration does.
