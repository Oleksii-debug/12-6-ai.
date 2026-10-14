# Owner adaptive routing -> Plan-4 local serving bridge (candidate v1)

**Scope:** LOCAL_FREE integration reference layered atop the adaptive selection component from PR #3232, and incumbent accepted Plan-4 S8 serving. No mutation to Plan-4, canonical Base or existing Plan-5/6 authority.

`src/twelve_six_agent_runtime/owner_adaptive_research_serving_bridge.py` composes a trusted `select_next` decision with an already-published Plan-4 ModelServing **local** generation role. It verifies the exact pinned `model_weights_sha256` from the live catalog, forbids multiple slots with the same role, invokes **one** model with explicit `failover: false`, and requires an independent trusted host callback to decide whether that result is acceptable progress. Then and only then is a typed route outcome admitted.

No arbitrary URL, credential, file, OS tool, paid charge, external model, remote action or model-generated permission is accepted here. No auto-retry after a service failure because provider execution can be ambiguous. Candidate reply is not promoted to trusted fact merely because a model returned it; verifier must make its own evidence checks, outside this module. Caller owns canonical TaskStore effect ledger, permissions and durable source provenance.

## Test acceptance

`PYTHONPATH=src pytest -q tests/test_owner_adaptive_research_serving_bridge.py tests/test_adaptive_research_routing.py tests/test_plan4_multimodel_serving.py`

`ruff check --select E4,E7,E9,F src/twelve_six_agent_runtime/owner_adaptive_research_serving_bridge.py tests/test_owner_adaptive_research_serving_bridge.py`

Fixture tests **actually invoke** the existing local serving plane and inspect pinned identities and usage, then verify selective route progression and no double invocation. They cover low-memory wait without generation, attempted identity substitution, unaudited results, ambiguous backend failures and lack of implicit external/paid access.

These are fixture-based model service calls, NOT a trained online 12-6 model, interactive browser, live local LLM, general research agent, full autonomous dispatcher or deployment proof. Physical qualification with real model weights, measured RAM/CPU/time and NVDA surface remains open.
