# POSTBASE-357 Verifier Ensemble Convergence

Worker: `POSTBASE-357-VERIFIER-ENSEMBLE-CONVERGENCE`

Execution profile: `LOCAL_FREE` only.

## Scope

This worker independently verifies the current POSTBASE-257 verifier-ensemble candidate without modifying its production implementation.

Exact candidate head:

`8744c8ab4c21299ed5fd12e937ab2dadb92d574e`

Exact production source Git blob:

`src/twelve_six/postbase/verification.py` = `e3c0504c1a6b3768c8aaea1aaaa3b3eb637eaab7`

The POSTBASE-357 workflow fails if that production blob changes on this branch or if the branch is not descended from the exact candidate head.

## Candidate evidence independently rechecked

The candidate contains deterministic verifiers for:

- exact-answer fixtures;
- unit-test/code evidence;
- bounded numeric calculation;
- source provenance;
- internal consistency;
- cross-candidate contradiction.

Aggregation exposes exactly `PASS`, `FAIL`, `INCONCLUSIVE`, and `CONFLICT`.

The critical precedence rule is fail-closed: a deterministic `FAIL` is checked before conflict/pass aggregation, records `HARD_DETERMINISTIC_FAILURE`, and cannot be promoted by a heuristic/model-judge `PASS`.

The candidate's own dedicated POSTBASE-257 workflow run `32961994373` completed `success` on the exact candidate head. The separate generic repository CI run `32961999048` completed `failure`; POSTBASE-357 does not relabel that repository-wide generic CI as green.

## Independent convergence matrix

POSTBASE-357 uses new fixtures, separate from the POSTBASE-257 test file, with all six required deterministic verifiers active in one ensemble plus a deliberately positive heuristic confidence verifier.

Required outcomes:

| Fixture | Required result |
| --- | --- |
| all six deterministic verifiers support the claim | `PASS` |
| exact-answer mismatch while heuristic passes | `FAIL` |
| unit-test/code failure while heuristic passes | `FAIL` |
| numeric mismatch while heuristic passes | `FAIL` |
| invalid provenance while heuristic passes | `FAIL` |
| internal structured contradiction while heuristic passes | `FAIL` |
| cross-candidate contradiction with correctness otherwise supported | `CONFLICT` |
| provenance/consistency/agreement support without deterministic correctness support | `INCONCLUSIVE` |

Every deterministic-failure fixture must also expose both `HARD_DETERMINISTIC_FAILURE` and `VERIFIER_DISAGREEMENT` in its reasons.

## Determinism and isolation

`tools/verify_postbase357_convergence.py` writes a canonical JSON evidence report with no wall-clock fields. CI materializes it twice and requires byte identity.

The verification path is AST-scanned for imports of external LLM/network clients (`openai`, `anthropic`, `requests`, `httpx`, `aiohttp`). No external model judge is implemented or called.

No model training, optimizer update, network verification, external LLM, paid compute, or Base-weight modification is authorized or performed.

## Finalization rule

Component convergence may be accepted only if the exact-head POSTBASE-357 workflow is terminal `success` and the deterministic report records `PASS_COMPONENT_CONVERGENCE`. Repository-wide release health remains a separate gate.
