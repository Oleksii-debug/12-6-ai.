# Plan 6 / Section 15 — LOCAL_FREE Evolution Engine release contract

This is the **independent engineering component** qualification/handoff, not
a production model, trained champion, external provider, Plan 9 campaign,
or Plan 10 whole-product release. Conflict keys: `post-base-learning` and
`evolution-engine`.

## Reuse, not replacement

The accepted Plan-6 Sections 1–14 remain the **only** authorities for
research, post-Base instruction/preferences/reasoning/tool trajectories,
teacher council, verified-data production, self-play, continual learning,
experience consolidation, curriculum, code evolution, autonomous research
and immutable champion→candidate→descendant lifecycle.

`configs/plan6/evolution_engine_release_v1.json` is the versioned Plan-9
**mechanism-handoff manifest**: exact Git blob SHA-1 of all fourteen component
modules, schema and permission limits. It is not a certificate that any
learned model is fit to use, or permission to run expensive training.

## Qualification contract

The canonical `.github/workflows/ci.yml` runs the **fourteen** existing
component suites and `tests/test_plan6_terminal_qualification.py` under
its repository-wide `pytest -q` gate, after its central workflow-budget policy
and `ruff check src tests` gates. No teacher/service/network,
production checkpoint or materially paid compute is required.

The terminal cross-engine test composes the incumbent Section-1 independently
verified five-stage research pyramid with Section-14 immutable lifecycle.
It validates the candidate's artifact/model identity, requires a separate
independently signed evaluation and **explicit unrelated authorizer** before
promotion, then JSON round-trip restart and signed rollback to the prior
immutable fixture champion. It also rejects omitted/forged/foreign/self-issued
research stages, failed evaluation, forged/stale grants, unauthorized re-promotion
and wrong-action rollback. A source hash parity test fails closed if any of
the exact fourteen manifest-bound implementation blobs drift.

The component suites exercise bounded synthetic/tiny CPU mechanisms,
negative/recovery/replay/policy gates as documented in the canonical
`MULTI_PLAN_CLOSURE_STATE.md` entries for Plan 6 Sections 1–14.

## Plan 9 handoff boundary

Plan 9 may consume versioned engine APIs and this manifest only after
Section-15 qualified integration/readback. It must independently bind any
real Base model, data, tokenizer, independent evaluation/holdout and run budget,
then authorize a campaign separately. No step here mutates Base weights or
production champion pointers, permits training/teacher billing, grants
self-verification, promotes a candidate without independent signatures, or
satisfies Plan-10 user/device/release acceptance.

## Terminal evidence ledger

- Exact fourteen incumbent source blob identities: in manifest; verified by
  `test_source_manifest_pins_all_fourteen_exact_engine_blobs`.
- Complete focused test matrix and cross-engine tests: included in shared CI
  `pytest -q`; the explicit 15-file LOCAL_FREE matrix independently passed
  **168/168** tests using the exact GitHub source/test files with canonical
  `twelve_six.training` imports on a local Python 3.13 CPU fixture; `compileall`
  also passed. This local result does not assert hosted CI success.
- GitHub shared-CI status and exact candidate SHA: read from the current
  workflow on the frozen candidate. The earlier dedicated-workflow attempt
  violated the incumbent workflow-budget gate; the extra workflow is removed,
  rather than bypassing or changing central CI policy.
- Accepted-main merge/readback and Drive Section-15 status: required before
  calling the overall plan terminal DONE.
