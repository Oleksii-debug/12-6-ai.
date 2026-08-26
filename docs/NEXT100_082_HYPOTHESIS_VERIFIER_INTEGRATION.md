# NEXT100-082 Hypothesis-Verifier Integration

Worker: `NEXT100-082-HYPOTHESIS-VERIFIER-INTEGRATION`

Execution profile: `LOCAL_FREE` only. No external model judge is permitted by the integration controller.

## Authority composition

This integration selectively composes the terminal hypothesis-search head and the terminal verifier-convergence head:

- hypothesis search: `ea1d8fff0d3235660dffe7ba411e192df83f5e1d`;
- verifier convergence: `7eac24e250c0853745208bab8ba9b2d3d104fbf5`;
- selective two-parent composition commit: `7f8dbb88c6c9ad92968f66bae94103600893d357`.

The verifier tree remains the base. Only the hypothesis-search-owned module, tests, documentation, deterministic fixture, probe and workflow were imported from the hypothesis head. No wholesale replacement of shared project files was performed.

## Selection invariant

A heuristic search score is only a search priority. It cannot override deterministic verifier evidence.

`HypothesisSearch.best()` excludes any active hypothesis that has version-bound verifier evidence marked `deterministic_failure=True`. `HypothesisVerifierController.verify()` additionally converts an ensemble hard deterministic failure into an explicit hypothesis rejection with score `0.0`.

This gives two fail-closed layers: selection exclusion is immediate when deterministic FAIL evidence is recorded, and the normal controller path durably rejects the failed hypothesis.

## Verification states

The controller preserves the verifier ensemble's four public states without inventing a second verifier policy:

- `PASS`;
- `FAIL`;
- `INCONCLUSIVE`;
- `CONFLICT`.

A deterministic `FAIL` is a hard rejection. `CONFLICT` remains explicit and does not silently become PASS or automatic rejection. `INCONCLUSIVE` remains unresolved. Deterministic correctness PASS is recorded as such but does not rewrite the hypothesis's heuristic score.

## Exact immutable version binding

Every hypothesis now has:

- `lineage_id`;
- integer `version`;
- deterministic content-bound `version_id` of the form `Hxxx@vN:<sha256-prefix>`.

A branch starts its own version-1 lineage. A revision creates a new hypothesis object with a new ID and a new version ID, while retaining the prior object and its evidence unchanged. Verifier evidence is never inherited by a revision.

Verifier requests accepted by `HypothesisVerifierController` must contain exactly one claim whose `claim_id` equals the current hypothesis `version_id`. Evidence records store both the owning hypothesis ID and exact version ID. A request constructed for an older revision is rejected before verification can be bound to a newer revision.

## Deterministic-local verifier boundary

The integration controller refuses any verifier that is non-deterministic or declares the `MODEL_JUDGMENT` dimension. It uses the existing deterministic verifier ensemble only; there is no external LLM/model-judge call path.

## Test coverage

Focused integration tests cover:

1. an initially preferred wrong hypothesis with score `0.95` losing preference and being rejected after an exact deterministic FAIL;
2. explicit PASS and INCONCLUSIVE propagation;
3. contradictory candidate evidence producing CONFLICT;
4. simultaneous deterministic support and contradiction producing FAIL, `VERIFIER_DISAGREEMENT`, and hard rejection;
5. revision creating a distinct immutable version with no inherited verdict;
6. `best()` remaining fail-closed even before controller rejection is applied;
7. refusal of model-judge/non-deterministic verifiers.

The pre-existing POSTBASE-256 hypothesis tests are retained unchanged as regression coverage.

## Truth boundary

This component verifies supplied deterministic evidence. It does not claim that heuristic hypothesis scores are probabilities, that PASS proves unrestricted scientific truth, or that unresolved/conflicted hypotheses are correct. It does not alter Base weights, tokenizer, training, checkpointing, or alignment policy.
