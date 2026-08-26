# POSTBASE-257 verifier ensemble V1

Worker: `POSTBASE-257-VERIFIER-ENSEMBLE-V1`

## Scope

This is a post-Base reasoning component. It is rooted on the learned execution spine but does not modify canonical Base model math, weights, tokenizer, checkpoint format, training semantics, or generation semantics.

The module separates deterministic verification from future model-based judgment. No external LLM is called by this implementation.

## Status model

Every verifier and aggregate claim result uses one of four statuses:

- `PASS`: the verifier's scoped check passed.
- `FAIL`: a deterministic check established a scoped failure.
- `INCONCLUSIVE`: available evidence is insufficient for the scoped check.
- `CONFLICT`: evidence or candidates disagree and the checker cannot select a truth value.

Every verdict carries explicit reason codes. The ensemble additionally records verifier id, verification dimension, deterministic/heuristic class, and per-claim status.

## Deterministic verifiers

1. `ExactAnswerFixtureVerifier` performs type-strict recursive exact comparison of fixture values.
2. `UnitTestCodeVerifier` verifies structured local test evidence and rejects malformed or zero-test success claims.
3. `NumericCalculatorVerifier` evaluates a bounded arithmetic AST without `eval`, names, calls, attribute access, indexing, or imports.
4. `ConsistencyChecker` detects contradictions in structured facts for the same claim/key. A consistency pass means only that supplied facts agree; it does not establish external truth.
5. `SourceProvenanceChecker` validates source identifiers, locators, and optional SHA-256 bindings. Provenance completeness is not treated as semantic correctness.
6. `CrossCandidateContradictionChecker` deterministically detects structured disagreement across candidates and returns `CONFLICT`; it does not choose a winner.

## Model-judge seam

`ModelJudge` is a protocol only. No model-judge implementation, provider SDK, HTTP client, remote endpoint, or external LLM call is present. Future model judgments are heuristic evidence and cannot independently promote a claim to verified correctness.

## Aggregation invariant

A deterministic `FAIL` is hard. If any deterministic verifier returns `FAIL` for a claim, the aggregate claim status is `FAIL` with `HARD_DETERMINISTIC_FAILURE`. A contrary heuristic `PASS` is retained as `VERIFIER_DISAGREEMENT`; it cannot override or hide the hard failure.

Without a deterministic correctness pass, heuristic/model-judge support remains `INCONCLUSIVE` with `HEURISTIC_ONLY_SUPPORT`. Source presence, internal consistency, and cross-candidate agreement also do not by themselves establish semantic correctness.

## Final-answer controller

`FinalAnswerController` maps aggregate evidence into explicit dispositions:

- `VERIFIED`: aggregate `PASS` backed by deterministic correctness evidence.
- `REJECTED`: aggregate `FAIL`.
- `CONFLICTED`: aggregate `CONFLICT`.
- `PROPOSED`: aggregate `INCONCLUSIVE`.

This lets a downstream answer controller distinguish verified claims from proposals rather than flattening all generated text into one confidence class.

## Adversarial coverage

The focused suite includes cases where:

- a weak heuristic verifier says `PASS` while an exact-answer fixture deterministically fails;
- a weak heuristic verifier says `PASS` while unit tests deterministically fail;
- two weak verifiers disagree, producing explicit `CONFLICT` rather than accidental promotion;
- heuristic-only `PASS` remains proposed/inconclusive;
- source provenance passes but no correctness verifier exists, so the claim remains proposed;
- candidate outputs contradict each other;
- internally inconsistent structured facts fail;
- the numeric calculator rejects call/import-like expressions;
- nested exact comparison does not conflate `bool` and `int`;
- zero-test successful process evidence is rejected as invalid proof.

## Truth boundary

The ensemble verifies scoped evidence. It does not infer philosophical or scientific truth from heuristic model scoring, candidate majority, source existence, or internal consistency alone. A future scientific verifier must define an explicit deterministic or evidence-qualified contract appropriate to that claim class.

## Execution boundary

The dedicated workflow is `LOCAL_FREE`, CPU-only through the repository universal bootstrap, and tests/lints only the isolated post-Base surface. No paid compute is requested.
