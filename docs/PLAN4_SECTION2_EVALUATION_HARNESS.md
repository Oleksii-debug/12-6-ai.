# Plan 4 — Section 2: general deterministic evaluation harness

Scope: LOCAL_FREE/public-fixture, candidate-neutral evaluation. This complements the
terminal/reserved-data boundary in `tools/evaluation_vault.py` (Section 1). It
**does not** accept reserved data, authorize final-test access, expose training
answers, authenticate arbitrary callback code, or authorize paid training.

## Contracts

* `FrozenSuite(suite_id, dataset_version, cases)` accepts only immutable,
  bounded public fixture cases. Each entry is (case_id, prompt, reference_answer).
  Duplicate IDs, malformed content, mutable cases or a non-public visibility are
  refused. Suite identity SHA-256 includes **all** case content and version.
* `FrozenProtocol(suite_sha256, seeds, metric, evaluator_version)` binds exact
  evaluation protocol, seed sequence and metric. Unknown metrics, repeated seeds,
  boolean/negative/out-of-bounds seeds, and suite mismatch fail closed.
* `evaluate(...candidate_sha256, predict)` passes only (prompt, seed) to the
  trusted-local inference adapter; reference answers stay inside the evaluator
  call. Two identical calls per case/seed must return identical text. Exceptions,
  non-text predictions, non-finite numeric values and unsupported metrics are
  failures, not zero scores or PASS results.
* Supported metrics: exact-string-match accuracy (higher is better) and
  mean-absolute-error over finite numeric strings (lower is better). A 95%
  descriptive Wilson interval is emitted for accuracy; MAE uses a normal-
  approximation descriptive interval when at least two observations exist.
  Insufficient observations produce explicit null bounds, not false certainty.
  Case/seed observations may be correlated: intervals **do not** establish a
  statistically independent confidence claim.
* Each machine-readable report includes suite, protocol, evaluator, candidate,
  metric, seed list, observation count, score, descriptive uncertainty and a
  canonical SHA-256 self-digest. `compare` rejects differing frozen
  suite/protocol/metric/seed/evaluator identities, and refuses self-comparison.
* `publish_report` is create-only and crash-aware; restart may re-publish only
  byte-identical evidence. `read_report` checks the digest and strict JSON
  (duplicate keys and non-finite values reject).

## Qualification

`python -m pytest -q tests/test_evaluation_harness_plan4.py`

`ruff check tools/evaluation_harness.py tests/test_evaluation_harness_plan4.py`

`.github/workflows/plan4-evaluation.yml` tests Sections 1–2 for relevant
pull requests and main pushes. Exact-SHA CI, integration and post-integration
readback are separate terminal acceptance gates.

No champion performance or cross-plan model-quality claim follows from public
fixtures. Plan 9 binds real candidate/checkpoint identities when available.
