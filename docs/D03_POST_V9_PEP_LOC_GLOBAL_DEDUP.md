# D03 post-V9 PEP + LoC global dedup composition

This package is the finish-first bridge from two already executed, independently
qualified LOCAL_FREE candidate materializations into the incumbent D03 global-dedup
science. It is stacked on PR #1055 exact head
`5dbf143c7ca15999eadb28fecb952118b59040de`.

It does **not** create a new matcher, a new source family, corpus credit, tokenizer
authority, or training authority.

## Why this is on the learned-20M critical path

The current DATA526 post-decontamination survivor ceiling is 5,921,963 unique causal
loss positions, below the binding LEARN-345 meaningful floor of 10,000,000.
Therefore source capacity is now a real blocker in addition to the remaining
mechanical gates.

Two candidate families have already completed real LOCAL_FREE materialization:

- PEP public-domain candidate: 86 records and 4,987,775 normalized UTF-8 bytes.
- Library of Congress selected digitized books: 28 records and 4,499,908 normalized
  UTF-8 bytes.

Neither candidate receives corpus or training credit from those executions. This
package only makes them eligible to proceed through the same global-dedup stage.

## Exact authority bindings

PEP is bound to:

- Product head `930471ef19c1afff70dee6ed46c4813ded035c9e`;
- repaired evidence head `facef815c8c3deac0dfa1843928241e162c7864b`;
- real execution run/job `34570806209/103172292323`;
- candidate SHA-256
  `65d1cf23354efc5a66d7132b67602a7af1715e18a0fb77e729595866c25d33c3`;
- report SHA-256
  `bc217b3b795bfe5001b8ef266270eee10089c88bc148e8a8f0fa540c2451edcb`.

LoC is bound to:

- terminal evidence head `cdde55b75a10639e8233de7ccccaaea335b4055a`;
- execution head `8ce516160b91ba48b4dd4096e0ffeea409d221de`;
- real execution run/job `34606219695/103285166822`;
- candidate SHA-256
  `f87201d71eb8c3f2510cdf2c9b3ce1a32b83d1a11198e8a1ff0ad67d0a3fec20`;
- report SHA-256
  `ae8de852ffeea6d2f697641b2f941eeb044e3377565d0bc1b3b7ba8b6cf3133a`.

The validator requires the exact candidate/report byte sizes and hashes before
semantic parsing. JSON and JSONL parsing rejects duplicate keys. Parsed rows are then
cross-bound to report counts, normalized byte totals, inventory identities, source
revisions and zero-credit eligibility boundaries.

## Matcher reuse

`run_expanded_dedup_v10()` first executes the hardened V9 entry point over exact
DATA526 + Rada authority. That step reuses the terminal PR #824 V3 matcher and V9's
runtime-identity closure.

Before the PEP+LoC-expanded call, V10 invokes the V9 matcher semantic-closure verifier
again. It then reconstructs the exact V9 matcher inputs, appends only authenticated
PEP/LoC matcher objects, and calls the same `audit_payloads()` / `verify_report()`
callbacks.

No matching threshold, normalization, fingerprinting, fragment, lineage,
connected-component, duplicate-cluster or survivor-selection rule is changed.

## Real execution

The runner consumes job-local candidate payloads. Raw text is never committed.

```bash
python tools/run_d03_expanded_global_dedup_v10_pep_loc.py \
  --v7-root /path/to/exact-v7-root \
  --bulk-workspace /path/to/fresh-bulk-workspace \
  --v8-report /path/to/v8-a.json \
  --v8-survivors /path/to/v8-survivors-a.json \
  --rada-quality-privacy-jsonl /path/to/rada-quality-privacy.jsonl \
  --rada-quality-privacy-report /path/to/rada-quality-privacy-report.json \
  --expected-rada-report-sha256 <independent-qualified-rada-report-sha256> \
  --pep-candidate /path/to/pep-candidate.jsonl \
  --pep-report /path/to/pep-report.json \
  --loc-candidate /path/to/loc-candidate.jsonl \
  --loc-report /path/to/loc-report.json \
  --output-report /tmp/expanded-v10-report.json \
  --output-survivors /tmp/expanded-v10-survivors.json
```

The current real run remains dependent on the active Rada Q/P execution and terminal
PR #1055 qualification/integration ordering. A blocked dependency is reported as
blocked; it is never converted into synthetic data.

## Scientific boundary

A V10 PASS means only:

1. the exact qualified PEP and LoC candidate bytes were authenticated;
2. exact hardened V9 DATA526+Rada authority was reproduced;
3. all four input groups were compared under incumbent PR #824 matcher semantics;
4. text-free report/survivor identities were produced.

It does not complete reserved-evaluation decontamination, canonical quality/privacy,
family balance/caps, cluster-safe split, deterministic packing, two-clean corpus
reproducibility, tokenizer fit, or positive unique-loss accounting.

Therefore after a V10 PASS:

- `training_authorized_bytes = 0`;
- `unique_causal_loss_positions_authorized = 0`;
- `tokenizer_fit_authorized = false`;
- `optimizer_updates = 0`;
- `model_training_executed = false`;
- `learned_weights_created = false`.

No paid compute, foreign pretrained weights, external LLM/API corpus intelligence, or
final-test payload access is introduced.
