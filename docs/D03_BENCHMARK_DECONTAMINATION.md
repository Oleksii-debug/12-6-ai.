# D03 / DATA-31 benchmark decontamination

## Authority and incumbent

D06 `12-6.benchmark-registry.v1` remains the benchmark authority. DATA-31 does not create a competing benchmark registry. D03's existing `reserved_registry_from_d06_manifest` validates the D06 manifest identity and held-out semantics. Near matching remains the maintained DataTrove 0.10.0 MinHash implementation already introduced by DATA-12; DATA-31 adds a reference-only execution path so benchmark matches are not conflated with candidate-candidate near deduplication.

The validated MinHash configuration is 5-word shingles, 14 buckets, 8 hashes per bucket, 64-bit hashes, seed 1, and the exact DataTrove 0.10.0 wheel identity already pinned by D03. `skip_completed=true` remains enabled in every DataTrove stage.

## Current reference truth boundary

At the observed D06 PR #28 head `914973502ab92a925a5cc29d72e4b3cce0e81c80`, the repository contains the `BenchmarkRegistry` implementation but no persisted populated production benchmark manifest. The canonical empty D06 manifest therefore has identity `10f7454f77eb2dc3871eeafa5055b1969eab42954eb8e19e61565f217c67df31`. This is a snapshot of the current authority state, not a second registry and not a claim that future D06 registries remain empty.

The current D03 S0 corpus has two locally available held-out validation documents in `data/s0/packaged/validation.jsonl`. Their source is the project-authored controlled S0 fixture according to `data/s0/source_registry.json`, so they are admitted only for local contamination checking. No conclusion about external-dataset licensing is made. The D06 live pre-integration report records no real hashed S0 generation probe; DATA-31 therefore invents no probe or benchmark identity.

The current S0 fixture contains two registered cross-language semantic overlaps that lexical MinHash must not be credited with finding: training document `project-authored-s0-fixture-v1::doc-001` corresponds semantically to held-out Ukrainian document `project-authored-s0-fixture-v1::doc-007`, and training Ukrainian document `project-authored-s0-fixture-v1::doc-011` corresponds semantically to held-out English document `project-authored-s0-fixture-v1::doc-005`. Both candidate documents are explicitly rejected from the publication candidate.

## Exact index and decisions

Exact matching uses the existing normalized `content_sha256` identity. Every exact hit records candidate source/document ID, reference source/document ID, both hashes, and decision `REJECT_FROM_TRAINING`. Exact hits are removed before the near-match pass so the near-match audit is not double-counted.

## Near-match reference index

The two held-out references are written to the existing DataTrove MinHash reference-index machinery. Candidate signatures are queried with `only_dedup_in_index=true`, which deliberately ignores candidate-candidate matches and returns only documents matching the frozen reference index. Every removed candidate records its source/document ID, content hash, reference-bundle identity, engine, and rejection decision.

DataTrove 0.10.0's public filter output identifies the rejected candidate but does not expose the paired reference document ID. DATA-31 records that attribution limit explicitly. It does not fabricate a reference document identity. The exact reference bundle itself is cryptographically frozen, so the matched scope remains auditable.

## Registry freshness and publication binding

Every decontamination report records the exact D06 benchmark-registry SHA-256, exact reference-bundle SHA-256, candidate corpus-manifest SHA-256, exact matches and decisions, DataTrove near matches and decisions, registered semantic exclusions and decisions, residual semantic-overlap limitations, and a deterministic report SHA-256.

`assert_fresh_decontamination` fails closed if the current D06 benchmark-registry identity differs from the identity used by the completed pass. A corpus publication manifest cannot be built from a stale report.

The DATA-31 publication manifest directly binds the D06 benchmark-registry SHA-256, reference-bundle SHA-256, decontamination report SHA-256, original corpus-manifest SHA-256, and hashes of surviving physical output files. Thus a benchmark-registry change necessarily changes publication identity and requires a fresh decontamination pass.

## Injection tests

The contract suite injects an exact benchmark copy and requires both candidate and reference identities to be recorded and rejected. With DataTrove installed, an integration test injects a high-overlap lexical near copy, builds a real MinHash reference index, executes the reference-only candidate pass, and requires that candidate to appear in the removed set. A separate registry-drift test proves that a changed D06 registry hash invalidates the prior pass.

## Semantic limitation

No lexical MinHash configuration proves universal semantic cleanliness. Cross-language translations, paraphrases, short snippets, and semantically equivalent code with low lexical overlap can survive. DATA-31 reports known registered semantic exclusions, rejects them when evidence is available, and keeps `semantic_universal_cleanliness_claimed=false` in both the decontamination report and publication manifest. Unknown semantic overlap remains a stated residual limitation, not a cleanliness claim.
