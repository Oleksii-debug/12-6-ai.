# Plan 2 / Section 10 — cluster-safe split component (LOCAL_FREE)

S10 reuses incumbent twelve_six.split_robustness mechanics at exact Git blob e49518f2e431dc8576005be4938c1f15881897c4. It does not create a second split scheduler or authority. Versioned policy pins Git blob, deterministic seed and 20% test plus 25% of remaining validation fraction.

## Source and family boundary

The tool tools/plan2_cluster_split_v1.py consumes the self-hashed Plan-2 S9 candidate identity, exact selected record IDs and document source identity. It rejects missing, duplicate, extra and false-source rows, S9 permission escalation, changed policy, changed incumbent implementation, corrupt or symlink outputs. All records with the same document source ID remain in only one partition: train, validation or test. Never perform line-level random splitting. Persist exact assignments, canonical corpus and dedup-relations identity, seed, policy digest and immutable manifest self-hash. Never persist source text or silently grant training permission.

Incumbent SplitRecord(training_eligible=True) is solely a transient in-memory adapter required by canonical split mechanics; real S9 and output authority flags remain false.

## Physical-data qualification boundary

The current physically materialized S9 LOCAL_FREE candidate has 50 normalized records from ONE Kubernetes document (uk.kubernetes.docs.what-is-kubernetes). This is one independent document family, NOT 50. It cannot be honestly split three ways without document-family leakage. S10 therefore fails closed: NO actual training/validation/test release is claimed.

A controlled synthetic fixture has 16 independent document IDs / 32 records; it qualifies only the component, not a real S8 or production dataset. When additional lawfully admitted independent documents exist, re-run this component to make a usable physical release, preserving versioned authority.

## Qualification

- PYTHONPATH=src:. pytest -q tests/test_plan2_cluster_split_v1.py tests/test_split_robustness.py
- ruff check tools/plan2_cluster_split_v1.py tests/test_plan2_cluster_split_v1.py (runner only, local Ruff unavailable)
- Run tools/plan2_cluster_split_v1.py twice at the same destination and once in a clean destination. Compare receipts and split manifest byte for byte.
- Negative cases: one-document fail-closed, forged S9 identity and permissions, missing/extraneous/duplicate rows, source drift, forged policy, modified incumbent, tampered manifest and symlinks.

No paid compute, actual training, tokenizer fit, real final-test access or model promotion. Section completion is only a repository-controllable LOCAL_FREE component closure under Migration Contract Baseline v1.
