# VERIFY-309 — Cluster-safe split v03

Worker: `VERIFY-309-CLUSTER-SAFE-SPLIT-V03`

Execution profile: `LOCAL_FREE`

## Scope

Independently validate that no exact-content identity, near-duplicate cluster, or record identity crosses `train`, `validation`, or `reserved` boundaries. Verify deterministic assignment evidence and source-family distributions. Run adversarial cluster fixtures.

## Authority binding

- Frozen build contract branch: `data300/corpus-v03-frozen-build-contract-20260826`
- Frozen build contract head: `8ea7f830e50a23754d189dd4134f4afad76a7ee9`
- Contract identity SHA-256: `07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5`
- DATA-301 branch observed during this audit: `data301/corpus-v03-terminal-build-20260826`
- DATA-301 observed head: `8ea7f830e50a23754d189dd4134f4afad76a7ee9`

The observed DATA-301 branch had not advanced beyond the DATA-300 contract commit. Therefore no Wave-3 clean-root manifests existed on that branch at the audit cutoff.

## Contract invariants checked

DATA-300 v2 requires:

- no dedup-cluster straddling;
- no exact-content overlap;
- no record-identity reuse;
- exact/normalized/near-copy clustering rerun on the exact Wave-3 materialization;
- reserved decontamination before training;
- two independently clean byte-identical builds.

DATA-298 is only prebuild evidence and does not authorize a final split by itself.

## Independent verifier

`src/twelve_six/verification/cluster_safe_split_v03.py` fails closed unless every manifest record carries:

- boundary;
- record identity;
- source identity;
- source family;
- exact content SHA-256;
- dedup/near-duplicate cluster identity.

It audits all three cross-boundary invariants and computes canonical assignment and source-family-distribution identities. It compares two clean roots and requires exact identity agreement.

The verifier reads only `manifests/final-test-reservation.jsonl` for the reserved boundary; it does not require final-test payload bytes.

## Adversarial fixtures

Independent LOCAL_FREE fixtures cover:

1. clean fixture passes;
2. input-order permutation preserves canonical identities;
3. exact-content alias crossing train/validation is rejected;
4. near-duplicate cluster crossing train/reserved is rejected;
5. near-duplicate cluster crossing train/validation across different source families is rejected;
6. record identity reused across boundaries is rejected.

Result: `PASS_ADVERSARIAL_FIXTURES` — 6/6 checks.

Local test command:

`PYTHONPATH=src pytest -q tests/test_verify309_cluster_safe_split_v03.py`

Result: `6 passed in 0.06s`.

Local self-test:

`PYTHONPATH=src python -m twelve_six.verification.cluster_safe_split_v03 --self-test-only`

Result: `PASS_ADVERSARIAL_FIXTURES`.

## Deterministic assignment and family distributions

The independent verifier canonicalizes records by `(boundary, record_id)` before hashing, so input iteration order cannot change assignment identity. Source-family records/known-byte counts are canonicalized before hashing and compared across clean roots.

The frozen v2 prebuild inventory is `uk=1`, `en=1`, `code=2` independent families. This is a separate DATA-295 balance/diversity blocker and remains blocked even if split-safety mechanics pass.

## Verdict

- Verifier mechanics: `PASS`.
- Adversarial fixtures: `PASS`.
- Actual DATA-301 Wave-3 split: `BLOCKED_NOT_YET_AUDITABLE`.
- Corpus frozen/terminal claim: `NOT_AUTHORIZED`.

Reason: at the audit cutoff DATA-301 still pointed at the DATA-300 contract head and exposed no built Wave-3 clean roots/manifests. A final actual-corpus PASS would be unsupported.

## Exact acceptance gate when DATA-301 materializes

Run the verifier against both `wave3-clean-a/corpus-v03` and `wave3-clean-b/corpus-v03` and require:

- zero cross-boundary record-ID reuse;
- zero cross-boundary exact SHA-256 overlap;
- zero cross-boundary dedup-cluster straddles;
- identical canonical assignment SHA-256 across clean roots;
- identical source-family distribution SHA-256 across clean roots;
- complete source-family and cluster evidence for every manifest record.

Any missing cluster/family evidence is `BLOCKED_INSUFFICIENT_SPLIT_EVIDENCE`; any detected crossing is `FAIL_CLUSTER_SAFE_SPLIT`.

No paid cloud/GPU/training compute was used.
