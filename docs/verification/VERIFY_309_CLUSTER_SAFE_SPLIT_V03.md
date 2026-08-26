# VERIFY-309 — Cluster-safe split v03

Worker: `VERIFY-309-CLUSTER-SAFE-SPLIT-V03`

Execution profile: `LOCAL_FREE`

## Scope

Independently validate that no exact-content identity, near-duplicate cluster, or record identity crosses `train`, `validation`, or `reserved` boundaries. Verify deterministic assignment evidence and source-family distributions. Run adversarial cluster fixtures.

## Exact authorities

- DATA-300 frozen contract head: `8ea7f830e50a23754d189dd4134f4afad76a7ee9`
- DATA-300 contract identity SHA-256: `07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5`
- DATA-301 audited head: `8820ba1b255f6bb95c7db0531fd846078a1aae01`
- DATA-301 evidence identity SHA-256: `939065abeefff8aed924415589608ff3fc721fe4b0a57fc200146a4b6a137e81`
- DATA-301 dedicated workflow run: `32978147377` / job `98207864001` / `SUCCESS`

The DATA-301 terminal evidence explicitly records `cluster_safe_split: NOT_REACHED`, `deterministic_sharding: NOT_REACHED`, and `two_clean_builds: NOT_PERMITTED_BECAUSE_PREBUILD_HARD_GATES_FAIL`. Therefore there is no actual Wave-3 train/validation/reserved split assignment to certify.

## Contract invariants

DATA-300 v2 requires:

- no dedup-cluster straddling;
- no exact-content overlap;
- no record-identity reuse;
- exact/normalized/near-copy clustering rerun on the exact Wave-3 materialization;
- reserved decontamination before training;
- two independently clean byte-identical builds.

DATA-298 is prebuild evidence only and does not authorize a final split by itself.

## Independent verifier

`src/twelve_six/verification/cluster_safe_split_v03.py` fails closed unless every materialized manifest record carries:

- boundary;
- record identity;
- source identity;
- source family;
- exact content SHA-256;
- dedup/near-duplicate cluster identity.

It audits `train`, `validation`, and `reserved` simultaneously, computes canonical assignment and source-family-distribution identities, and compares two clean roots.

For `reserved`, the verifier consumes only `manifests/final-test-reservation.jsonl`; final-test payload bytes are not read.

## Adversarial fixtures

Independent LOCAL_FREE fixtures cover:

1. clean fixture passes;
2. input-order permutation preserves canonical identities;
3. exact-content alias crossing train/validation is rejected;
4. near-duplicate cluster crossing train/reserved is rejected;
5. near-duplicate cluster crossing train/validation across different source families is rejected;
6. record identity reused across boundaries is rejected.

Result: `PASS_ADVERSARIAL_FIXTURES` — 6/6 checks.

Local test:

`PYTHONPATH=src pytest -q tests/test_verify309_cluster_safe_split_v03.py`

Result: `6 passed in 0.06s`.

Local self-test:

`PYTHONPATH=src python -m twelve_six.verification.cluster_safe_split_v03 --self-test-only`

Result: `PASS_ADVERSARIAL_FIXTURES`.

Fail-closed no-root probe:

`PYTHONPATH=src python -m twelve_six.verification.cluster_safe_split_v03`

Result: `BLOCKED_TWO_CLEAN_ROOTS_REQUIRED`, exit `2`.

## Deterministic assignment

Verifier mechanics are deterministic: records are canonicalized by `(boundary, record_id)` before assignment hashing, and input-order reversal produces the same assignment and family-distribution identities.

Actual DATA-301 assignment determinism is `NOT_REACHED / NOT_TESTABLE`: DATA-301 did not execute cluster-safe split or construct two clean roots.

## Source-family distributions

DATA-301 prebuild inventory is:

- `ua.rada.open-data.laws-texts`: 1 document / 88,565 normalized bytes;
- `en.standardebooks.manual`: 2 documents / 84,793 normalized bytes;
- `github:encode/httpx`: 1 document / 8,161 normalized bytes;
- `github:psf/requests`: 1 document / 1,542 normalized bytes.

By stratum: `uk=1 family`, `en=1 family`, `code=2 families`, total `183,061` prebuild normalized bytes.

No post-split family distribution exists because split construction was never reached. The prebuild family counts independently fail DATA-295 minimum-family diversity for UK and EN.

## Verdict

- Independent verifier mechanics: `PASS`.
- Adversarial fixtures: `PASS`.
- Exact DATA-301 cluster-safe split: `BLOCKED / NOT_REACHED`.
- Actual cross-boundary exact/near leakage count: `NOT COMPUTABLE` because no split exists.
- Actual deterministic assignment: `NOT TESTED` because no assignment exists.
- Actual post-split source-family distribution: `NOT AVAILABLE`.
- Corpus frozen/terminal claim: `NOT AUTHORIZED`.

This is a truthful `BLOCKED` audit, not a split PASS.

## Acceptance gate for a successor materialization

Run the verifier against both `wave3-clean-a/corpus-v03` and `wave3-clean-b/corpus-v03` and require:

- zero cross-boundary record-ID reuse;
- zero cross-boundary exact SHA-256 overlap;
- zero cross-boundary dedup-cluster straddles;
- identical canonical assignment SHA-256 across clean roots;
- identical source-family-distribution SHA-256 across clean roots;
- complete source-family and cluster evidence for every manifest record.

Any missing cluster/family evidence is `BLOCKED_INSUFFICIENT_SPLIT_EVIDENCE`; any detected crossing is `FAIL_CLUSTER_SAFE_SPLIT`.

No paid cloud/GPU/training compute was used.
