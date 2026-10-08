# Plan 2 Section 10 — cluster-safe candidate split (LOCAL_FREE)

## Canonical reuse and boundaries

S10 consumes the immutable S9 mixture candidate and the existing S7 complete-link near/mirror family manifest, reproduced by physical S3–S9 pipeline. It does not implement another dedup, source-registry, rights, eval-firewall or training authority. It follows existing DATA-36 cluster-hash principle, extending the Plan-2 component to train/validation/test without calling an earlier training-eligible-only DATA-36 API on a candidate explicitly NOT authorized for training.

S9 selected record identities and immutable SHA-256 identity are included in the split manifest, alongside S7 and S8 identities. Only S7 retained records are eligible. Excluded/held-out S9 records cannot re-enter. Every S7 near-family is assigned atomically (S7 retained representative selected, suppressed aliases never enter); others are singleton clusters. Under S7 near-family cap each family contributes at most one representative. Source-family is a categorization, not itself a semantic duplicate cluster, because this single-source fixture has distinct records.

## Deterministic policy and manifest

The Git-pinned policy path is configs/data/plan2_cluster_split_policy_v1.json, schema 12-6.plan2-cluster-split-policy.v1 with fixed 64-hex seed, 10% validation clusters and 10% test clusters (rounded with minimum one cluster each). Cluster SHA ranks, not caller iteration order, select test and validation clusters; others train. Fewer than three independent clusters fail closed. Emitted immutable canonical JSON includes split_manifest_sha256. Three partitions are pairwise disjoint and cover exactly the selected S9 candidate IDs. Separately bound S9 dataset, S7 near-family evidence, S8 decontamination, policy revision, seed and record-to-cluster identities.

The test partition is an INTERNAL CANDIDATE split, NOT the independent reserved final-test set from S8, and NOT proof of legally admissible production final-test access. Flags training_corpus_authorized, tokenizer_fit_authorized, production_test_release_authorized and real_final_test_material_accessed stay false. No real tokenizer counts or paid compute claimed.

## Qualification

- Ruff check: tools/plan2_cluster_split_v1.py and tests/test_plan2_cluster_split_v1.py.
- PYTHONPATH=src:. pytest -q tests/test_plan2_cluster_split_v1.py tests/test_plan2_corpus_mixture_v1.py tests/test_plan2_near_dedup_v1.py
- PYTHONPATH=src:. python tools/plan2_cluster_split_v1.py --out-dir DIR; repeat exact same destination and clean rebuild in a second directory, byte-compare emitted CLI receipt and cluster-split-manifest.json.
- Canonical workflow .github/workflows/plan2-data-component.yml retains S1–S9 and adds S10 gates.

Negative tests cover forged S9 and S7 identities, foreign/duplicate/overlap selection, forged/rehashed near families, widened training authority, bad fraction/duplicate-key policy, and self-rehashed split assignment drift. Immutable restart and symlink rejection reuse Plan-2 physical publication.

This is component-level LOCAL_FREE qualification, not production corpus balancing, Plan9/10 release, or true separate final-test custody.
