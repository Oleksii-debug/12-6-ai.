# Plan 2 / Section 7 — near/global duplicate and mirror-family candidate

Scope: Section 7 only. Reuses accepted S3 physical cohort, S4 normalized
record boundaries, S5 incumbent G06 privacy execution and S6 exact-global
dedup. This is NOT an admission or release gate.

Versioned configs/data/plan2_near_policy_v1.json implements Unicode
NFKC/casefold word-trigram similarity. A pair is near/mirror/derivative
when Jaccard >= 0.82 or containment >= 0.94 with size ratio >= 0.50.
Too-short records are excluded from near matching (not from exact dedup).
Disjoint shingle sets cannot match. Candidate pair resource ceilings fail
closed instead of silently dropping comparisons. The bounded LOCAL_FREE
algorithm qualifies the current cohort; production-scale capacity is NOT asserted.

Transitive components select a lexicographic representative with cap one.
All matched edges, family members and provenance remain in the receipt.
The full S6 exact receipt is reconstructed from physical input and checked
before matching. S5 tombstoned rows cannot return, and S6 exact exclusion
remains authoritative. Outputs contain IDs, hashes and counts, never text.

Versioned synthetic tradeoff controls live in
configs/data/plan2_near_audit_samples_v1.json. Both positive and negative
controls must pass; false-positive or false-negative regressions fail closed.
Policy hash, audit hash, exact-parent hash and receipt hash bind publication.
Changing policy/samples requires explicit versioning and requalification.

Run: PYTHONPATH=src:. python -m tools.plan2_near_dedup_v1 --root . --out-dir /tmp/plan2-s7
Test: PYTHONPATH=src:. python -m pytest -q tests/test_plan2_near_dedup_v1.py

Repeat same destination and clean rebuild; manifests must be byte-equal.
Corrupt manifests and symlink destinations fail closed.

No training, tokenizer fit, reserved evaluation, paid compute or
whole-product authority is granted. S8 evaluation firewall is subsequent.
