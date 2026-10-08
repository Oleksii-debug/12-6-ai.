# Plan 2 Section 8 — reserved evaluation firewall (component)

The accepted Plan-2 S1–S7 lineage remains the exclusive source, rights,
normalization, privacy and exact/near-dedup authority. Section 8 introduces
**no new matcher**: it invokes the incumbent DATA-232
`twelve_six.data.decontamination_authority_v2.build_report` and verifies its
hash-only report. This interface works with multiple separately physical,
rights-verified S4/S5 cohorts. It does not manufacture new evaluation rights.

For each whole source, the caller supplies `role`
(`train_candidate`, `selection_validation`, `final_test`), `origin`,
S4 normalization manifest SHA-256 and S5 privacy manifest SHA-256.
`reservation_identity(...)` is only a content hash. The independently
approved `expected_reservation_sha256` MUST be pinned before packing.
Self-consistent rehashing of source roles cannot replace that trust boundary.

All clean records, including near/exact-suppressed records, participate in
DATA-232 cross-boundary exact/near/cluster comparisons. Reserved evaluation
sources never enter the training candidate; matches and contaminated
cross-source training families are excluded. Teacher-generated material is
categorically denied automatic training re-entry. The resulting manifest
includes only record IDs, integrity hashes and DATA-232's text-free matching
report; all three downstream authorization flags remain false.

The checked `stage_firewall` function publishes with immutable no-clobber
atomic writes and supports identical restart/clean rebuild byte parity;
corrupt existing evidence and symlink destinations fail closed. The tests
exercise multiple independent synthetic S4/S5-verified source fixtures;
these are **component qualification**, not a real production
reservation/data-corpus approval or a claim of final evaluation readiness.

Local/free gate:
`PYTHONPATH=src:. pytest -q tests/test_plan2_reserved_firewall_v1.py`

No cloud training, paid compute, tokenizer fitting, evaluation outcomes or
model-selection behavior is performed by this component.
