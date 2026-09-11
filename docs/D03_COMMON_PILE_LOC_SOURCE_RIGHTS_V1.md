# D03 Common Pile Library of Congress source-rights qualification v1

This package resolves the **source-policy review only** for the exact Common Pile
`library_of_congress` lineage. It is not a legal opinion and it does not admit a payload,
credit corpus capacity, authorize tokenizer fitting, authorize training, or expose final-test
data.

## Exact authority

The policy is bound to:

- mainline Common Pile source-rights registry blob
  `7b4d6828288672bf25c551e85a5d7f7399e8ef0f`, where `library_of_congress` remains
  `REVIEW_REQUIRED`;
- audited `r-three/common-pile` revision
  `9457f04a14cb2355ab00023420369d46ffd4a395`;
- collector, metadata collector, and source README blobs frozen in
  `configs/data/d03_common_pile_loc_source_rights_v1.json`;
- the exact dataset/shard contract first carried by PR #1135;
- record-level `loc_books` / `Public Domain` / `english` / LCCN-to-LoC-item /
  `tile.loc.gov|tiles.loc.gov` lineage.

The project decision is `CONDITIONAL_SOURCE_ADMISSION`: a downstream payload may be
considered only if every exact record-contract check is satisfied. Collector, revision, rights,
language, item-URL, or OCR-host drift fails closed.

## Primary evidence

The frozen evidence is the Library of Congress Selected Digitized Books Rights and Access
page plus the official English-filtered collection endpoint used by the audited metadata
collector. The evidence text in the policy is a project-normalized statement of what those
sources support; it is not a license substitution and cannot be replaced by package-level
Hugging Face metadata.

## Validation

Run:

```text
python tools/validate_d03_common_pile_loc_source_rights_v1.py
pytest -q tests/test_d03_common_pile_loc_source_rights_v1.py
```

The validator independently freezes field sets, evidence facts, upstream identities, dataset
identity, record contract, decision semantics, and zero-credit truth values. It also recomputes
Git's blob SHA-1 for the checked-in generic registry and verifies the exact LoC row remains
non-authorizing.

## Scientific truth boundary

This package executes no LoC payload and creates no training evidence. On merge:

- `payload_source_admission_executed = false`;
- `source_capacity_bytes_credited = 0`;
- `training_authorized_bytes = 0`;
- `authorized_optimized_target_exposure = 0`;
- `model_training_executed = false`;
- `optimizer_updates = 0`;
- `final_test_payload_accessed = false`;
- `paid_compute_used = false`;
- `foreign_pretrained_weights_used = false`.

A repaired/materialized LoC candidate must still pass canonical materialization, global dedup,
reserved-evaluation decontamination, quality/privacy, balance/family caps, cluster-safe split,
deterministic packing/two-clean-build, and positive unique-loss accounting before it can
contribute to training authorization.
