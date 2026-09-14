# D03 Rada_Trees Hugging Face object pinning

This successor step closes the metadata-provenance gap in the initial Rada_Trees discovery probe without downloading the 1.23 GB dataset and without granting corpus or training credit.

## Problem

The discovery probe binds the dataset name and immutable revision `1b994a5804dcda122721e8d33a03fd172cf8d867`, but its two large archive rows still contain `exact_blob_identity = null`. Displayed sizes such as `536 MB` and `698 MB` are UI metadata, not immutable object identities.

For Xet-backed Hub files, the repository tree exposes the Git blob OID and Xet object hash. Hugging Face also documents that a non-followed `resolve` request returns `X-Xet-Hash`. The probe requires those independently exposed identities to agree for each archive.

## Implementation

`tools/probe_d03_rada_trees_hf_objects.py`:

1. validates the existing D03 Rada_Trees parent probe;
2. requests the Hub dataset tree at the exact 40-hex revision, never `main`;
3. requires exactly one `Rada_Trees.7z` and one `rada_xtag_texts.7z` entry;
4. binds each exact byte size, 40-hex Git blob OID and 64-hex Xet hash;
5. performs a non-following `resolve` request and requires `X-Xet-Hash` to equal the tree Xet hash;
6. refuses a direct `200` resolve response rather than accidentally streaming a large archive;
7. emits a deterministic self-hashed JSON snapshot;
8. keeps training-authorized bytes, optimizer updates and paid-compute use at zero.

The command also accepts `--tree-json` and `--resolve-json` for deterministic offline replay/testing.

## Deliberate boundary

A green object-pin snapshot proves repository/Xet object identity only. It does **not** prove archive content SHA-256, archive-member paths, member hashes, plain-text-vs-annotation classification, quality/privacy, lineage deduplication, evaluation decontamination, family-cap compliance, corpus identity, tokenizer eligibility or model-training readiness.

The next data step is to download the **primary plain-text candidate archive only**, verify full-content SHA-256, inventory members with traversal/symlink/size defenses, and classify the actual member graph before any nonzero source-capacity proposal.

No new GitHub Actions workflow is added. The repository's shared CI remains the execution surface so this data lane does not add more workflow fan-out.
