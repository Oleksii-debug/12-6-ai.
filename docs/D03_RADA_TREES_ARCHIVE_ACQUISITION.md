# D03 Rada_Trees bounded primary-archive acquisition

## Mission

This is the next distinct successor after the metadata-only Hugging Face object-pinning layer in PR #638.

It consumes the exact object snapshot from `probe_d03_rada_trees_hf_objects.py`, re-checks the primary archive Xet identity immediately before download, downloads only `Rada_Trees.7z`, computes the full archive SHA-256 while streaming, and inventories 7z member metadata without extracting member bodies.

## Safety and determinism

The downloader is bounded by the exact byte size from the pinned object snapshot. It fails if the stream is shorter or longer, and writes through a temporary `.part` file before atomic replacement.

The archive inventory uses `7z l -slt -ba`, never extraction. Every member path is NFC-normalized and rejected if it is absolute, drive-prefixed, contains `..`, NUL/control characters, duplicate normalized names, or case-fold collisions. Link-like and encrypted entries are rejected. The parent probe's existing 50 MB per-member and 10 GB total-uncompressed limits are enforced.

Member classification is deliberately conservative:
- `.txt` / `.text` is only `PLAIN_TEXT_CANDIDATE_EXTENSION_ONLY`;
- common structured/annotation extensions are held;
- everything else remains unclassified.

No extension label is corpus admission.

## Evidence boundary

A valid report proves:
- exact primary Xet object was re-checked;
- exact compressed byte count was downloaded;
- full archive content SHA-256 was computed;
- archive member path/size metadata passed bounded safe-inventory checks.

It explicitly does **not** prove member-content SHA-256, language/quality/privacy, original-transcript provenance, family independence, lineage dedup, evaluation decontamination, normalized capacity or training exposure.

`training_authorized_bytes` remains exactly `0`.

## Verification

Network-free regression tests:

`python -m unittest tests/test_d03_rada_trees_archive_acquisition.py`

Live acquisition requires the object snapshot from #638 and `7z` or `7zz` installed:

`python tools/acquire_d03_rada_trees_primary_archive.py acquire`

No dedicated Actions workflow is added. The repository is runner-saturated and the shared workflow-budget policy should not be bypassed.
