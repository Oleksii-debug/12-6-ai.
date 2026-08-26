# D03 Rada_Trees primary archive materialization and inventory

This successor consumes the immutable Hugging Face/Xet object snapshot produced by
`tools/probe_d03_rada_trees_hf_objects.py` and advances one step only:

`exact object pin -> exact primary archive bytes -> content SHA-256 seal -> safe metadata inventory`.

It deliberately stops before member extraction, member-content hashing, plain-text
classification, quality/privacy, deduplication, decontamination, corpus promotion,
tokenizer fitting, or model training.

## Why this exists

The `Rada_Trees.7z` archive is hundreds of megabytes. CI must not download it merely
because a pull request exists. The tool therefore requires an explicit `--download`
flag before network acquisition. Without that flag it only accepts an already present
local archive whose byte count matches the immutable object-pin snapshot.

The download URL is bound to the exact dataset commit, never `main`. Redirects must
remain HTTPS, the stream is capped at the exact pinned byte count, and the full file is
hashed while it is written to an atomic staging path.

## Archive inventory boundary

The archive is listed with 7z/7zz technical output and is not extracted. The inventory
fails closed on:

- absolute, parent-traversal, Windows-style, drive-prefixed, empty, or case-colliding paths;
- encrypted files;
- symbolic/hard-link metadata;
- single-member size above the parent D03 limit;
- total uncompressed size above the parent D03 limit;
- excessive member count;
- missing/malformed file sizes.

The report contains only member paths, sizes, optional CRC metadata, and deterministic
identities. It does not claim member SHA-256 because member bytes have not been
extracted or streamed yet.

## Operator sequence

1. Produce and validate the exact object-pin snapshot from PR #638.
2. Ensure `7z` or `7zz` is installed locally.
3. Run:

   `python tools/materialize_d03_rada_trees_primary_archive.py --archive <local-path> --download`

4. Preserve the emitted archive content SHA-256, member-inventory identity, and report
   identity as the input to the member-hashing/plain-text-classification successor.

If the archive has already been acquired by an approved process, omit `--download`;
the tool will hash and inventory that exact local file instead.

## Truth boundary

`training_authorized_bytes` remains exactly zero. This layer does not extract data,
does not inspect final-test material, does not fit a tokenizer, does not execute an
optimizer update, and does not authorize paid compute.

The next gate is member-content SHA-256 plus classification of original plain-text
transcript members, followed by attribution/provenance, language/quality/privacy,
lineage-aware deduplication against Rada/ParlaMint/GRAC/live corpus, evaluation
decontamination, mixture/family-cap recomputation, deterministic split/packing, and
unique causal-loss accounting.
