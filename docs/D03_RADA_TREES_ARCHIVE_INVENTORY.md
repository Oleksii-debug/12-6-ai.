# D03 Rada_Trees archive inventory successor

## Purpose

This package fills the next missing acquisition seam after PR #638. It can bind the exact parent Hugging Face object metadata to a locally acquired `Rada_Trees.7z`, verify an independently established content SHA-256 and byte size, preflight every archive member path/size, extract into an isolated temporary directory, re-check the extracted file set against the pre-extraction listing, hash every regular member, and emit one deterministic member inventory identity.

It deliberately does not decide that any member is training material. Archive discovery, object metadata, content identity, member inventory, plain-text classification, rights/provenance, quality/privacy, lineage deduplication, evaluation decontamination, corpus composition, packing, and optimized-loss accounting remain distinct gates.

## Parent authority

- PR #638 exact parent head: `92c1fd05d4399b0f0c4a35f0689160383f963c9c`.
- Dataset: `uacorpus/Rada_Trees`.
- Observed immutable dataset revision: `1b994a5804dcda122721e8d33a03fd172cf8d867`.
- Parent now contains a metadata-only probe that binds exact Git blob OID, Xet object identity and exact byte size at that revision without streaming either archive.
- Primary archive candidate: `Rada_Trees.7z`.
- `rada_xtag_texts.7z` remains held as annotation/derived material and receives zero credit.

The committed successor config still does not invent a full-content SHA-256 or claim a production archive download. A real execution must obtain the primary archive from the exact pinned revision and bind an expected content SHA-256 before this layer is treated as exact inventory evidence.

## Execution

A real run requires the exact archive, expected SHA-256, exact expected byte count, and the immutable upstream object identity from the parent metadata probe:

```text
python tools/inventory_d03_rada_trees_archive.py \
  --archive /path/to/Rada_Trees.7z \
  --expected-sha256 <64-lowercase-hex> \
  --expected-size <exact-bytes> \
  --upstream-object-identity <immutable-xet-or-lfs-id> \
  --output evidence/d03-rada-trees/archive-inventory-v1.json
```

The implementation uses `7zz` or `7z` only as the archive codec. It first hashes and sizes the transport file, then parses a technical listing and rejects traversal/absolute paths, NFKC path collisions, over-large members, and over-large total expansion before extraction. After extraction it rejects symlinks/special files, requires exact listing-to-file-set and size agreement, and computes SHA-256 for every member.

`inventory_identity_sha256` is derived from dataset head, archive path, upstream object identity, exact archive hash/size, and the sorted member path/size/hash vector. Extractor-version prose is not part of that data identity.

## Truth boundary

A successful run means only `EXACT_ARCHIVE_AND_MEMBER_INVENTORY_MATERIALIZED_CLASSIFICATION_NOT_RUN`.

It still authorizes exactly zero training bytes and zero optimizer updates. It does not claim plain-text selection, member-level attribution, privacy/quality pass, lineage independence, dedup/decontamination pass, Research Corpus V1 admission, tokenizer fit, learned-20M training, or paid compute.

The immediate successor is member-level classification and provenance binding, with original plain-text transcripts separated from annotations/derivatives before any capacity proposal.
