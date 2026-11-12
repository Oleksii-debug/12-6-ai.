# Plan 2 local release for Plan 9

The accepted data release consists of the Section 16 dataset, the original
frozen Section 15 tokenizer, the physical Section 17 shards and exposure
ledger, and the source notices. The Plan 9 consumer checks the release pin
before yielding the existing Trainer's aligned `input_ids`, `target_ids`
and `loss_mask` tensors. The validation and final-test reservations are
represented by their original authorities and metadata.

Build the physical Section 17 bundle, then stage and verify the local release.
The notice cache is optional. An initial build retrieves exact, pinned
license files, and the verified `notices/` directory can supply offline replay.

```bash
PYTHONPATH=src python -m tools.plan2_s18_physical_release_v1 \
  --root . --bundle "$BUNDLE" --out-dir "$RELEASE"
PYTHONPATH=src python -m tools.plan2_s18_physical_release_v1 \
  --root . --out-dir "$RELEASE" --verify-release
```

Two physical builds verified the same 2,264 inventory members. The release
manifest SHA-256 is
`883f0e3949b7a64a07bcc1236bf7aaf86ed5d84deba4c5aa60d2bfcb83c21ad2`.
The [repository release authority](../evidence/plan2_s18/release_authority_v1.json)
binds the dataset, tokenizer, shards, exposure ledger, notices and Plan 9
consumer as one version.

The real training data contains 1,985,122 next-token targets. The consumer
checked all seven configured scales and ran an aligned loss probe from a
verified physical shard. The Plan 4 reference inference check used the
frozen tokenizer and a tiny fresh decoder. The decoder now supports the
runtime's default replacement behavior for incomplete UTF-8 sequences.
The tokenizer's merge order, vocabulary identity and strict decoding
remain the same. Campaign authorization and public redistribution have
their own approval paths.
