# D03 USGPO bounded zero-credit intake

This package is a source-specific **candidate materializer**, not corpus admission or training
authority. It is pinned to `common-pile/usgpo` revision
`b1685d7dc7e3e71a62ec6777ed5e637b184f5f6c`, shard
`data/00017_usgpo.jsonl.gz` (208,278,377 compressed bytes,
SHA-256 `ae0b8573ce72853590ca7a84a7b2dcf74bd984031694012d2cb6a89309033563`).

The upstream Common Pile USGPO producer is independently bound by Git blob
`073095cd64a9d9d1ddffe8e853720b0eb410988e` at audited code revision
`9457f04a14cb2355ab00023420369d46ffd4a395`. The project rights registry remains
`REVIEW_REQUIRED`; candidate rows remain zero corpus/capacity/loss credit and require item-level
rights review plus every configured downstream gate.

Run, after separately obtaining the exact pinned shard:

```text
python tools/materialize_d03_common_pile_usgpo.py \
  --source <path-to-00017_usgpo.jsonl.gz> \
  --output-dir <empty-or-review-output-dir>
```

The tool verifies exact compressed size and SHA-256 **before** parsing. It scans in source order
under fixed line/decompressed/document/byte ceilings, validates the audited producer schema,
requires exact GovInfo package-bound provenance, keeps only a conservative preregistered federal
collection subset, and rejects contact/secret/low-quality/non-English-like payloads plus exact
normalized duplicates.

Outputs:

- `usgpo_candidates.jsonl`: text-bearing candidates, each still `training_eligible=false`.
- `usgpo_item_review_inventory.jsonl`: no payload text; item URL, collection, hash, byte count,
  and decision/reason for rights/privacy/quality review.
- `usgpo_materialization_report.json`: text-free bounded-run counts and zero-credit truth state.
- `usgpo_materialization_manifest.json`: authenticated transport/artifact hashes and zero-credit
  truth state.

A successful run does **not** set corpus admission, tokenizer-fit authority, optimizer exposure,
training execution, learned weights, evaluation eligibility, paid compute, or final-test access.
