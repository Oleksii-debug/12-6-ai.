# D03 Common Pile Foodista source-rights review v1

This authority closes one narrow review question for the existing Common Pile Foodista
lineage. The result is **BLOCKED, zero credit**. It does not claim a legal conclusion and it
does not undo the useful bounded materialization work in PR #1174.

## Why the review is blocked

Foodista's own Help page supplies a genuine positive rights signal: it invites copying and
reuse with attribution and says Foodista content is provided under a Creative Commons
attribution license. The pinned Common Pile collector and the exact PR #1174 record contract
further narrow candidate rows to Foodista origins whose metadata says CC BY 3.0.

The same primary Foodista Help page also says the recipe database is developed from both user
submissions **and an automated crawl of the Web**. The current bounded dataset contract has
`authors`, `license`, and `url` metadata but no record-level field that establishes the
original-rights lineage of each externally crawled work. The Common Pile Foodista dataset card
also explicitly warns that license laundering or inaccurate metadata may lead to an incorrect
license assignment for some documents.

A sitewide CC marker therefore cannot, by itself, prove that Foodista had authority to relicense
every third-party work that may have entered through the automated crawl. This project must fail
closed rather than turn that uncertainty into training authority.

## Exact frozen scope

The authority binds:

- main `7c14db5f6e43d369bf859cb5c6803024c929943c`;
- `COMMON-PILE-SOURCE-RIGHTS-V1` blob
  `7b4d6828288672bf25c551e85a5d7f7399e8ef0f`, where `foodista` is still
  `REVIEW_REQUIRED` and zero-credit;
- `r-three/common-pile@9457f04a14cb2355ab00023420369d46ffd4a395` and exact Foodista
  README/preprocess/to-Dolma blobs;
- PR #1174 exact head `0b94ff93fc7860833ed24faa40a7d3e185740547` and config blob
  `9d960fa62ceaa5d2cf1804ccd3f52b5beaef2836`;
- `common-pile/foodista@04d1b7a6562c6d6459426d2a3b88184b8a98f7b6`,
  shard `v0/documents/00000_foodista.jsonl.gz`, SHA-256
  `286b801bc826f161efad892af5529b201f91f42c5a932e3962d74d24037fe087`.

Any product, collector, rights, or provenance drift requires a new policy review.

## Required resolution

Before canonical payload admission, retained records need either:

1. record-level original-rights provenance sufficient to distinguish and qualify the underlying
   source of every retained work; or
2. an equivalent primary-source grant establishing that Foodista possessed sublicensable rights
   for the automated-crawl material in the exact retained scope.

Until then, zero-credit mechanics may continue, but the Foodista lineage cannot contribute
canonical corpus bytes, tokenizer-fit authority, optimized-target exposure, evaluation bytes, or
optimizer updates.

## Validation

Run:

```text
python src/twelve_six/common_pile_foodista_rights.py
pytest -q tests/test_d03_common_pile_foodista_source_rights_v1.py
```

The validator recomputes the checked-in generic registry Git blob identity, requires the exact
`foodista` registry row to remain `REVIEW_REQUIRED` and zero-credit, freezes upstream/product
identities, rejects boolean aliases for integer-zero claims, and refuses any silent transition to
`source_scope_qualified=true`.

## Scientific truth boundary

This package executes no corpus payload and creates no learned weights. It authorizes zero
training bytes and zero optimized-target positions; tokenizer fitting, evaluation eligibility,
optimizer updates, and final-test access remain unauthorized. No paid compute or foreign
pretrained weights are used.
