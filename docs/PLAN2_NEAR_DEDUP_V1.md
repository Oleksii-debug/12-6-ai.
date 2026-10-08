# Plan 2 / Section 7 — deterministic near and mirror families

The Section-7 adapter consumes verified Section-6 exact survivors; it never
reimplements exact dedup, source rights, normalization or the incumbent G06
privacy scanner. Each run re-verifies S6 before using the corresponding
physical normalized UTF-8 records.

A versioned conservative matcher detects casefold/whitespace mirrors, numeric
template variants, and longer derivative records using both token-set Jaccard
and five-character-gram Jaccard. Families use **complete-link** admission:
each member must match every other member, avoiding transitive single-link
contamination. Source/record IDs are sorted before clustering, so input order
cannot change decisions. A cap of one retained representative per family is
enforced; family members, relation kinds and suppressed IDs remain auditable.
No record text is copied into the published manifest.

The versioned fixture at configs/data/plan2_near_dedup_audit_v1.json carries
positive and negative pairs, including a declared lexical false negative.
The gate requires no observed false positives and >=75% recall on that bounded
fixture. This is an audit sample, not a universal scientific accuracy claim.

Run:
python tools/plan2_near_dedup_v1.py --root . --out-dir <fresh-dir>

The immutable local candidate is restartable and does **not** grant training,
tokenizer, evaluation, model, or production release authority. No paid compute.
