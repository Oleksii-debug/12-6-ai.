# Plan 2 Section 16 source admission checkpoint

Section 16 remains **OPEN**. This checkpoint verifies only the five physical
S15 source identities, their source-level license evidence, G06 privacy
execution and DATA232 comparison with the pinned synthetic evaluation fixture.
It does not admit a production S3–S9 corpus.

`tools/plan2_s16_source_rights_privacy_v1.py` rereads the committed source
bytes and existing S1/S2/S15/G06/DATA232 authorities. It binds their source,
rights, privacy and decontamination identities into an immutable, text-free
receipt. The receipt explicitly denies training, tokenizer fit, release and
Plan9 binding. Changed or missing source and license bytes fail before
publication. Existing output can only be replayed byte-for-byte; a changed
output is refused. The existing Plan2 workflow runs the positive, negative,
restart and clean-rebuild checks without creating another workflow.

The physical cohort still has only three source families: one English
Gutenberg family and two Ukrainian documentation families. The English books
are independent documents within one source family. There are no independently
admitted Ukrainian and English train, validation and test cohorts. The
existing DATA232 comparison uses a synthetic evaluation fixture and does not
establish real final-test custody or cross-language semantic deduplication.
UA-GEC remains a candidate with unresolved contributor, translation, privacy
and evaluation rights; its CC BY repository label is not a corpus grant.

To close Section 16, independently verify actual source-specific copyright,
license and attribution terms; execute privacy and exact/near decontamination
over the proposed physical UA/EN cohorts; reserve non-overlapping family-safe
validation and test sources for both languages; approve and measure the
production language balance; then pass the native S3–S9 admission, negative
tests and exact-head CI on the canonical PR. Candidate-only receipts must
never be renamed or re-signed as production admission.
