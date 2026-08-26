# NEXT100-102 — converged global cross-source dedup

This successor consumes the exact current `NEXT100-063` source-authority convergence head and reruns the incumbent `NEXT100-065` V3 exact/normalized/near/fragment/code-skeleton and lineage-aware dedup engine over the full numeric convergence vector.

Scope is exactly 22 source objects and 320,632 declared source-capacity bytes before successor dedup: the 11-object terminal V3 inventory plus six KMu Secretariat objects, three bounded NIST Technical Series objects, one bounded MDN prose object, and one bounded Verba/Nomis1864 object. The zero-capacity CPython-docs admission remains recorded by NEXT100-063 but is not fabricated into a byte/object ledger here.

NIST and MDN are the materialization edge cases. MDN is reconstructed with the exact terminal prose-only Markdown normalization and its raw Git blob plus normalized SHA-256 are both pinned. NIST is materialized from PDF: the workflow verifies exact source bytes/SHA-256, binds `pdftotext` 24.02, applies the NEXT100-034 start-page/NFKC/email-redaction/line-normalization/bounded-truncation rules, then requires materialized UTF-8 bytes/SHA-256 to match the terminal seal before the common V3 matcher sees the text.

The report is intentionally fail-closed. A successful exact-head run means global cross-source dedup executed on this exact converged vector and produced a conservative post-dedup source-capacity number. It does not create Research Corpus V1, perform evaluation decontamination, create post-pack loss positions, or authorize tokenizer fitting or 20M training.

Next order after an exact-head green report: rerun balance/diversity on the post-dedup vector, materialize an immutable candidate corpus and split/shard identity, run evaluation decontamination and quality/privacy gates, produce the unique-loss ledger, then requalify tokenizer/training/checkpoint mechanics for the 20M campaign.

`LOCAL_FREE` only. No training, optimizer update, paid compute, external model, or final-test payload access.
