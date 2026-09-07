# D03 Rada_Trees acquisition probe

## Decision

`HIGH_LEVERAGE_SOURCE_DISCOVERED_EXACT_ACQUISITION_REQUIRED`

This package opens a new Ukrainian data-acquisition lane for Research Corpus V1 without granting corpus or training credit from discovery metadata.

## Why this source matters

The public `uacorpus/Rada_Trees` dataset card describes Ukrainian Verkhovna Rada plenary-session transcripts covering 1990–2024 and approximately 88 million tokens. The repository reports plain-text original transcripts plus UD and `nlp_uk` annotated forms, under CC BY 4.0. The visible repository tree is approximately 1.23 GB and contains two large archives.

The observed dataset history link resolves the current visible head to `1b994a5804dcda122721e8d33a03fd172cf8d867`.

None of those observations is treated as exact training capacity. The large archives still need immutable object identity, member-level inventory and purpose-specific review.

## Lineage rule

The proposed family is `ua.rada.plenary-transcripts.1990-2024`. It is provisionally distinct from the incumbent `ua.rada.open-data.laws-texts` family because plenary-session transcripts and primary-law texts are different document lineages.

That distinction is not terminal until archive members are inspected. All representations or mirrors of the same underlying plenary transcript must collapse to one family lineage. In particular, UD annotation, `nlp_uk` annotation, ParlaMint-UA overlap, GRAC parliamentary derivatives and other mirrors do not earn independent family credit merely because their formatting or hosting differs.

For pretraining acquisition, the preferred payload is the original plain-text transcript layer. Annotated derivatives remain zero-credit by default unless a separate authority proves a distinct justified purpose.

## Required executable successor

1. Pin the exact Hugging Face dataset head and exact Xet/LFS object identity for the primary archive.
2. Download the primary archive and verify SHA-256.
3. Emit a deterministic archive-member inventory with normalized safe paths, byte sizes and SHA-256 for every member.
4. Classify members into original plain text versus annotation/derivative layers before any capacity arithmetic.
5. Bind attribution and member-level provenance.
6. Run Ukrainian-language, quality and privacy filters.
7. Run exact and near-lineage dedup against the incumbent Rada laws family, ParlaMint-UA, GRAC and the live composed corpus.
8. Run evaluation decontamination.
9. Recompute family-cap and Ukrainian mixture feasibility.
10. Only a later terminal authority may propose nonzero source-capacity credit.

## Truth boundary

No archive has been downloaded by this package. No archive SHA-256 or member inventory is claimed. Approximate token count and displayed repository sizes are discovery evidence only. Training-authorized bytes remain exactly zero. No tokenizer fit, model training, optimizer update, evaluation/final-test use, paid compute or learned-model claim is authorized.
