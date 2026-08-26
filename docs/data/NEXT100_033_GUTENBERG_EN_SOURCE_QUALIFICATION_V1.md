# NEXT100-033 — bounded English Project Gutenberg source qualification

Worker: `NEXT100-033-DATA-EN-GUTENBERG`

## Scope and terminal boundary

This authority evaluates exactly three English Project Gutenberg ebook editions for model-training intake. It does not admit Project Gutenberg as a whole, does not authorize evaluation use, does not claim worldwide public-domain status, and does not claim broad English representativeness.

All three admitted titles count as one independent source family: `en.project-gutenberg.public-domain-books`. GITenberg is used only as an immutable transport mirror and earns no additional family credit. Copies or mirrors of the same Project Gutenberg edition must remain in the same lineage for dedup/family accounting.

## Rights and trademark treatment

The source-specific rights evidence is bound at `data/external/rights-evidence/next100_033/project_gutenberg_rights_20260826.json`. The three selected Project Gutenberg catalog records were reviewed as public domain in the USA, while permission-only/copyrighted Project Gutenberg items are explicitly outside this authority. The selected authors died in 1897, 1932, and 1945, so the ordinary Ukrainian author-life-plus-70-years horizon had elapsed before the 2026 review.

Project Gutenberg's trademark and redistribution terms are kept separate from copyright status. The training normalization removes the Project Gutenberg START/END envelope and all text outside it so the normalized body is not redistributed as a branded Project Gutenberg-tm electronic work. Provenance metadata remains out-of-band. No right to use the Project Gutenberg trademark for branding, endorsement, or advertising is asserted.

Redistribution is authorized only for rights-cleared jurisdictions represented by the recorded review. This authority does not infer a universal worldwide public-domain result from a US catalog status.

## Exact bounded editions

- Ebook 37177 — Simon Christian Hammer, `Ludvig Holberg, The Founder of Norwegian Literature and an Oxford Student`; immutable transport commit `6ba1ee491cfba2aa56729967b48886bf71b20ac7`, path `37177.txt`, Git blob `cc35bab195c6d55b14568cd3da8fecfd0499f868`, 87,742 raw bytes.
- Ebook 37985 — Reynold Alleyne Nicholson, `A Literary History of the Arabs`; immutable transport commit `b8477090720ab89858bfed937f799760cb4f23e3`, path `37985-0.txt`, Git blob `cd62e73adf6d502823c4b59de304212cc3d63361`, 1,203,657 raw bytes.
- Ebook 40652 — Ebenezer Cobham Brewer, `A Guide to the Scientific Knowledge of Things Familiar`; immutable transport commit `4d69c9e1daad0951f2baecbae2709e72d8d7d53f`, path `40652-8.txt`, Git blob `52c517db02e5e5a7f5bfef806cf98dd8c92e06e1`, 482,371 raw bytes.

## Preregistered deterministic normalization

`NEXT100_033_PG_BODY_NFC_LF_V1` is frozen before materialization. It decodes with the exact source-specific encoding, canonicalizes line endings, requires exactly one canonical Project Gutenberg START marker and one later END marker, discards those markers plus all content outside them, and preserves every character inside the body except a leading BOM and edge-only blank lines. The retained body is NFC-normalized and emitted as UTF-8/LF with exactly one final newline.

No chapter parsing, generic boilerplate regex deletion, spelling correction, dehyphenation, modernization, OCR repair, semantic cleanup, or random sampling is permitted. Producer credits, transcriber notes, title pages, footnotes, illustrations-as-text, and edition-specific matter inside the Gutenberg markers remain training data.

## Quality, dedup, and evaluation firewall

Each work must exceed 50,000 normalized UTF-8 bytes, contain no replacement or NUL characters, pass a conservative English stopword screen, and have at least 75% ASCII letters among alphabetic characters. Exact normalized hashes must be unique both inside this candidate and against the live DATA-287 registry. Pairwise 8-token shingle Jaccard must not exceed 0.05.

The gate scans repository paths associated with evaluation, reserved, selection, or validation material for exact ebook IDs 37177, 37985, and 40652; any match fails closed. This is only an exact source-ID firewall. Downstream corpus construction must still perform the project's content-hash/decontamination gate, and this source authority grants no evaluation permission.

## Machine evidence

`tools/qualify_next100_033_gutenberg_en.py` performs exact immutable acquisition, Git blob verification, deterministic normalization, quality screening, registry dedup, pairwise near-dedup, and evaluation-reservation checks using Python standard library only. `.github/workflows/next100_033_gutenberg_en.yml` independently materializes the bounded corpus twice and requires byte-identical output. Once an `ADMIT` report is committed, the exact-head workflow also requires byte equality between the committed report and a fresh realization.

LOCAL_FREE only. No model training, paid compute, pretrained weights, corpus freeze, evaluation scoring, or stage promotion is performed by this authority.
