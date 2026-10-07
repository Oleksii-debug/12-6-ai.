# D03 Franko1901 exact materialization

## Purpose

This package opens one current-main, collision-safe Ukrainian natural-text candidate from a frozen public-domain digital transcription. It does not create a new corpus architecture and does not grant canonical capacity or training authority.

The primary exact source is `MurzikVasilyevich/ukr-proverbs-franko@7f62a9d8f0673d325b0a565d508461f4b44dae5b`, file `franko.csv`, Git blob `45f33ac620907e1d1ed727524975b3a0fd1a0994`, 6,225,761 bytes. That commit records the transformation as reducing the dataset to clear proverbs. The repository license is exact CC0 1.0, Git blob `0e259d42c996742e9e3cba14c677129b2c1b6311`.

A separate frozen authority, `dmytro-yemelianov/verbacorpus@34a2c10ac35e1febad6c270a88fc8b83790407da`, vendors the exact same source blob and is used only to corroborate source semantics: its data card classifies Franko as an existing digital transcription, defines source text as verbatim source orthography that is never modified, and its adapter selects `prov_clean.strip()`.

## Why Franko and not a bulk Nomis expansion

Historical NEXT100-027 admitted only a small Nomis1864 subset. The frozen Verba report says the full Nomis path used OCR plus batched Sonnet extraction and retains rare extraction/normalization errors. A mass Nomis expansion would therefore be a poor clean-Base shortcut without independent quality review.

Franko has the stronger path: historical public-domain work, exact CC0 machine-use transcription, byte-identical corroborating vendored snapshot, and no LLM-generated field in the admitted payload path.

Only `prov_clean` may become candidate text. `modern_text`, generated categories, cleaned explanations and variant groups are forbidden.

## Rights and provenance boundary

- historical Franko source text: public domain;
- exact primary machine-use dataset: CC0 1.0;
- Verba compilation/enrichment layer: CC BY 4.0, corroborating only and not used as payload authority;
- evaluation: not granted by this package;
- model-training purpose: candidate only, still subject to every project corpus gate.

The output retains provenance for Ivan Franko, the exact CC0 transcription dataset, and the frozen Verba corroboration.

## Deterministic materialization

`tools/materialize_d03_franko1901.py`:

1. hard-binds the exact primary repository, commit, raw URL, source blob, byte count and CC0 license blob;
2. hard-binds the exact corroborating Verba revision/blob/adapter/data-card/source-registry identities;
3. locks the complete acquisition and filter policy so thresholds cannot be silently weakened;
4. performs two bounded `Accept-Encoding: identity` acquisitions and requires byte identity;
5. requires strict UTF-8 and the exact CSV columns;
6. selects only `prov_clean`;
7. applies NFC plus outer trim without modernizing historical spelling;
8. rejects empty, replacement-character, control-character, oversized, low-alphabetic, low-Cyrillic, URL-like and email-like payloads;
9. collapses exact normalized-text duplicates;
10. emits deterministic record IDs, text hashes, byte counts, inventory identity and self-identified report.

Pre-execution focused/adversarial validation is 12/12 PASS. The suite includes a Unicode-local-part email regression discovered and fixed before publication and a mutation proving the Cyrillic threshold cannot be lowered without failing closed.

## Truth boundary

The output is `CANDIDATE_MATERIALIZED_ZERO_CREDIT` only. Until current canonical D03 owners consume the terminal exact artifact and rerun the required downstream gates:

- canonical capacity credit = 0;
- family credit = false;
- corpus admitted = false;
- current global dedup = NOT RUN;
- fresh reserved-evaluation decontamination = NOT RUN;
- post-composition quality/privacy = NOT RUN;
- balance/family-cap evaluation for this addition = NOT RUN;
- tokenizer fit = unauthorized;
- training-authorized bytes = 0;
- authorized unique loss positions = 0;
- optimizer/model training = 0/false;
- final-test access = false;
- paid compute = false;
- learned-20M promotion = false.

## Successor handoff

After terminal exact materialization evidence, the current source-registry/global-dedup owner should consume the exact report, candidate payload identity and record inventory. It must not import the candidate by URL or descriptive row count alone. Fresh global exact/near/fragment/lineage dedup and all downstream corpus gates remain mandatory before any byte can become training exposure.
