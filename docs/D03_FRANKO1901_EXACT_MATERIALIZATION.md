# D03 Franko1901 exact materialization

## Purpose

This package opens one current-main, collision-safe Ukrainian natural-text candidate from an already frozen upstream source. It does not create a new corpus architecture and does not grant canonical capacity or training authority.

The exact source is `dmytro-yemelianov/verbacorpus@34a2c10ac35e1febad6c270a88fc8b83790407da`, file `data/sources/franko.csv`, Git blob `45f33ac620907e1d1ed727524975b3a0fd1a0994`, 6,225,761 bytes.

## Why Franko and not a bulk Nomis expansion

Historical NEXT100-027 admitted only a small Nomis1864 subset. The frozen Verba data card and expansion report say the full Nomis pipeline used OCR plus batched Sonnet extraction and retains rare extraction/normalization errors. That makes a mass Nomis expansion unsuitable as a clean-Base shortcut without independent quality review.

Franko1901 has a materially stronger provenance boundary in the same frozen upstream revision:

- the source registry identifies the 1901–1910 work as public domain;
- the data card classifies Franko as an existing digital transcription rather than the OCR/LLM extraction path used for Nomis;
- the data card defines `text` as verbatim source orthography and says it is never modified;
- the adapter takes only `prov_clean.strip()` as canonical source text.

This package therefore admits only the `prov_clean` field to candidate materialization. It deliberately excludes `modern_text`, generated categories, cleaned explanations and variant groups.

## Rights and attribution boundary

The historical Franko source text is treated as public-domain source material under the exact upstream source registry/data-card authority. The Verba compilation/enrichment layer is CC BY 4.0 and the materialization retains the required project attribution:

`Yemelianov, Dmytro (2026). verba — Ukrainian Proverbs Corpus (v1.0.2). https://verbacorpus.org`

This is a source-candidate purpose decision, not a corpus-release or evaluation grant. Evaluation use is not granted here.

## Deterministic materialization

`tools/materialize_d03_franko1901.py`:

1. hard-binds the exact upstream revision, path and Git-blob identity;
2. performs two bounded `Accept-Encoding: identity` acquisitions and requires byte identity;
3. requires strict UTF-8 and the exact required CSV columns;
4. selects only `prov_clean`;
5. applies NFC plus outer-whitespace trim without modernizing historical spelling;
6. rejects empty, replacement-character, control-character, oversized, low-alphabetic, low-Cyrillic, URL-like and email-like payloads;
7. collapses exact normalized-text duplicates;
8. emits deterministic record IDs, exact text hashes and byte counts;
9. emits a self-identified report and exact inventory identity.

The focused/adversarial suite includes a Unicode-local-part email regression found during pre-PR testing. That defect was fixed before publication.

## Truth boundary

The output is `CANDIDATE_MATERIALIZED_ZERO_CREDIT` only.

Until the current canonical D03 owners consume the exact candidate and rerun all required downstream gates:

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
