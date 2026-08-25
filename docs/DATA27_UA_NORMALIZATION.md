# DATA-27 Ukrainian normalization

Normalization identity: `12-6.ua-normalization-v1`.

The audited incumbent D03 path used Unicode NFKC and collapsed each natural-text line with `split()/join()`. DATA-10 additionally applied NFKC to code while preserving indentation. Both behaviors are too broad for a corpus path that must retain orthographic and source distinctions.

## Allowed natural-text transformations

- Decode strict UTF-8 only; reject U+FFFD and surrogate code points.
- Canonicalize CRLF/CR line endings to LF.
- Remove a leading decoded BOM.
- Decode HTML entities, convert obvious residual block-break tags to line breaks, and remove residual HTML tags conservatively.
- Convert NBSP and narrow NBSP to ordinary space.
- Remove soft hyphen U+00AD.
- Apply Unicode NFC canonical normalization.
- Remove trailing horizontal whitespace and outer blank lines only.

The normalizer does not globally lowercase, transliterate, compatibility-fold, or collapse internal ordinary spaces/newlines.

## Preserved distinctions

Common Ukrainian apostrophe representations ASCII `'`, U+2019 `’`, and U+02BC `ʼ` remain distinct. `ґ` is never folded to `г`. `і`, `ї`, and `й` remain distinct letters. NFC may compose canonically equivalent decomposed sequences, e.g. `і` plus combining diaeresis to `ї`, without compatibility folding. Quote styles and dash styles are preserved. Compatibility characters such as `①` are preserved, unlike NFKC.

## Code modality

Code bypasses natural-text normalization. The code path only canonicalizes CRLF/CR to LF. It does not apply NFC/NFKC, entity decoding, HTML stripping, NBSP conversion, soft-hyphen removal, indentation trimming, repeated-space collapse, or newline stripping.

## Provenance and fingerprints

Each `NormalizationResult` contains a `NormalizationTrace` with raw and normalized SHA-256 fingerprints, raw and normalized codepoint/UTF-8-byte counts, current byte-token delta, reason counters, modality, source ID/version, raw document ID, and raw source SHA-256 when available. D03 packaged records retain the trace and corpus identity binds the normalization schema and raw/normalized document fingerprints. DATA-10 multilingual admitted records retain the same trace.

## Evidence

`tests/fixtures/ukrainian_normalization_regression_v1.jsonl` covers apostrophe variants, ґ/г, і/ї/й, combining marks, NBSP, quote/dash distinctions, soft hyphen, line endings, HTML residue, Unicode compatibility characters, and code-layout isolation.

`docs/evidence/DATA27_UA_NORMALIZATION_AUDIT.json` records the current S0 Ukrainian sample audit: 6 project-authored Ukrainian documents, 811 codepoints and 1,515 current byte-token baseline units before and after normalization, with zero changed documents.

`docs/evidence/DATA27_UA_REAL_SAMPLE_AUDIT.json` records a separate read-only audit over six text segments from the real Ukrainian Verkhovna Rada source currently being exercised by DATA-21. The concatenated sample is 1,502 codepoints / 2,788 current byte-token units before and after DATA-27 normalization, with zero changes. DATA-27 does not promote that source, copy DATA-21's rights decision, or claim that six segments characterize the full source.

`tools/audit_ukrainian_normalization.py` reproduces document-level fingerprints and aggregate codepoint/current-byte-token changes for JSONL inputs without modifying source bytes. The DATA-27 workflow runs the regression/integration tests, reproduces the S0 audit, and uploads exact evidence artifacts.

## Truth boundary

This change validates and hardens normalization. It does not establish language-identification quality, external-source rights, external-corpus representativeness, or future BPE/Unigram token deltas. The real-source excerpt audit is normalization evidence only; canonical source acquisition and promotion remain DATA-21/D03 responsibilities.
