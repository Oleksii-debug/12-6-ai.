# DATA-32 document quality filter

## Scope

DATA-32 adds a lightweight deterministic document-quality decision to the incumbent D03 record-policy seam. It does not decide source rights, acquisition permission, PII, copyright, benchmark contamination, or language admission. Those remain separate D03 hooks and gates.

The implementation is `src/twelve_six/data/document_quality.py`. A quality decision is adapted to the existing `PolicyHookEvidence` as `hook_id=document_quality`; it never creates or modifies the rights, language, PII, or copyright result.

## Incumbent audit

The D03 lineage already had three relevant layers:

- the early S0 pipeline used coarse minimum/maximum length, control-character checks, PII checks and a natural-text alpha-ratio heuristic;
- the D03 corpus foundation introduced the correct independent `quality`, `language`, `pii`, and `copyright` hooks, with source rights explicitly outside those hooks;
- the pretraining factory retained simple length/alpha checks while DATA-10 added strict UTF-8, replacement-character rejection, modality-aware normalization and UK/EN/code admission.

The residual defect was therefore not a missing policy framework. It was the absence of an interpretable, modality-aware quality implementation behind the existing quality hook. In particular, natural-text alpha-ratio logic is not a valid generic gate for source code, and PII is not a quality feature.

DATA-32 does not create a second rights registry, LID system, dedup engine, or black-box quality model.

## Features and conservative policy

Policy identity is the canonical SHA-256 of every threshold and switch. Current policy ID is `d03-lightweight-uk-en-code-v1`; policy SHA-256 is `97b9fe1452b22c6275a27f85524f670253a7f4012377361c4cb007004aeccd1d`.

The filter records:

- character count, UTF-8 byte count, U+FFFD, surrogate and disallowed-control counts;
- Latin, Cyrillic and other-script letter counts;
- punctuation/symbol ratio;
- repeated non-empty line ratio;
- URL count and URL-character density;
- template-line and boilerplate-line density from fixed visible marker lists;
- token count, unique-token ratio and dominant-token ratio;
- code identifiers, keyword hits, indentation and an explicit six-part code-structure score.

Natural-text thresholds are intentionally permissive: minimum 60 characters, symbol ratio at most 0.40, repeated-line ratio at most 0.60, URL-character ratio at most 0.25, template/boilerplate line ratio at most 0.50, diversity checks only from 30 tokens onward, and at most 0.20 other-script letters. UK versus EN identity is not re-decided here; D03/DATA-10 LID owns that question.

Code has a separate policy: minimum 30 characters, symbol ratio at most 0.78, repeated-line ratio at most 0.75, URL-character ratio at most 0.45, diversity checks only from 20 identifiers/tokens onward, and code-structure score at least 2. This prevents the old natural-text alpha heuristic from deleting legitimate structured code.

The integer score is diagnostic only: 100 minus 25 per named reject reason and 5 per named warning. Admission is based on explicit threshold reasons, not on an opaque learned score.

## Calibration

`data/quality/calibration_uk_en_code_v1.jsonl` contains 30 project-authored labeled samples: 10 Ukrainian, 10 English and 10 code; each stratum has five ACCEPT and five REJECT labels. Reject examples target short fragments, line repetition, template density, URL density, low diversity/dominant tokens, symbol-dominated fragments and missing code structure.

Retained calibration evidence is `reports/d03/document_quality_calibration_20260825.json`.

Measured result on this bounded authored set:

- samples: 30;
- correct: 30;
- false accepts: 0;
- false rejects: 0;
- accuracy: 1.0 overall and 1.0 in each UK/EN/code stratum.

This is calibration evidence only. It is deliberately not presented as a general-corpus precision/recall estimate.

The deterministic edge selector chooses the smallest absolute threshold margin and then record ID. Textual inspection of the selected calibration summaries found the nearest rejects to be one symbol-dominated code fragment and repeated navigation lines in EN and UK. The nearest accepts were structured shell, JavaScript and Python samples. Those selections are consistent with the labels and with the intended conservative code policy.

## Current corpus execution

The runner is `tools/run_document_quality.py`. It verifies the current source byte size and SHA-256 before evaluating records, binds the run to the physical current-view manifest SHA-256 and the complete quality-policy SHA-256, and emits deterministic decision and run hashes.

The current view is `configs/data/document_quality_current_corpus_v1.json`. It binds DATA-10's committed `data/synthetic/data10/uk-en-code-train.txt`:

- source SHA-256: `059f04e01d6fc6b8224b373b08efbb37f09d546de35ed510afdb4587ebdb6012`;
- bytes: 1,454;
- quality-view SHA-256: `7ecfe6a72640d1956d6b0c1ee1675c4226f020c10fdcbfae63835da56feaf799`;
- records: 9 total, three each UK/EN/code.

Retained result is `reports/d03/document_quality_current_corpus_20260825.json`:

- input 9;
- accepted 9;
- rejected 0;
- UK 3/3 accepted;
- EN 3/3 accepted;
- code 3/3 accepted;
- decision-set SHA-256 `b74b9404d47463ca51c20d8bec1913df00e9b3e38ed80e2589a3150cccf1d011`;
- run SHA-256 `a032772427d7a0fdcfe6eadf68973556ddb4112421eef8e978bee5408954f140`.

The closest accepted edge samples are the three code records. There are no rejected current-corpus edge samples because this small project-authored fixture is clean by construction.

## 100K-1M truth boundary

At the DATA-32 start point, the visible DATA-25 corpus-v01 branches still resolve to DATA-10 head `077205ef2b1662a5029bc77b8fc762078cabeb17`; no representative 100K-1M corpus manifest or shards are present there yet. DATA-10 itself explicitly labels the 1,454-byte corpus as `PROJECT_AUTHORED_SYNTHETIC_ONLY` and `representative_corpus=false`, with zero external sources approved at recipe creation.

Therefore this package does **not** claim that the requested representative 100K-1M corpus has been quality-filtered. The production filter and manifest contract are ready for that corpus, but the retained real-current execution is mechanics evidence on the exact data actually present. Once DATA-25 publishes a new immutable corpus manifest, the quality run must be repeated against that exact identity; current counts cannot be carried forward.

## Tests

`tests/test_document_quality.py` covers:

- all 30 labeled calibration cases;
- code structure without a natural-text alpha gate;
- line repetition, template density and U+FFFD reasons;
- policy-hash drift when a threshold changes;
- deterministic output under input-order changes;
- run identity change when the input manifest identity changes;
- adaptation to the incumbent D03 quality hook without rights authority;
- exact current source hash/size and expected UK/EN/code counts;
- fail-closed invalid manifest identity.

A local focused run executed 8 tests successfully before publication. GitHub exact-head CI remains the repository authority for full-suite/Ruff status.
