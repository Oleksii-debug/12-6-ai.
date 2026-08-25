# DATA-32 document quality filter

## Scope

DATA-32 adds a lightweight deterministic document-quality decision to the incumbent D03 record-policy seam. It does not decide source rights, acquisition permission, PII, copyright, benchmark contamination, or language admission. Those remain separate D03 hooks and gates.

Implementation: `src/twelve_six/data/document_quality.py`. A decision is adapted to the existing `PolicyHookEvidence` as `hook_id=document_quality`; it never creates or modifies rights, language, PII, or copyright evidence.

No external pretrained model is used.

## Incumbent audit

The D03 lineage already had the right policy separation but weak quality implementations:

- early S0 `pipeline.py` combined minimum/maximum length, control characters, PII and a natural-text alpha-ratio heuristic;
- D03 `corpus_foundation.py` introduced independent `quality`, `language`, `pii`, and `copyright` hooks, with source rights explicitly outside those hooks;
- DATA-10 added strict UTF-8, replacement-character rejection, modality-aware normalization, and UK/EN/code admission;
- DATA-25 corpus V0.1 used a deterministic but narrow `quality()` gate: natural text was primarily length + script ratio, while code required Python-like `def` and `return` structure.

The residual defect was therefore not a missing governance framework. It was the absence of an interpretable, modality-aware quality implementation behind the existing quality hook. Natural-text alpha ratio is not a valid generic source-code quality gate, PII is not a quality feature, and a Python-only structural gate is not a general code policy.

DATA-32 does not create a second rights registry, language-ID system, dedup engine, or black-box quality model.

DATA-32 also does not rewrite the retained DATA-25 builder for corpus V0.1. That would change DATA-25's builder SHA and therefore its corpus identity even if content remained unchanged. Instead the exact retained DATA-25 corpus identity is treated as immutable input to a post-filter quality pass.

## Features and thresholds

Policy ID: `d03-lightweight-uk-en-code-v1`.

Policy SHA-256: `97b9fe1452b22c6275a27f85524f670253a7f4012377361c4cb007004aeccd1d`.

Recorded interpretable features include:

- character count and UTF-8 byte count;
- U+FFFD, surrogate, and disallowed-control counts;
- Latin, Cyrillic, and other-script letter counts;
- punctuation/symbol ratio;
- repeated non-empty line ratio;
- URL count and URL-character density;
- fixed-marker template and boilerplate line density;
- token count, unique-token ratio, and dominant-token ratio;
- code identifiers, keyword hits, indentation, delimiters/comments, and an explicit code-structure score.

Natural-text policy is deliberately permissive: minimum 60 characters, maximum 250,000, symbol ratio <= 0.40, repeated-line ratio <= 0.60, URL-character ratio <= 0.25, template/boilerplate line ratio <= 0.50, diversity checks only from 30 tokens onward, and other-script-letter ratio <= 0.20. UK versus EN identity is not re-decided here; the D03 language hook owns that question.

Code has a separate policy: minimum 30 characters, maximum 400,000, symbol ratio <= 0.78, repeated-line ratio <= 0.75, URL-character ratio <= 0.45, template/boilerplate line ratio <= 0.70, diversity checks only from 20 tokens onward, and code-structure score >= 2. This allows structured SQL, shell, JavaScript/TypeScript and similar material instead of imposing a natural-text alpha gate or Python-only `def/return` rule.

The integer score is diagnostic only: 100 minus 25 per reject reason and 5 per warning. Admission is based on explicit named threshold reasons, not on a learned score.

## Calibration

`data/quality/calibration_uk_en_code_v1.jsonl` contains 30 project-owned labeled samples: 10 Ukrainian, 10 English, and 10 code; each stratum has five ACCEPT and five REJECT labels.

Reject examples cover short fragments, repeated lines, template density, URL density, low lexical diversity/dominant-token repetition, symbol-dominated fragments, and insufficient code structure.

Retained evidence: `reports/d03/document_quality_calibration_20260825.json`.

Measured calibration result:

- samples: 30;
- correct: 30;
- false accepts: 0;
- false rejects: 0;
- accuracy: 1.0 overall and 1.0 in each UK/EN/code stratum.

This is calibration evidence only. It is not a population-level precision/recall claim for arbitrary web or code corpora.

The deterministic edge selector chooses the smallest absolute threshold margin and then record ID. Manual textual inspection of selected calibration summaries found the nearest rejects to be one symbol-dominated code fragment and repeated navigation lines in EN and UK. The nearest accepts were structured shell, JavaScript, and Python snippets. Those classifications are consistent with the labels and intended conservative policy.

## Full current corpus execution

DATA-32 is bound to DATA-25 corpus V0.1 head `8af17afa7baf3d75c2328caf8b08af2400a95e09` and corpus identity:

`422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`

The retained DATA-25 manifest describes 36 shards and 46,207 final documents / 21,411,248 byte tokens. Physical shard bytes are generated artifacts, not committed. Therefore `tools/run_document_quality.py` deterministically rebuilds the exact DATA-25 corpus from its retained builder/config, requires corpus identity and ordered shard hash/size/document tuples to match the retained manifest, re-hashes every rebuilt shard, then runs DATA-32 over every record.

Rebuild evidence matched the DATA-25 manifest, including the published first and last shard identities. Final DATA-32 result:

- input: 46,207 documents;
- accepted: 46,207;
- rejected: 0;
- UK: 13,899 accepted / 0 rejected;
- EN: 20,093 accepted / 0 rejected;
- code: 12,215 accepted / 0 rejected;
- reproduced byte tokens: 21,411,248;
- reproduced shards: 36;
- decision-set SHA-256: `ed090cc8380365451b0788d18931bb81ebbcf5ed3a270673deeb297bcda115c4`;
- quality run SHA-256: `5087d5b944d6bd129254728ffd5033b53c24039cf82f4692db67494c08518b64`.

Retained evidence: `reports/d03/document_quality_current_corpus_20260825.json`.

Zero rejection is not evidence that the filter is ineffective. DATA-25 V0.1 is already a project-authored, mechanically filtered corpus. DATA-32 is intentionally conservative and does not manufacture rejects to improve-looking metrics.

The five nearest accepted full-corpus edge samples are English project-authored passages. Their closest threshold is dominant-token concentration: the passages are coherent and remain below the rejection limit. Textual review also exposes a separate limitation: DATA-25's project-authored generator produces highly regular phrasing across documents. DATA-32 measures document-local quality; it does not claim to measure cross-document generator homogeneity or external-source diversity. Those are corpus representativeness/diversity concerns and must not be hidden behind a document-quality score.

## Determinism and fail-closed binding

`run_quality_filter()` sorts by record ID before evaluation. Policy identity is the canonical SHA-256 of all thresholds and switches. Run identity includes the exact input corpus identity, policy identity, per-record decisions/features, counts, reasons, and deterministic edge selection.

`tools/run_document_quality.py` fails closed if the configured DATA-25 corpus identity, document count, byte-token count, shard count, external-source eligibility count, rebuilt corpus identity, or ordered shard identities drift. A new DATA-25 corpus identity therefore requires a fresh quality pass.

Rights remain separate. A DATA-32 PASS never grants model-training eligibility.

## Tests

`tests/test_document_quality.py` covers:

- all 30 project-owned calibration labels;
- code structure without a natural-text alpha gate;
- line repetition, template density, and U+FFFD reasons;
- quality-policy hash drift when a threshold changes;
- deterministic output under input-order changes;
- run identity change when the input corpus identity changes;
- adaptation to the incumbent D03 quality hook without rights authority;
- DATA-25 authored UK/EN/code compatibility samples;
- retained full-corpus report binding to the exact DATA-25 manifest;
- fail-closed invalid manifest identity.

The full-corpus execution itself also validates the uncommitted generated shards against the retained DATA-25 manifest before quality evaluation.

## Truth boundary

At this corpus identity, `external_training_eligible_sources` is zero. DATA-25 V0.1 contains project-authored data only. It is useful and sufficiently large for the intended local 100K-1M small-model mechanics, but it is not evidence of representative real-world external source diversity.

DATA-32 makes no claim of universal document cleanliness, semantic quality, semantic deduplication, or corpus representativeness.
