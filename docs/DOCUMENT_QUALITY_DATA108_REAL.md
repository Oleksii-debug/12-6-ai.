# DATA-108 — real-corpus document-quality calibration

## Scope

DATA-108 extends the incumbent DATA-32 scorer in `twelve_six.data.document_quality`. It does not replace feature extraction, decisions, D03 policy hooks, corpus building, source-rights review, language admission, deduplication, packing, training, or evaluation.

The reason for this recalibration is narrow: DATA-32's original 30/30 calibration set was project-authored, while its retained DATA-25 corpus was also project-authored. Perfect classification on that set is not evidence of false-accept or false-reject behavior on admitted external source material.

## Real evidence boundary

DATA-108 consumes the retained terminal-success DATA-21/22 bounded intake artifact with SHA-256 `3b4ef6c0d42725f7b660935f70ef1f8b41b90eb9d5d73c83455401a434233122` and internal manifest identity `9d50c0baf98247c1babc5fca8dead5b1fa87264ad92ea62527c34e342a7dd735`.

That artifact contains exactly three accepted normalized records from two currently admitted source families:

- `ua.rada.open-data.laws-texts` — Ukrainian official legislation.
- `en.standardebooks.manual` — English CC0 technical/style documentation pinned to the reviewed Standard Ebooks commit.

No independent real code source family is currently admitted. Code-like calibration examples are therefore taken only from legitimate HTML/CSS/XML/structured-data passages inside the admitted Standard Ebooks manual, plus project-authored controls. This is not represented as evidence for a separate real code corpus.

The DATA-21/22 artifact itself is a bounded real intake, not a canonical full external-source snapshot. DATA-108 preserves that authority boundary explicitly.

## Labels and privacy

`data/quality/calibration_real_sources_v1.jsonl` is the threshold-selection partition. `data/quality/holdout_real_sources_v1.jsonl` is a separately frozen holdout. External samples store public source record identities and deterministic non-empty-line ranges rather than private material. Project-authored controls contain only deliberately constructed text.

Every label includes a human-readable rationale. The combined labeled evidence covers navigation/boilerplate, legal-template-like text, tables, short legitimate records, repetitive legitimate references, mechanical repetition, mixed-language material, code-heavy prose, structured data, replacement-character OCR-like corruption, and high-symbol legitimate code. No OCR corruption was observed in the admitted real artifact, so that case is represented by an explicit project-authored negative control rather than invented as a real-source observation.

Calibration and holdout source ranges are non-overlapping. The holdout is never passed to the policy-selection function.

## Policy selection

DATA-108 evaluates a small predeclared set of threshold variants while retaining DATA-32's exact feature extractor and decision logic:

1. incumbent DATA-32 policy;
2. `data108-real-balanced-v3`;
3. `data108-real-preserve-v2`;
4. `data108-real-strict-v2`.

The first balanced candidate exposed a real calibration failure: a legitimate Rada amendment register had zero repeated-line density but a distinct-token ratio of `0.1120689655` and a dominant-token ratio of `0.3275862069`, so DATA-32-style natural-language diversity gates incorrectly rejected it. Using the calibration partition only, the balanced candidate was revised to natural-language `min_distinct_token_ratio=0.10` and `max_dominant_token_ratio=0.34`. The frozen holdout was not consulted during that change.

Selection uses only calibration labels. The deterministic objective is, in order: minimize the maximum number of errors in any source family; minimize total classification errors; minimize false rejects; minimize false accepts; then prefer the earlier candidate. This makes false-removal avoidance explicit without allowing aggregate accuracy to hide a source-family failure.

The selected policy identity from the deterministic local replay is `data108-real-balanced-v3`, SHA-256 `64ee3abbb3e349314d945f6a8914a5c5331cda37ef7424614584ccbe8583acd5`.

The local replay classified all 20 calibration examples correctly: 0 false accepts and 0 false rejects. After policy selection was frozen, the separate 14-example holdout also produced 0 false accepts and 0 false rejects. Reports are broken down by source family and modality; the retained summary is `reports/d03/data108/local_replay_summary.json`.

## Complete-current-corpus effect

The execution runner rebuilds and verifies the exact DATA-25 V0.1 project corpus through the incumbent builder, then appends the exact three accepted DATA-21/22 normalized external records. The selected policy is run over that complete current evidence scope.

The local deterministic replay reproduced the retained DATA-25 key counts exactly: 46,207 documents and 21,411,248 byte-tokens. It then evaluated the exact three retained DATA-21/22 real records, adding 173,358 byte-tokens. The complete current scope therefore contains 46,210 documents and 21,584,606 byte-tokens.

`data108-real-balanced-v3` removed 0 documents, 0 UTF-8 bytes and 0 byte-tokens from that scope. The exact removed-document list is therefore empty.

A controlled training A/B is required only if the selected policy removes at least 0.5% of current byte-token mass or at least 1.0% of current documents. Both observed fractions are exactly 0.0, so DATA-108 records `NOT_REQUIRED_NO_MATERIAL_COMPOSITION_CHANGE`. A ~250K training A/B would compare identical corpus composition and is therefore not executed.

Model loss is not used to define the quality labels or choose the policy.

## LOCAL_FREE reproduction

The retained local replay ran on Python 3.13.5, Linux x86-64 under KVM, with 5 visible Intel Xeon Platinum 8573C CPUs. It reconstructed the exact retained DATA-25 generation/split/accounting logic and reproduced its document and byte-token totals before applying the unchanged DATA-32 feature/decision semantics plus the selected threshold policy. The exact DATA-21/22 artifact was read locally and hash-verified.

The dedicated GitHub Actions workflow uses only the repository's existing hosted evidence path, Python 3.11, pytest, the exact retained DATA-21/22 artifact, and the existing corpus builder. It does not launch paid accelerators or paid compute.

The runner command is:

```text
PYTHONPATH=src python tools/run_document_quality_real.py --data21-artifact data21-22-external-source-intake.zip --output-dir reports/d03/data108
```

The authoritative hosted execution outputs are `calibration.json`, `holdout.json`, `complete_current_corpus_effects.json`, and `recommendation.json`. A queued hosted run is not represented as PASS; when terminal-success evidence exists, its outputs supersede the local replay summary for hosted-run claims.
