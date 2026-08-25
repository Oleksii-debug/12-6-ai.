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
2. `data108-real-balanced-v2`;
3. `data108-real-preserve-v2`;
4. `data108-real-strict-v2`.

Selection uses only calibration labels. The deterministic objective is, in order: minimize the maximum number of errors in any source family; minimize total classification errors; minimize false rejects; minimize false accepts; then prefer the earlier candidate. This makes false-removal avoidance explicit without allowing aggregate accuracy to hide a source-family failure.

Reports include false-accept and false-reject counts and conditional rates by source family and modality as well as overall.

## Complete-current-corpus effect

The execution runner rebuilds and verifies the exact DATA-25 V0.1 project corpus through the incumbent builder, then appends the exact three accepted DATA-21/22 normalized external records. The selected policy is run over that complete current evidence scope.

For each rejected document the report retains record id, source family, modality, UTF-8 bytes, byte-token count, and rejection reasons. Aggregate removed documents, bytes/tokens, modality effects, and source-family effects are reported exactly.

A controlled training A/B is required only if the selected policy removes at least 0.5% of current byte-token mass or at least 1.0% of current documents. Below both thresholds, DATA-108 records `NOT_REQUIRED_NO_MATERIAL_COMPOSITION_CHANGE`; training on effectively unchanged composition would not be a meaningful quality-filter A/B.

Model loss is not used to define the quality labels or choose the policy.

## LOCAL_FREE reproduction

The dedicated GitHub Actions workflow uses only the repository's existing hosted evidence path, Python 3.11, pytest, the exact retained DATA-21/22 artifact, and the existing corpus builder. It does not launch paid accelerators or paid compute.

The runner command is:

```text
PYTHONPATH=src python tools/run_document_quality_real.py --data21-artifact data21-22-external-source-intake.zip --output-dir reports/d03/data108
```

The authoritative execution outputs are `calibration.json`, `holdout.json`, `complete_current_corpus_effects.json`, and `recommendation.json`. Claims must be taken from a terminal-success exact-head run, not from queued or stale workflows.
