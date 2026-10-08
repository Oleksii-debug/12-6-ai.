# Plan 2 / Section 4 — deterministic normalization and source-candidate record identities

Section 4 **reuses**, rather than replaces, the incumbent DATA324 source normalization and Plan-2 S3 physical materialization. `tools/plan2_normalization_v1.py` consumes only a freshly verified S3 `normalized.utf8` member and its integrity-bound manifest. It does not attempt to reinterpret HTML/raw acquisitions, re-admit source rights, or grant training/evaluation/tokenizer authority.

Versioned policy `12-6.plan2-normalization-policy.v1` decodes strictly as UTF-8; refuses undecodable bytes, NUL and invalid surrogate boundaries; removes one leading BOM; converts CRLF/CR to LF; applies Unicode NFC; and joins non-empty paragraph records with exactly two LF characters. Records bind stable ordinal IDs, source identity, exact Unicode codepoint offsets, SHA-256, script evidence, and explicit text modality.

The language identifier is a deliberately conservative deterministic script heuristic, **not** a calibrated model. Ukrainian evidence requires substantial Cyrillic coverage and a distinct Ukrainian character; English evidence requires strong basic Latin coverage. Very short, mixed, unsupported or ambiguous text stays `unknown` with `supported=false`. Confidence field is named `script_share_not_calibrated` and is not probabilistic. Per-record unknown language never becomes synthetic Ukrainian/English.

Every derivative binds the exact S3 manifest hash and source UTF-8 hash; both the normalization policy and resulting output/record inventory are self-hashed. Changing policy, source, boundaries, script classification, or provenance produces a distinct downstream artifact identity; the verifier recomputes record content hashes, spans, script evidence, full canonical JSON, output hash and scope denials instead of trusting a resealed claim.

Publication reuses the S3 atomic create-if-absent/fsync write mechanism. A valid directory may restart idempotently or recover unpublished missing material; a published manifest cannot be overwritten, patched, repaired or silently extended. The verifier rejects missing, extra, tampered or symlink members, including a tampered manifest resealed with a new checksum.

Local fixture / current candidate qualification:

```bash
PYTHONPATH=src:. pytest -q tests/test_plan2_source_inventory_v1.py tests/test_plan2_source_admissibility_v1.py tests/test_plan2_physical_materialization_v1.py tests/test_plan2_normalization_v1.py tests/test_data324_current_main_port.py
python tools/plan2_normalization_v1.py --root . --out-dir /tmp/plan2-s4-normalized
python tools/plan2_normalization_v1.py --root . --out-dir /tmp/plan2-s4-normalized
```

The source remains an evidence-only **candidate**. `training_corpus_authorized=false`, `tokenizer_fit_authorized=false`, `evaluation_authorized=false`; no GPU, network acquisition or model training. Section 5 may consume verified per-record bytes from this candidate; it must apply its existing G06 privacy authority before any later training eligibility, which remains separately gated.
