# Plan 2 / Section 2 — rights and member provenance

The incumbent DATA324 validator is the *only* rights authority used by the Plan-2 adapter. `tools/plan2_rights_provenance_v1.py` performs a fresh independent read of its bound source snapshots, pinned licence bytes, attribution, source and normalization hashes, and explicit permission boundaries. It never invents a blanket LLM-training licence or corpus eligibility.

The output is a deterministic source-level provenance receipt including upstream revision, source family, source path, raw and normalized SHA-256, legal basis, actual licence evidence, and record/member identity. The receipt verifier requires independently reacquired source evidence, not a self-issued hash. Unknown, candidate-promoted, drifted, forged, or symlinked inputs are rejected.

**Current exact boundary:** `uk.kubernetes.docs.what-is-kubernetes` has CC-BY-4.0 source-level evidence, but is a **candidate** in the Plan-2 registry. Corpus-level deduplication, evaluation decontamination and training approval are still absent. `corpus_training_authorized=false`, `tokenizer_fit_authorized=false`, and `evaluation_authorized=false` remain explicit. There is no production corpus materialization or paid compute here.

Qualification command: `pytest -q tests/test_plan2_rights_provenance_v1.py tests/test_plan2_source_inventory_v1.py tests/test_data324_current_main_port.py`. Conflict key: `data-lineage`.
