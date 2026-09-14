# peS2o source-rights and contamination audit V1

Status: `REVIEW_REQUIRED_ZERO_TRAINING_CREDIT`

Authority: SWARM-745 / issue #745, implementing the peS2o audit item in #720 without touching the active Research Corpus V1 composition, deduplication, tokenizer, trainer, checkpoint, or CI-workflow surfaces.

## Result

peS2o is useful as an academic-text acquisition candidate, but this audit does **not** authorize any peS2o record for 12-6 training. The current project credit is exactly zero bytes and zero optimizer updates.

The decisive boundary is that peS2o exposes several different licensing layers:

- The upstream peS2o code repository is pinned here to `allenai/peS2o@cc4800ffd798bb9baa29552059b294073cf91e90`; its repository `LICENSE` is Apache-2.0.
- The Hugging Face dataset is pinned here to revision `636a503e44a3ca1b58e01fb61eab0825cd574de0`; the dataset card identifies the dataset as ODC-By.
- The peS2o README says the data is derived from Semantic Scholar and records only `s2orc` full text or `s2ag` title/abstract in the `source` field. The documented record fields are `added`, `created`, `id`, `source`, `text`, and `version`; there is no documented per-record license field in that schema.
- Semantic Scholar states that S2 data can include third-party content whose underlying terms remain separately applicable, and that permissions for paper content must come from the relevant author or publisher rather than Semantic Scholar.

Therefore Apache-2.0 for the code, ODC-By for a dataset layer, an `open access` label, or a Semantic Scholar Corpus ID is not sufficient by itself to mark an underlying paper `TRAINING_AUTHORIZED` for this project.

## Upstream evidence pinned by this audit

1. peS2o code repository, exact commit: <https://github.com/allenai/peS2o/commit/cc4800ffd798bb9baa29552059b294073cf91e90>
2. peS2o code license at that commit: <https://github.com/allenai/peS2o/blob/cc4800ffd798bb9baa29552059b294073cf91e90/LICENSE>
3. peS2o dataset, exact Hugging Face revision: <https://huggingface.co/datasets/allenai/peS2o/commit/636a503e44a3ca1b58e01fb61eab0825cd574de0>
4. peS2o README and source schema at the exact code commit: <https://github.com/allenai/peS2o/blob/cc4800ffd798bb9baa29552059b294073cf91e90/README.md>
5. S2ORC current license/readme: <https://github.com/allenai/s2orc/blob/master/README.md>
6. Semantic Scholar API license boundary for third-party content: <https://www.semanticscholar.org/product/api/license>
7. Semantic Scholar permission FAQ: <https://www.semanticscholar.org/faq>

This is a technical provenance/rights gate, not a legal opinion. A later admission decision must use the exact terms attached to the records actually selected for the intended use.

## Source-channel findings

### `s2orc` full text

The peS2o README describes these as full-text papers processed from S2ORC. A Semantic Scholar Corpus ID is useful provenance, but the peS2o schema does not carry document-level license evidence. `S2ORC_FULLTEXT` therefore remains `REVIEW_REQUIRED`, with zero admitted bytes.

### `s2ag` title and abstract

The peS2o README describes these as title/abstract records from S2AG. The same missing document-level license-evidence problem applies. `S2AG_TITLE_ABSTRACT` remains `REVIEW_REQUIRED`, with zero admitted bytes.

## Contamination boundary

The upstream peS2o v1/v2 README describes a publication-date train/validation split. That split is an upstream dataset-design choice, not 12-6 evaluation clearance. It cannot substitute for the project's reserved benchmark/final-test firewall.

Before any peS2o-derived record can be admitted to training, a later D03 owner must:

1. freeze one immutable dataset revision and exact shard inventory with content hashes;
2. resolve source-specific or document-level rights evidence for every admitted record;
3. retain Semantic Scholar Corpus ID and peS2o source channel in the project provenance ledger;
4. reject unknown, ambiguous, conflicting, or restrictive rights evidence for the intended use;
5. exclude all project-reserved evaluation/final-test material and run exact plus near decontamination;
6. then run the normal quality/privacy/dedup/split/packing and unique-loss accounting gates;
7. obtain ordinary dataset-freeze/promotion evidence before tokenizer fitting or optimizer updates.

No benchmark or final-test payload was accessed by this audit. No peS2o data was bulk-downloaded. No paid compute, tokenizer fitting, model training, or foreign pretrained weights were used.

## Machine check

Run:

```bash
python tools/validate_pes2o_source_audit_v1.py
pytest -q tests/test_pes2o_source_audit_v1.py
```

The validator intentionally rejects any mutation that converts package-level licensing, open-access status, or unresolved source channels into training authority; grants nonzero corpus credit; fabricates a document-level license field; treats the upstream validation split as project clearance; claims decontamination that did not happen; accesses final-test payloads; or authorizes paid compute.
