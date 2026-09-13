# D03 HPLT 3.0 Ukrainian source-rights audit

## Verdict

`BLOCKED_SOURCE_RIGHTS_AND_IMMUTABLE_ACQUISITION`

This package qualifies HPLT Monolingual Datasets 3.0 `ukr_Cyrl` as a high-capacity discovery candidate. It does **not** admit any HPLT bytes to the canonical 12-6 training corpus. Training-authorized bytes, unique corpus tokens, unique causal-loss positions, tokenizer-fit permission, optimizer updates, downloaded bytes, and paid compute are all exactly zero.

The distinction is deliberate. HPLT provides unusually useful multilingual web-corpus engineering and metadata, but its own terms state that HPLT does not own the underlying extracted text and that CC0 applies to the packaging. The project therefore cannot transform the dataset-card `cc0-1.0` tag into blanket source-level training rights.

## Live upstream facts bound by this package

The official HPLT 3.0 release page and HPLT dataset card report that the monolingual 3.0 release was completed in July 2025 and was derived from Internet Archive and Common Crawl web crawls from 2012 through 2024. HPLT documents Trafilatura-based extraction, OpenLID 2.0 language identification, Monotextor processing, quality sorting, provenance metadata, PII annotation, and global deduplication for language portions other than Chinese, English, and Russian. Ukrainian is therefore reported as globally deduplicated upstream.

For `ukr_Cyrl`, HPLT reports approximately 137.16 GB sorted data, 80.03 million documents, 81.22 billion tokens, 244.66 billion characters, and 1.61 billion segments. These are upstream catalogue statistics. They are **not** 12-6 source capacity, training-authorized capacity, tokenizer output, or unique causal-loss capacity.

The immutable dataset-card reference used for this audit is Hugging Face commit `3394d6ba8dae4da834e3b11771daf95028a960b1`. The actual corpus is not hosted on Hugging Face. HPLT publishes language-specific data through its NIRD-hosted download site, including:

- `https://data.hplt-project.org/three/sorted/ukr_Cyrl.map`
- `https://data.hplt-project.org/three/sorted/ukr_Cyrl.md5`

The map and MD5 files are discovery/integrity inputs. This audit did not retrieve or freeze those remote objects, so it deliberately records their snapshot SHA-256 identities as unresolved. MD5 is not accepted as the final 12-6 artifact identity; a successor acquisition package must snapshot the control files and independently SHA-256 every acquired shard.

## Rights boundary

HPLT's Terms of Use make two points that must stay coupled:

1. HPLT says it does not own the text from which the dataset was extracted.
2. HPLT licenses the packaging under Creative Commons CC0.

Consequently:

`CC0 packaging != ownership of underlying text != source-level training authorization`

Every acquired record or source family still needs project-owned provenance and rights treatment appropriate to the intended use. Unknown, ambiguous, restrictive, removed, or legally unresolved material fails closed. HPLT's takedown mechanism and its statement that downstream users are responsible for applicable legal compliance are additional reasons not to treat the package-level label as universal authorization.

This package is a technical/scientific control, not legal advice. It preserves the fact that legal and privacy review remains a downstream gate instead of pretending the worker can settle source rights from a dataset card.

## Metadata that must survive acquisition

A successor must retain enough HPLT fields to reconstruct origin and later decisions. The HPLT card documents crawl/source metadata, document and segment language information, source URLs, crawl IDs, MinHash cluster size, PII annotations, document scores, register labels, and text/XML representations. A local conversion that strips provenance, source identity, PII metadata, or cluster information before audit would be a regression.

Upstream processing is evidence, not a substitute for project gates:

- HPLT global dedup does not replace the 12-6 global cross-source dedup graph.
- HPLT PII annotations do not replace the 12-6 privacy detector/review gate.
- OpenLID output does not replace a project language-ID validation policy.
- WDS quality ordering does not become a project quality threshold without an experiment contract.
- HPLT source metadata does not itself prove rights.
- HPLT corpus statistics do not establish one-pass unique causal-loss positions.

## Successor acquisition sequence

The safe next package is bounded and deterministic rather than a 137 GB bulk pull.

1. Snapshot `ukr_Cyrl.map` and `ukr_Cyrl.md5`; record retrieval time, HTTP metadata where available, raw byte length, and SHA-256 for both control objects.
2. Parse the map into a stable ordered shard inventory. Reject non-HTTPS, unexpected host/path, duplicate shard identities, malformed WDS bins, or map/checksum disagreement.
3. Select a small, preregistered shard sample before transfer. Selection must be deterministic and must not use evaluation outcomes.
4. For each acquired shard, verify the HPLT-provided MD5 and separately compute SHA-256. The SHA-256 becomes the project artifact identity.
5. Stream a bounded record sample and preserve origin/provenance fields. Produce source-family/domain distributions rather than granting the whole shard one blanket right.
6. Perform source-level rights classification. Unknown rights remain excluded from training credit.
7. Run project privacy checks independently, using HPLT PII annotation only as an upstream signal/cross-check.
8. Run project quality/language-ID validation and the global cross-source dedup pipeline.
9. Apply the evaluation-reservation/decontamination authority before tokenizer fitting or training credit.
10. Only after the corpus is immutable and split can D04 materialize tokenizer/packing identities and the unique-loss ledger. Until then, learned training authorization stays zero.

## Machine-readable contract

`configs/data/d03_hplt3_ukr_cyrl_source_audit_v1.json` is the canonical artifact for this worker. `src/twelve_six/hplt3_source_policy.py` validates the science/rights boundary and a self-hash. The validator intentionally only accepts the audit-only state: every downstream project gate remains `BLOCKED`, all training credit is zero, and immutable acquisition is false. A future acquisition package should define a successor schema instead of editing this evidence to retroactively claim terminality.

Validation command:

`PYTHONPATH=src python tools/validate_d03_hplt3_ukr_cyrl_source_audit.py`

Focused tests:

`PYTHONPATH=src pytest -q tests/test_d03_hplt3_ukr_cyrl_source_audit.py`

The adversarial matrix rejects package-license overclaim, any training credit, premature project-gate PASS, false immutable acquisition, mutable upstream identity, missing project gates, and evidence tampering.

## Non-claims

This package did not download HPLT corpus shards, did not read final-test payloads, did not train a tokenizer or model, did not run optimizer updates, did not provision GPU/cloud storage, and did not use paid compute. It does not claim that all HPLT Ukrainian text is lawful or appropriate for training. It does not claim that HPLT's reported 81.22B tokens are tokens under the future 12-6 tokenizer. It does not claim that upstream global dedup guarantees project-global uniqueness or benchmark decontamination.

Those limits are the deliverable: HPLT can now enter the project as a precisely bounded acquisition candidate rather than as an unqualified 137 GB shortcut.
