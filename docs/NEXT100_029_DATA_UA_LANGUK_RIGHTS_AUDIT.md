# NEXT100-029 — Ukrainian Lang-UK corpus rights and quality audit

Worker: `NEXT100-029-DATA-UA-LANGUK`

Execution profile: `LOCAL_FREE`

## Terminal result

`RETEST_LANGUK_COURT_DECISIONS_ONLY`

No source is admitted for model training by this worker. No training payload is materialized and no source-registry mutation is authorized.

The audit fails closed wherever research availability, a corpus-level license label, or a downstream pretraining use case is not backed by source-specific reuse rights and a project-native privacy decision.

## Incumbent boundary

This audit is based on DATA-301 head `8820ba1b255f6bb95c7db0531fd846078a1aae01`. That authority is `TERMINAL_BLOCKED` and binds the current five-object / four-family external-real candidate with only one Ukrainian family: `ua.rada.open-data.laws-texts`.

A new Ukrainian corpus must therefore be genuinely independent in lineage and must not be counted merely because it has another download URL or downstream transformation.

## Candidate 1 — UberText / UberText 2.0

Verdict: `REJECT_MIXED_RIGHTS`.

The Lang-UK UberText page explicitly says that license restrictions of some periodicals prevent unchanged publication. It distributes shuffled sentences under a stated Fair Use rationale for statistical and scientific analysis. That is not treated as a source-specific redistribution license or as explicit project model-training permission.

The corpus aggregates periodicals and other web/text sources. UberText 2.0 further combines news, Wikipedia, fiction, court decisions, and social media. The mixed provenance cannot be safely converted into one rights-clean family without separating and re-qualifying components.

Privacy is also not cleared: news and social/web material can contain personal data. No payload is acquired.

Evidence:
- https://www.lang.org.ua/en/produkty/korpusi/ubertext-corpus/
- https://www.lang.org.ua/uk/produkty/korpusi/ubertext-2-0/

## Candidate 2 — Brown Ukrainian corpus

Verdict: `REJECT_CHAIN_OF_TITLE_UNPROVEN`.

Pinned upstream commit: `11ed17ea1c74b138fe6cc80b03dfea4f0e63abd9`.

The repository README states `CC BY-NC-SA 4.0` for corpus data. The metadata, however, enumerates many third-party newspaper and web fragments with named authors, publications, and source URLs. This worker found no immutable component-level evidence establishing that every included source was licensed or assigned in a way that permits the corpus maintainers to relicense the underlying text under the stated corpus license.

Under the project fail-closed rule, the corpus-level label is insufficient to cure an unverified chain of title. It is lineage-distinct from Rada, but it is not admitted and contributes zero family diversity.

Pinned evidence blobs:
- README blob `79bbc783338133b035a4b03c657824ec3ad38b0d`
- `meta/meta.csv` blob `f6fafd2d971899c334570957e933d59fb80c72b5`

## Candidate 3 — Lang-UK `court-decisions-uk`, 2024 Supreme Court 5K file

Verdict: `RETEST_PRIVACY_AND_BYTE_MATERIALIZATION`.

This is the only candidate whose rights layer is strong enough to preserve for a bounded retest.

Dataset repository: `lang-uk/court-decisions-uk`

Current observed repository revision: `289c0316fc076db3e1607db6776a29df42f4ffc5`

Selected file: `2024-5K-supreme-court-decisions-deduplicated.parquet`

File-origin commit: `2dcac4c941b87bf9c242bdc919cef4b40f4a4813`

Upstream content identity: SHA-256 `9b8870d10695715e4a0540c6f8fdca381599c0e6cdbaf3ecdf3c0782207b6597`, 20,220,778 bytes.

Upstream dataset license label: `MIT`.

The underlying documents are official judicial decisions. Article 8(1)(3) of current Ukrainian Law No. 2811-IX excludes official documents of judicial character from copyright protection. This removes the principal third-party copyright defect that blocks UberText news and Brown-corpus fragments. The Lang-UK landing page independently identifies the court-decisions corpus as MIT.

Rights evidence:
- https://www.lang.org.ua/uk/produkty/korpusi/
- https://huggingface.co/datasets/lang-uk/court-decisions-uk
- https://www.wipo.int/wipolex/en/legislation/details/22385
- https://zakon.rada.gov.ua/laws/main/l20392z22

The source family would be `ua.languk.supreme-court-decisions`. It is independent from the incumbent Rada statute family because its primary publishing lineage is Supreme Court / Unified State Register of Court Decisions, not Verkhovna Rada legislation. It must nevertheless be cross-family deduplicated because court decisions frequently quote statutes and other legal boilerplate.

### Why it is not admitted now

The dataset viewer exposes `person_count`, `address_count`, and occurrence structures. Court decisions can contain personal or sensitive case information even when source redaction placeholders are present. This worker did not obtain the exact parquet bytes into the project execution environment and therefore did not run the deterministic project-native privacy detector over every candidate row.

Without byte-exact acquisition and privacy clearance, neither a raw subset hash nor normalized subset hash may be fabricated. The materialized subset is therefore exactly zero records / zero bytes.

The synthetically deanonymized file `250-deanonymized-court-cases.parquet` is explicitly excluded from any future language-model training intake under this authority.

### Bounded retest contract

A successor may only use the pinned 20,220,778-byte Supreme Court file and must first verify its SHA-256. It may retain at most 256 rows, selected by ascending stable source ID after privacy PASS and after excluding any evaluation-reserved identity.

Required order:

1. Verify exact raw file SHA-256 before parsing.
2. Run deterministic privacy/PII scanning and quarantine any failing record.
3. Measure Ukrainian-language quality on retained rows; do not infer language quality from metadata alone.
4. Normalize deterministically while preserving source IDs.
5. Remove raw and normalized exact duplicates.
6. Detect template and near-copy clusters within the court family.
7. Cross-family compare with Rada for quoted-statute fragments and near-mirror clusters.
8. Cluster before train/selection splitting; a near-duplicate cluster may not cross boundaries.
9. Publish raw and normalized SHA-256 for the final <=256-record bounded subset.
10. Refresh the live terminal source registry before any admission.

## Candidate 4 — Malyuk

Verdict: `REJECT_MIXED_UPSTREAM_RIGHTS`.

Malyuk is explicitly a combined pretraining corpus incorporating UberText 2.0, OSCAR, Wikipedia, and additional sources. A downstream pretraining purpose is not source-level permission. Since the mixed upstream rights are not separated and independently qualified here, Malyuk is not admitted.

## Quality and family interpretation

The Supreme Court candidate is formal, edited Ukrainian legal language and is useful as a narrow legal/administrative register. It is not representative of general Ukrainian language by itself and must not be promoted as a general-language replacement for a second independent broad Ukrainian family.

Upstream reports LSH deduplication before the 5K Supreme Court release. That is useful provenance but is not a substitute for the project's deterministic dedup and cross-family contamination policy.

## Truth boundary

Training executed: false.

Optimizer updates: 0.

Payload materialized: 0 records / 0 bytes.

New Ukrainian family admitted: false.

Registry mutation authorized: false.

Evidence identity is recorded in `configs/data/next100_029_languk_rights_audit_v1.json` and is validated by `tools/validate_next100_029_languk_rights_audit.py`.
