# D03 eCFR Versioned Acquisition Probe V1

## Purpose

This package opens a discovery-only acquisition lane for a large English regulatory-text prospect from the official Electronic Code of Federal Regulations (eCFR). It is deliberately a zero-credit probe. It does not materialize a corpus object, grant training rights, fit a tokenizer, or authorize model training.

The learned ~20M campaign is currently data-gated. Adding another small hand-picked document would not materially change that bottleneck. eCFR is attractive because the Office of the Federal Register exposes point-in-time data and GPO publishes structured XML, so a successor can acquire exact title/date objects and bind their bytes instead of scraping moving HTML.

## Official technical basis

The official eCFR API documentation is `https://www.ecfr.gov/developers/documentation/api/v1` and declares `https://www.ecfr.gov` as the API base. GPO also exposes eCFR XML through its bulk-data infrastructure.

This probe binds:

- API documentation: `https://www.ecfr.gov/developers/documentation/api/v1`
- title metadata: `https://www.ecfr.gov/api/versioner/v1/titles.json`
- historical full-title template: `https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml`
- conservative source ID: `en.us.ecfr.regulations`
- conservative family ID: `us.federal-regulations.ecfr`

The official title-metadata response checked on 2026-08-26 reported `meta.date=2026-08-06`, `import_in_progress=false`, titles 1-50, and title 35 as reserved. The probe hard-binds that observation instead of an older discovery snapshot. It remains discovery evidence rather than an immutable training artifact.

A successor must not request a purported point-in-time object later than the frozen title-metadata availability date without first publishing a refreshed source-authority observation. It must also reject titles marked reserved rather than letting a syntactically valid title number masquerade as materializable content.

## Rights/provenance boundary

17 U.S.C. §105 generally makes copyright protection unavailable for works of the United States Government. It also explicitly permits the U.S. Government to receive and hold transferred copyrights. Legislative history and Copyright Office guidance distinguish employee-produced federal works from contractor/private works.

Therefore this package does not infer training rights from federal hosting, from the word “regulation,” or from public availability.

Before any eCFR bytes can become training-eligible, the materialized XML must be classified so that incorporated-by-reference material, contractor/private authorship, transferred-copyright material, third-party tables/images/media, and provenance-ambiguous payloads are excluded or separately cleared. Foreign copyright status is not inferred from U.S. §105.

This is an engineering gate, not legal advice or a blanket legal conclusion about the entire CFR/eCFR.

## Family accounting

At this stage all eCFR content is one conservative family. Different titles, agencies, chapters, or parts must not be counted as independent training families merely because the XML hierarchy differs. Any later family split requires a dedicated lineage analysis and must still survive global exact/near/fragment/lineage deduplication.

This prevents a single federal compilation from manufacturing diversity credits.

## Successor materialization contract

The next implementation should remain fail-closed:

1. Query the bound title metadata and select only an already available point-in-time date no later than its frozen `meta.date` unless a new observation is separately sealed.
2. Reject reserved titles; in the current frozen observation Title 35 is reserved.
3. Select a bounded set of exact `(date, title)` objects.
4. Fetch each exact historical XML object twice independently.
5. Reject redirects to unexpected origins, wrong content type, malformed XML, byte drift, or unequal repeat acquisitions.
6. Seal raw SHA-256 and byte count per exact object.
7. Parse XML deterministically and preserve enough structural provenance to trace every emitted text record back to title/chapter/part/section.
8. Execute rights/provenance classification before granting any training eligibility.
9. Apply language/quality/privacy filtering.
10. Run global exact, near, fragment, and lineage dedup against the live corpus authority.
11. Apply evaluation-reservation decontamination.
12. Recompute balance and family caps without replay.
13. Create cluster-safe splits, deterministic tokenization/packing, and two clean byte-identical builds.
14. Count exact post-pack non-ignored unique causal-loss positions.
15. Only after tokenizer, D05 checkpoint, evaluation and compute gates are terminal may a learned campaign consume the resulting exposure.

No step in this document turns source bytes into token-budget or training authorization.

## Validation

Run locally:

`python tools/validate_d03_ecfr_acquisition_probe.py`

Focused tests:

`python -m pytest -q tests/test_d03_ecfr_acquisition_probe.py`

The validator intentionally rejects:

- drift from the official API documentation/title endpoint and the frozen title-metadata observation;
- weakening the metadata-date or reserved-title successor boundary;
- mutable/current acquisition authority;
- any nonzero source/corpus/family/training credit at probe stage;
- title/agency family inflation;
- public-availability rights shortcuts;
- removal of incorporated-by-reference / third-party exclusions;
- skipped dedup/decontamination/split/pack/unique-loss gates;
- tokenizer, model-training, final-test, or paid-compute claims;
- a dedicated workflow addition while repository Actions fanout is being remediated.

## Truth boundary

Status: `DISCOVERY_ONLY_ZERO_CREDIT`.

`LOCAL_FREE` only. Bulk acquisition is not executed by this package. Canonical capacity is 0 bytes. Training-authorized bytes are 0. Authorized unique loss positions are 0. Optimizer updates are 0. Research Corpus V1 is not released. No learned ~20M or ~100M checkpoint is claimed.
