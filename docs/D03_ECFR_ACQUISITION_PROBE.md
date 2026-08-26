# D03 eCFR Versioned Acquisition Probe V1

## Purpose

This package opens a discovery-only acquisition lane for a large English regulatory-text prospect from the official Electronic Code of Federal Regulations (eCFR). It is deliberately a zero-credit probe. It does not materialize a corpus object, grant training rights, fit a tokenizer, or authorize model training.

The learned ~20M campaign is currently data-gated. Adding another small hand-picked document would not materially change that bottleneck. eCFR is attractive because the Office of the Federal Register exposes point-in-time data and GPO publishes structured XML, so a successor can acquire exact title/date objects and bind their bytes instead of scraping moving HTML.

## Official technical basis

The eCFR Developer Resources state that eCFR data originates from files used for print/PDF production, is transformed into GPO bulk XML, and is processed into point-in-time units. The API requires no key.

This probe binds:

- title metadata: `https://www.ecfr.gov/api/versioner-import/v1/titles`
- historical full-title template: `https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml`
- conservative source ID: `en.us.ecfr.regulations`
- conservative family ID: `us.federal-regulations.ecfr`

The 2026-08-26 discovery observation saw the titles metadata at `meta.date=2026-07-31`, `import_in_progress=false`, titles 1-50 with title 35 reserved. This observation is not an immutable data authority. A successor must select explicit historical date/title objects and verify exact response bytes.

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

1. Query title metadata only to choose an already completed point-in-time date.
2. Select a bounded set of exact `(date, title)` objects.
3. Fetch each exact historical XML object twice independently.
4. Reject redirects to unexpected origins, wrong content type, malformed XML, byte drift, or unequal repeat acquisitions.
5. Seal raw SHA-256 and byte count per exact object.
6. Parse XML deterministically and preserve enough structural provenance to trace every emitted text record back to title/chapter/part/section.
7. Execute rights/provenance classification before granting any training eligibility.
8. Apply language/quality/privacy filtering.
9. Run global exact, near, fragment, and lineage dedup against the live corpus authority.
10. Apply evaluation-reservation decontamination.
11. Recompute balance and family caps without replay.
12. Create cluster-safe splits, deterministic tokenization/packing, and two clean byte-identical builds.
13. Count exact post-pack non-ignored unique causal-loss positions.
14. Only after tokenizer, D05 checkpoint, evaluation and compute gates are terminal may a learned campaign consume the resulting exposure.

No step in this document turns source bytes into token-budget or training authorization.

## Validation

Run locally:

`python tools/validate_d03_ecfr_acquisition_probe.py`

Focused tests:

`python -m pytest -q tests/test_d03_ecfr_acquisition_probe.py`

The validator intentionally rejects:

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
