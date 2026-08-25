# DATA-21 / DATA-22 external source intake

## Scope and incumbent reuse

This package starts from the exact DATA-10 head `077205ef2b1662a5029bc77b8fc762078cabeb17`. It does not create a second corpus framework. Acquisition/extraction is a thin front-end; normalization and language/script validation use `twelve_six.data.multilingual_pretraining`, exact dedup uses the D03 `SQLiteExactDedupIndex`, and the scale handoff remains DataTrove `0.10.0`.

The canonical D03 registry `data/external/external_sources.json` remains unchanged until bytes are copied into controlled immutable storage and represented by `SnapshotSpec` with exact byte size/SHA-256. The bounded CI artifact is real-source evidence, not a canonical snapshot URI or corpus freeze.

## Rights decision rule

`configs/data/external_source_candidates_ua_en_v1.json` is a machine-readable candidate/rights registry, not the canonical external-source registry. `ELIGIBLE` requires the existing D03 rights status `APPROVED_FOR_TRAINING` and `allows_model_training=true`. A source marked `BLOCKED_BY_RIGHTS` has neither an acquisition adapter nor acquisition URLs, so the intake runner cannot download it accidentally.

The current candidate set has 8 source candidates: 2 eligible and 6 blocked by rights. Of the blocked sources, 1 is explicitly rejected and 5 require further review.

### Ukrainian eligible source

`ua.rada.open-data.laws-texts` is based on the Verkhovna Rada open-data dataset "Тексти первинних законів бази даних «Законодавство України»". The publisher's dataset terms state that open data may be freely used, reused and redistributed, including commercial use. The source is also limited to official legislative acts; Article 8 of Law of Ukraine No. 2811-IX excludes official legislative/administrative/judicial acts from copyright protection. The bounded adapter downloads one explicitly listed primary-law HTML document rather than the complete archive.

Evidence references:

- `https://data.rada.gov.ua/open/data/laws-texts`
- `https://data.rada.gov.ua/ogd/zak/perv/text/d23314.htm`
- `https://zakon.rada.gov.ua/laws/show/2811-20`

This project rights decision applies only to the registered official-law source/version and does not generalize to arbitrary material on Rada domains.

### English eligible source

`en.standardebooks.manual` is pinned to Standard Ebooks Manual of Style git revision `d1143a9b459b5e6f9cdda93a7c1e04676bff4f6b`. The repository `LICENSE.md` at that revision states that repository contents are released under CC0 1.0, except `build-manual.py`, which is GPLv3. The bounded acquisition allowlist contains only two `.rst` content files and never includes the GPL exception.

Evidence references:

- `https://github.com/standardebooks/manual/commit/d1143a9b459b5e6f9cdda93a7c1e04676bff4f6b`
- `https://raw.githubusercontent.com/standardebooks/manual/d1143a9b459b5e6f9cdda93a7c1e04676bff4f6b/LICENSE.md`

## Fail-closed candidates

- Ukrainian Wikipedia: `REVIEW_REQUIRED`; CC-BY-SA-4.0 ShareAlike compliance for model/training artifacts has not been reviewed by project policy.
- Ukrainian Wikisource: `REVIEW_REQUIRED`; per-work copyright and translation status is heterogeneous and has not been resolved to immutable works.
- Ukrainian mC4/Common-Crawl-derived material: `REVIEW_REQUIRED`; crawl availability does not establish page-level model-training rights.
- OpenStax Psychology 2e: `REJECTED`; the provider page explicitly prohibits use for training/ingestion into LLM or generative-AI offerings without permission.
- GovInfo/U.S. Code candidate: `REVIEW_REQUIRED`; U.S.-government public-domain rules alone are not treated as a reviewed worldwide training-rights decision and GovInfo warns that third-party copyrighted material may appear.
- English Wikipedia: `REVIEW_REQUIRED`; same unresolved ShareAlike policy boundary as Ukrainian Wikipedia.

No blocked source is fetched by the runner.

## Bounded real intake

Reproduction command:

```bash
PYTHONPATH=src python tools/run_external_source_intake.py \
  --output external-source-intake-evidence \
  --max-download-bytes 2000000 \
  --max-normalized-chars 50000
```

For each allowlisted real object the runner records deterministic document ID, source/version identity, acquisition URL, raw byte size and SHA-256, decoded encoding, normalized SHA-256, normalized UTF-8 byte count, incumbent language label/confidence/reason, rights status/license, and accepted text path. Exact normalized duplicates are rejected through the incumbent SQLite D03 index.

The manifest separately reports candidate/eligible/blocked source counts and attempted/accepted/rejected record counts. DataTrove MinHash near-dedup is intentionally not claimed as executed for this bounded sample; the manifest retains `NOT_RUN_BOUNDED_SAMPLE` and the existing DataTrove `0.10.0` handoff identity.

## Promotion boundary

A successful bounded run creates real training-eligible bytes under the reviewed source policy, but does not by itself satisfy D03 canonical snapshot promotion. Promotion requires immutable controlled-storage snapshot URIs, exact snapshot hashes/sizes, registry conversion into `ExternalSourceSpec`, record-policy hook evidence, reserved-fingerprint/decontamination checks, and the existing downstream D03/D04/D06/D10 audit gates.
