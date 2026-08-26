# D03 eCFR versioned source probe

## Purpose

This package opens a fail-closed English acquisition prospect for Research Corpus V1 using the Electronic Code of Federal Regulations (eCFR). It is a discovery and successor-request contract only. It does not download bulk data, admit corpus bytes, fit a tokenizer, train a model, or authorize paid compute.

The project is data-bound before the first learned MODEL-341 ~20M campaign. eCFR is relevant because the official federal publishing stack exposes structured regulatory XML and point-in-time access. That makes exact version binding possible, unlike treating a mutable current page as an immutable training artifact.

## Official source boundary

The source contract binds:

- portal: `https://www.ecfr.gov`;
- eCFR API documentation: `https://www.ecfr.gov/developers/documentation/api/v1`;
- titles metadata endpoint: `https://www.ecfr.gov/api/versioner/v1/titles.json`;
- historical full-title template: `https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml`;
- GovInfo developer hub: `https://www.govinfo.gov/developers`;
- GovInfo eCFR XML user guide: `https://www.govinfo.gov/bulkdata/ECFR/resources/ECFR-XML-User-Guide.pdf`.

The eCFR is updated as regulatory changes are incorporated. Therefore a mutable/current endpoint is discovery material, not a corpus identity. A successor must use an exact historical date and title, then freeze the returned bytes with exact hashes and byte counts.

## Rights and provenance boundary

17 U.S.C. §105 generally withholds United States copyright protection from works of the United States Government, but that is not a blanket rule for every object a government system can host or reference. The United States may hold transferred copyrights, statutory exceptions exist, and third-party or contractor material may require a separate basis.

For this project the conservative rule is therefore:

1. public availability does not equal training permission;
2. government hosting does not automatically make every embedded or incorporated object a United States Government work;
3. incorporated-by-reference material, third-party/contractor text, transferred-copyright material, images/media, external attachments, and unknown provenance remain excluded until independently classified;
4. issuing-agency/source context must be preserved so later rights decisions are auditable;
5. this probe grants zero family credit, zero training bytes, and zero training-authorized loss positions.

Primary statutory reference: `https://www.copyright.gov/title17/92chap1.html`.

## Deterministic operator use

Validate the frozen zero-credit contract from the repository root:

```bash
python tools/validate_d03_ecfr_versioned_probe.py
```

Build a deterministic successor request envelope for one historical title without performing network access:

```bash
python tools/validate_d03_ecfr_versioned_probe.py --date 2026-08-25 --title 12
```

The request envelope contains the exact historical URL plus the frozen source-contract identity. It still has `family_credit = 0`, `training_authorized_bytes = 0`, and all downstream scientific gates set to `NOT_RUN`.

## Required successor materialization

A later acquisition worker may only move beyond the probe after all of the following are satisfied:

1. select an exact historical date and non-reserved title;
2. acquire the exact request twice independently and require byte-identical results;
3. record raw SHA-256, exact byte count, media type, request URL/date/title, and acquisition evidence;
4. parse XML with DTD/external-entity/network resolution disabled and bounded input sizes;
5. classify rights/provenance at a sufficiently fine record/subrecord level to exclude ambiguous or third-party material;
6. deterministically extract and normalize eligible regulatory text while preserving source/agency lineage;
7. run quality/privacy review;
8. run global exact, normalized, near-copy, fragment, and lineage deduplication against every admitted training family;
9. run evaluation-reservation decontamination before any training split;
10. rerun family-cap and stratum-balance gates;
11. perform cluster-safe split, deterministic shard/pack, and two byte-identical clean builds;
12. compute exact post-pack unique causal-loss positions;
13. obtain separate tokenizer, checkpoint/evaluation, compute, and training authorities.

Source bytes are never interchangeable with tokenizer tokens or unique causal-loss positions. Repetition/replay may be an explicitly measured optimization choice later, but cannot manufacture unique-data capacity.

## Security boundary

The probe contract forbids DTDs, external entities, and parser network resolution. This is intentional XML/XXE hardening. A successor materializer must preserve those restrictions and enforce byte/node ceilings before parsing untrusted upstream XML.

## Role in the 20M → 100M → 1B path

eCFR is a potentially high-yield English source family, not the whole corpus solution. It does not repair Ukrainian or code-stratum deficits and should not delay higher-priority terminalization of already-active Ukrainian and code bulk lanes. Its main value is to add a versionable, structured, independently governed English reservoir that can become useful as the corpus grows beyond the first learned ~20M campaign toward later ~100M/1B experiments.

The project must still complete source convergence, global deduplication, decontamination, quality/privacy, balance, split/pack, unique-loss accounting, checkpoint/recovery qualification, learned-ladder verification, bounded smoke training, and explicit material-compute authorization before a long learned MODEL-341 run.

## Truth boundary

`LOCAL_FREE_DISCOVERY_AND_SUCCESSOR_REQUEST_ONLY`.

This package proves only that the eCFR prospect is represented by a deterministic, fail-closed, zero-credit contract. It does not prove rights eligibility for any fetched object, does not create a corpus, and does not authorize tokenizer fitting, model training, GPU provisioning, paid compute, model promotion, or any learned-model quality claim.
