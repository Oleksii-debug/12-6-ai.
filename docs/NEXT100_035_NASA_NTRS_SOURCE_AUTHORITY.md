# NEXT100-035 NASA NTRS scientific/technical source qualification

Worker: `NEXT100-035-DATA-EN-NASA`

## Candidate family

`nasa.sti.ntrs.usgov-technical-abstracts`

This authority is deliberately narrower than the NASA web domain and narrower than an NTRS publication family. It can admit only the deterministic `title` + `abstract` metadata returned by the NASA Technical Reports Server citation API for the fixed document-ID list in `configs/data/next100_035_nasa_ntrs_usgov_abstracts_v1.json`.

PDF bodies, full-text downloads, images, figures and tables contribute zero language-capacity bytes under this authority.

## Rights boundary

NASA STI's own copyright guidance says U.S.-government works are generally not copyrighted under 17 U.S.C. 105 while also warning that NASA-hosted works can contain privately created copyrighted material, contractor or grantee works, and joint works with retained private rights. NASA STI also states that document/STI metadata carries additional copyright information.

Therefore NASA-domain location, publication type, or generic attribution is never enough. Each retained title+abstract record must satisfy all of these gates:

- public distribution;
- `copyright.determinationType == GOV_PUBLIC_USE_PERMITTED`;
- `copyright.licenseType == NO`;
- no positive `copyright.containsThirdPartyMaterial` flag;
- no positive `copyright.belongsToContractor` flag;
- no positive `copyright.belongsToPublisher` flag;
- every author affiliation is NASA civil service or explicitly NASA-affiliated;
- no EAR/ITAR restriction and `sensitiveInformation` is the NTRS public value `2`/NONE (or absent in a legacy public record);
- no third-party textual permission/copyright marker in the retained title/abstract.

NASA's API examples may omit nested false-valued copyright booleans. This authority does not reinterpret missing body-level exclusions as permission for the full document. Instead, the retained payload remains only the government-determined, NASA-civil-authored title+abstract metadata. Any positive third-party/contractor/publisher flag rejects even that metadata record. The PDF/full-text body remains `NOT_ADMITTED_BY_THIS_AUTHORITY` unless a separate authority binds explicit body-level rights; even records with explicit false body flags do not expand this source's payload.

Model-training use is admitted only for records satisfying the retained-metadata gates. Redistribution of the bounded title+abstract snapshot is admitted only with NASA STI/NTRS source acknowledgment and without implying NASA endorsement. Evaluation remains `NOT_SEPARATELY_ADMITTED`.

## Exact identity and extraction

The acquisition endpoint is `https://ntrs.nasa.gov/api/citations/{document_id}`. The fixed candidate IDs are enumerated in the config rather than discovered dynamically.

For each record the qualifier binds:

- NTRS document ID and citation/API URLs;
- NTRS `modified` value;
- SHA-256 and byte count of the HTTP response body;
- canonical sorted-JSON record SHA-256 and byte count;
- retained normalized text SHA-256 and byte count;
- the exact copyright metadata and the separate full-document-body rights state.

The training payload is only `title + "\n\n" + abstract + "\n"`. Normalization `NASA_NTRS_TITLE_ABSTRACT_NFKC_WS_V1` performs HTML entity unescape, Unicode NFKC, newline canonicalization and deterministic whitespace normalization. It performs no OCR, summarization, paraphrase or table reconstruction.

A terminal ADMIT authority is valid only after a live probe is sealed into config `pins` and a fresh exact-head `verify` run reproduces the authority identity, family identity, admitted IDs, NTRS modification markers, canonical-record hashes and normalized-text hashes.

## Quality, privacy and dedup

Every retained record must have at least 80 word tokens, at least 99% printable characters and at least 55% alphabetic characters. Image-only and table-only documents cannot contribute language bytes because those media are never acquired into the training payload.

Author names, affiliations and the rest of NTRS metadata are provenance only and are not part of training text. The retained title+abstract is scanned for email, U.S.-phone and SSN-like patterns; a match rejects the record.

Within this family, exact normalized duplicates and 5-token-shingle Jaccard near-duplicates at or above 0.85 fail the terminal family authority. Before incorporation into a successor corpus registry, admitted objects still require the project-wide exact/near-copy audit against all other external-real families.

## Family lineage

All admitted records receive exactly one independent-family credit: `nasa.sti.ntrs.usgov-technical-abstracts`. Different NASA centers, NTRS document IDs, mirrors, URLs, publication types, or multiple records do not manufacture additional family independence.

## LOCAL_FREE boundary

The qualifier is Python-standard-library only. It performs bounded metadata acquisition and hashing, not model training. `training_executed` must remain false in terminal evidence.
