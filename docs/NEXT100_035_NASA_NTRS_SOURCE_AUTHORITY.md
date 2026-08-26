# NEXT100-035 NASA NTRS scientific/technical source qualification

Worker: `NEXT100-035-DATA-EN-NASA`

## Candidate family

`nasa.sti.ntrs.usgov-technical-abstracts`

This authority is deliberately narrower than the NASA web domain and narrower than an NTRS publication family. It can admit only the deterministic `title` + `abstract` text returned by the NASA Technical Reports Server citation API for the fixed document-ID list in `configs/data/next100_035_nasa_ntrs_usgov_abstracts_v1.json`.

PDF bodies, full-text downloads, images, figures and tables contribute zero language-capacity bytes under this authority.

## Rights boundary

NASA STI's own copyright guidance says U.S.-government works are generally not copyrighted under 17 U.S.C. 105 while also warning that NASA-hosted works can contain privately created copyrighted material, contractor or grantee works, and joint works with retained private rights. NASA STI also states that document/STI metadata carries additional copyright information.

Therefore domain ownership, publication type, or a generic NASA attribution is never enough for admission. Each exact NTRS record must pass all document gates:

- public distribution;
- `copyright.determinationType` is one of the predeclared public-use determinations;
- `copyright.licenseType == NO`;
- `copyright.containsThirdPartyMaterial == false`;
- `copyright.belongsToContractor == false`;
- `copyright.belongsToPublisher == false`;
- every author affiliation is NASA civil service or explicitly NASA-affiliated;
- no EAR/ITAR or sensitive-information flag;
- no third-party textual permission/copyright marker in the retained title/abstract.

Missing or ambiguous metadata fails closed.

Model-training use is admitted only for records satisfying every gate. Redistribution of the bounded title+abstract snapshot is admitted only with NASA STI/NTRS source acknowledgment and without implying NASA endorsement. Evaluation remains `NOT_SEPARATELY_ADMITTED`.

## Exact identity and extraction

The acquisition endpoint is `https://ntrs.nasa.gov/api/citations/{document_id}`. The fixed candidate IDs are sorted and enumerated in the config rather than discovered dynamically.

For each record the qualifier binds:

- NTRS document ID and citation/API URLs;
- NTRS `modified` value;
- SHA-256 and byte count of the HTTP response body;
- canonical sorted-JSON record SHA-256 and byte count;
- retained normalized text SHA-256 and byte count.

The training payload is only `title + "\n\n" + abstract + "\n"`. Normalization `NASA_NTRS_TITLE_ABSTRACT_NFKC_WS_V1` performs HTML entity unescape, Unicode NFKC, newline canonicalization and deterministic whitespace normalization. It performs no OCR, summarization, paraphrase or table reconstruction.

A terminal authority is valid only after the first live probe is sealed into the config `pins` and a fresh exact-head `verify` run reproduces every admitted ID, modification marker and content hash.

## Quality, privacy and dedup

Every retained record must have at least 80 word tokens, at least 99% printable characters and at least 55% alphabetic characters. Image-only and table-only documents cannot pass by contributing non-text bytes because those media are never acquired into the training payload.

Author names, affiliations and the rest of NTRS metadata are provenance only and are not part of training text. The retained title+abstract is scanned for email, U.S.-phone and SSN-like patterns; a match rejects the record.

Within this family, exact normalized duplicates and 5-token-shingle Jaccard near-duplicates at or above 0.85 fail the terminal family authority. Before incorporation into a successor corpus registry, the admitted objects still require the project-wide exact/near-copy audit against all other external-real families.

## Family lineage

All admitted records receive exactly one independent-family credit: `nasa.sti.ntrs.usgov-technical-abstracts`. Different NASA centers, NTRS document IDs, mirrors, URLs, publication types, or multiple records do not manufacture additional family independence.

## LOCAL_FREE boundary

The qualifier is Python-standard-library only. It performs bounded metadata acquisition and hashing, not model training. `training_executed` must remain false in terminal evidence.
