# NEXT100-022 Ukrainian Wikisource qualification

Worker: `NEXT100-022-DATA-UA-WIKISOURCE`

Execution class: `LOCAL_FREE`

Terminal source-qualification verdict: `ADMIT_BOUNDED_PD_EDITION_SNAPSHOT`.

This authority is intentionally narrow. It admits one immutable, proofread public-domain edition page as a rights-qualified source snapshot. It does not admit Ukrainian Wikisource as a whole, and it does not make the snapshot eligible for corpus training selection until the standard DATA-232/DATA-299 near-match decontamination gate has run.

## Exact provenance

The retained text is the approved Ukrainian Wikisource page `Сторінка:Леся Українка. На крилах пісень. 1892.pdf/13`, permanent revision `560107`, belonging to index `Індекс:Леся Українка. На крилах пісень. 1892.pdf`, permanent revision `729499`. The index identifies Lesya Ukrainka, *На крилах пісень*, 1892, Lviv, “З друкарні Товариства імени Шевченка”, and reports that all index pages are verified and the work is fully included.

The underlying scan is Wikimedia Commons file `Леся Українка. На крилах пісень. 1892.pdf`, 112 pages, 2,069,409 bytes, SHA-1 `4ae5ba96e7e76d7fb26b37d5277a2e82c1443407`, sourced by Commons from Google Books id `VuD74cFPso0C`. Commons marks the work public domain and states that it is public domain in the United States because it was published before 1 January 1931. The Commons description has a `Date=1902` metadata field that conflicts with the Wikisource index and filename; this field is not used as the edition publication identity.

## Rights decision

Underlying literary work/edition: `PUBLIC_DOMAIN`. Lesya Ukrainka died 1 August 1913. Article 31 of Ukraine Law No. 2811-IX, current WIPO Lex consolidation UA274 dated 31 July 2026, provides the normal life-plus-70 term measured from 1 January following death; that term ended before 1984. Commons independently marks this pre-1931 publication public domain in the United States.

Wikimedia text-layer obligations remain separate from the underlying public-domain status. Ukrainian Wikisource copyright policy revision `911097` says works must be public domain or compatibly free-licensed, distinguishes derivative translations, and says original works/translations are automatically CC BY-SA unless otherwise stated. Wikimedia Terms of Use revision `554852` permits reuse but requires compliance with underlying licenses; for community-developed text pages it requires attribution by page URL/stable copy/author list, and modified/distributed copies require applicable license notice and indication of changes. This authority therefore uses the conservative redistribution state `ALLOWED_WITH_ATTRIBUTION_AND_LICENSE_NOTICE`.

Model-training permission for the exact retained source is `ALLOWED_BY_RIGHTS`. This is a source-use decision, not a conclusion about licensing of model weights or outputs. `model_output_license_inference=NONE`.

Generic Ukrainian Wikisource, modern/original translations, contributor-authored works, and CC-BY-SA-only content are `NOT_ADMITTED_BY_THIS_AUTHORITY`.

## Immutable bounded snapshot

Path: `data/external/snapshots/next100022/ua_wikisource_lesia_1892_page13.txt`

Bytes: `1479`

SHA-256: `65e570c3cd954b595b586554b89a90da6efad0deca6a84d2316937745db17ef2`

Normalization: select only the approved rendered literary body for oldid `560107`; remove MediaWiki chrome and scan image; convert leading NBSP stanza markers into blank-line stanza boundaries; preserve spelling/punctuation/accent semantics; NFC; LF; exactly one final LF. The attribution sidecar is separately hashed and mandatory on redistribution.

## Source-family lineage

Family id: `ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv`.

The family is based on canonical underlying edition/document lineage, not on the `wikimedia.org` host. The Commons scan, the Google Books scan it identifies, and Wikisource transcriptions of the same edition are one family and must not receive duplicate independent-family credit. Mirrors or further transcriptions of this edition alias to this family.

Relative to the consumed DATA-287 registry identity `917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c` at source SHA `b0523ccbc4b957615aac849d476cfa851be87578`, no existing family resolves to this underlying edition; therefore it is independently sourced relative to that registry snapshot. A final live registry refresh is still mandatory before downstream canonical registry admission.

## Evaluation exclusion and privacy

This worker reserves no evaluation material. The retained snapshot SHA-256 is `65e570c3cd954b595b586554b89a90da6efad0deca6a84d2316937745db17ef2`; the DATA-25 NFKC+whitespace-collapse fingerprint is `6712a26753547e9cb26e3993fad001e89f25e66a6fe40032785aa4b493b22ecb`. No exact collision was observed against the consumed public hash-only reservation metadata. Raw evaluation payloads were not inspected. Because the project requires near-match decontamination before corpus construction, `corpus_training_selection` remains `BLOCKED_UNTIL_STANDARD_DATA232_DATA299_NEAR_MATCH_DECONTAMINATION`.

The snapshot contains only the selected 1892 literary body. Contributor usernames/history are not copied; attribution is via the permanent public revision URL, minimizing incidental personal-data collection.

## Verification

Run:

```bash
python tools/validate_next100022_ua_wikisource.py
python -m unittest -v tests.test_next100022_ua_wikisource
```

The validator fails closed on snapshot/notice hash drift, authority identity drift, registry-parent drift, weakened attribution/redistribution conditions, evaluation permission inference, host-based family identity, generic CC-only scope expansion, or premature corpus-training selection.
