# NEXT100-027 — Ukrainian Public-Domain Literature Admission

Worker: `NEXT100-027-DATA-UA-PUBLIC-DOMAIN-LIT`

Verdict: **ADMIT**, bounded snapshot only.

## Admitted object

This authority admits exactly 24 Ukrainian proverb records from Matvii Nomys's 1864 collection *Українські приказки, прислів'я і таке інше*, as represented in the pinned Verba corpus landing object. It does not admit the full Verba corpus and it excludes all Bobkova 1961, Mlodzynskyi 2009, Franko 1901 and Ilkevych 1841 rows from the materialized payload.

Deterministic selector: traverse `app/public/data/landing.json` at Verba commit `34a2c10ac35e1febad6c270a88fc8b83790407da` in array order and retain the first 24 records whose `sources` value is exactly `["Nomis1864"]`.

Upstream data object Git blob: `a8e31fd41bd3dbbde7d43ec3c04f56e5beb37d1b`.

## Rights

Underlying work: public domain. The work was published in 1864. Matvii Nomys died 1901-01-08. Current Ukrainian copyright law sets the ordinary property-right term at 70 years from 1 January following the author's death; the compiler term therefore expired no later than 1972-01-01. The edition is also a collection of historical folk material.

Digital scan: the corresponding faithful scan is marked public domain on Wikimedia Commons, sourced from Internet Archive item `nomis1864`.

Digital transcription/compilation: Verba's exact Data Card at commit `34a2c10...` classifies Nomis 1864 historical text as public domain and licenses the unified corpus structure/enrichment under CC BY 4.0. CC BY 4.0 permits sharing and adaptation for any purpose, including commercial use, with attribution.

Required attribution for redistribution of the Verba digital layer:

`Yemelianov, Dmytro (2026). verba — Ukrainian Proverbs Corpus (v1.0.2). https://verbacorpus.org`

No LLM-generated `modern_text`, categories, explanations or variant metadata are included in this training payload.

## Snapshot identities

Raw JSONL: `492d985fd5c364fecc76a4ef387bdcbda936ebf47769e6fff011f113411d9b3b`, 2859 bytes.

Normalized text: `1eb91dbd631898c6a2efe274b700a5be0deaca243c0a9d5d30994ddadcf43598`, 1659 bytes.

Authority identity: `85f596e79b0ec6479d2ef815e2a6a9bdbfaa55993c797309c1ea4d93b1d9b0e7`.

Normalization is deliberately conservative: NFC + outer-whitespace trim only, preserving case, punctuation and historical spelling.

## Language / quality / privacy

All 711 alphabetic code points in the bounded normalized snapshot are Cyrillic; 31 Ukrainian-specific `і/ї/є/ґ` occurrences are present.

Verba warns that the full Nomis OCR is roughly 75–80% character fidelity. For that reason this authority does **not** admit all 9,785 Nomis rows. The admission scope is the committed 24-row bounded snapshot only. Downstream corpus construction must retain project quality gates.

The payload is historical folk text only and carries no modern contact fields or personal-record dataset fields.

## Dedup and evaluation firewall

The 24 record ids are unique and the 24 normalized text records are exact-unique.

Live DATA-293 training families at the admission base are Rada, Standard Ebooks, HTTPX and Requests; `ua.verba.public-domain.nomis1864` is not one of them.

EVAL-233 final-test lineages are Rada/Standard-Ebooks. EVAL-303 selection-validation lineages are Kubernetes Ukrainian docs, Lang-UK Perestoroha OCR, HTTPX and Requests. The Verba/Nomis family is disjoint. This worker does not read final-test or selection-validation payload text.

No near-copy/cluster decontamination claim is made. A successor composed corpus must rerun project G07/G08 on exact bytes.

## Execution boundary

`LOCAL_FREE` only. No model training. No final-test outcome access. This is a source/snapshot admission authority, not a DATA-300 corpus freeze or repository release-green claim.
