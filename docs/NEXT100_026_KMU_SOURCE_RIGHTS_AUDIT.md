# NEXT100-026 Ukrainian Cabinet/Secretariat source rights audit

Worker: `NEXT100-026-DATA-UA-CABINET-MINISTRY`

Verdict: **ADMIT** a bounded immutable pilot snapshot for **pretraining only**.

## Family boundary

Family ID: `ua.kmu.portal.secretariat-news`.

Publisher lineage is the Department of Information and Public Communications of the Secretariat of the Cabinet of Ministers of Ukraine on `kmu.gov.ua/news/`. This is independent from the incumbent Verkhovna Rada family `ua.rada.open-data.laws-texts`: different publisher/origin namespace and non-normative public-information prose. Cabinet normative acts are deliberately excluded to avoid Rada document-lineage overlap. Ministry-authored syndicated items are also excluded because ministry sites can carry different reuse terms.

## Rights

The Cabinet portal states that all content is available under CC BY 4.0 unless otherwise noted. CC BY 4.0 permits sharing and adaptation for any purpose, including commercial use, subject to attribution. The project therefore maps model-training transformation to `ALLOWED_PRETRAINING` and redistribution to `ALLOWED_WITH_ATTRIBUTION`.

Attribution must retain: Secretariat of the Cabinet of Ministers of Ukraine / `kmu.gov.ua`, source URL, CC BY 4.0 link, snapshot date, and an indication that article-body extraction/normalization was performed.

Negative control: the Ministry of Economy and Environment site uses CC BY-NC-ND 4.0. It is **REJECTED** from this family and proves that government provenance alone is not sufficient rights evidence.

## Immutable bounded materialization

Six Secretariat-authored news/public-information items are stored as raw article-body extractions; normalized identities are deterministically derived and hash-bound. Aggregate normalized size: **9153 bytes**. Manifest identity: `1f068e6cc5ce3fc4a51d8477acee31fab5a0178e15f49225b57de94c5178f7d9`.

Normalization: NFKC, collapse intra-line whitespace, remove blank lines, LF final newline.

The snapshot does not claim the mutable live HTML itself is immutable. The repository-retained extracted bytes, their source URLs/publication timestamps, and SHA-256 identities are the immutable training authority.

## Privacy, quality, language, dedup

Contact/request/submission pages are excluded. Selected records contain no email addresses or phone numbers. Public-official names appear in an official-duty context; no sensitive personal data was detected.

All six records contain substantive Ukrainian administrative/public-information prose, have >=60 words, Cyrillic-letter ratio >=0.99, and >=20 Ukrainian-specific letters.

Raw and normalized hashes are unique within the snapshot and do not equal any normalized hash in DATA-229. Maximum pairwise 5-word-shingle Jaccard is <0.02. Normative acts are excluded to avoid cross-lineage duplication with Rada.

## Evaluation boundary

`evaluation=NOT_SEPARATELY_ADMITTED`; `final_test=PROHIBITED`. The DATA-229 reserved-fingerprint registry at the bound base contains zero sets, but this source is still not granted evaluation use.

## Authority boundary

This PR is a rights + immutable-source admission authority. It does **not** rewrite the canonical real-snapshot registry. A registry successor may consume this family only after its own live concurrency check and normal corpus quality/privacy/dedup gates.
