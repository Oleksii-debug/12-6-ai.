# NEXT100-031 — English Wikipedia source authority

Worker: `NEXT100-031-DATA-EN-WIKIPEDIA`

Terminal verdict: **REJECT** for model-training admission under the current project policy.

## Exact upstream candidate

The authoritative source candidate is the Wikimedia English Wikipedia `20260801` dump. The bounded retest anchor is the first articles shard:

- file: `enwiki-20260801-pages-articles1.xml-p1p41242.bz2`
- official SHA-1: `97ea1ad5a871e951ddadaaf199f69b1ddf121b34`
- reported compressed size: 284.6 MB
- source family: `en.wikipedia.enwiki`
- language: English
- family lineage: independent from `en.standardebooks.manual`; it counts as one family only and cannot support a broad representativeness claim by itself.

## Rights finding

English Wikipedia community text is generally reusable under CC BY-SA 4.0. The upstream terms allow commercial reuse and adaptation subject to license conditions. Redistribution must preserve required attribution, identify/link the license, indicate modifications, and respect ShareAlike and any additional attribution obligations attached to imported text. Page-specific exceptions and other rights can still apply.

This upstream permission does **not** automatically satisfy the repository's own admission policy. The incumbent candidate registry already records `en.wikipedia.enwiki` as `BLOCKED_BY_RIGHTS` because the project has no approved ShareAlike compliance policy for training artifacts/model outputs. This worker does not silently reverse that policy decision.

Purpose decisions:

- model training: `REJECT`
- redistribution: `NOT_ADMITTED`
- evaluation: `NOT_SEPARATELY_ADMITTED`

The authority does not decide whether distributed model weights or arbitrary model outputs are adaptations under copyright law. That unresolved classification is precisely why the existing project ShareAlike gate remains fail-closed.

## Materialization / normalization

No Wikipedia bytes were fetched. `network_bytes_downloaded=0`; raw and normalized SHA-256 are therefore intentionally null. A rejected source must not be materialized merely to manufacture hashes.

If the policy blocker is later resolved, the retest must use a bounded deterministic acquisition: pin the dump and checksum/index boundary, take one multistream block of 100 pages, retain only namespace-0 non-redirect current revisions, cap normalized output at 1 MiB without partial-page truncation, and retain page ID, revision ID, revision timestamp, raw SHA-256 and normalized SHA-256.

## Language, quality, privacy, dedup

These content gates were not run because the rights gate failed first. Any future retest must pass the incumbent English language-ID gate, deterministic quality filtering, PII/privacy screening, exact and near dedup, and contamination checks. Wikipedia can contain biographies and other public personal data, so source availability is not a privacy clearance.

The current external snapshot registry has one English training family, `en.standardebooks.manual`; Wikipedia is not counted. Evaluation/final-test/selection material was not used or consumed.

## Retest conditions

Retest only after an owner-approved or terminal project policy explicitly resolves CC BY-SA attribution/ShareAlike compliance for training artifacts and model-output/weight distribution claims. Then materialize immutable bounded bytes, retain hashes, pass language/quality/privacy/dedup/contamination gates, and refresh the live registry before any admission.

Machine authority: `configs/data/next100_031_en_wikipedia_source_authority_v1.json`

Authority identity SHA-256: `c7cd80788b5143f90331b72b2dbaba148acabbea225a383557571c77c8c23204`
