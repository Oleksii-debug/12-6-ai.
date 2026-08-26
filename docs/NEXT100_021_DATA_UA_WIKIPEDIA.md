# NEXT100-021 Ukrainian Wikipedia source-family authority

Worker: `NEXT100-021-DATA-UA-WIKIPEDIA`

Terminal decision: `REJECT` for current model-training admission.

This change does not admit Ukrainian Wikipedia, does not create a training snapshot, does not reserve evaluation material, and does not mutate the corpus contract or admitted-source registry.

## Authoritative upstream and exact candidate identity

Upstream is Wikimedia Foundation's official dump service. The exact candidate is:

- dump family: `ukwiki`
- dump date: `20260801`
- object: `ukwiki-20260801-pages-articles1.xml-p1p194007.bz2`
- exact dated URL: `https://dumps.wikimedia.org/ukwiki/20260801/ukwiki-20260801-pages-articles1.xml-p1p194007.bz2`
- reported compressed size: `166733586` bytes
- official upstream SHA-1: `e6aa53cf981f53807ca59d4fc7ab8e1d97151461`

The dated object identity is used for the candidate. Mutable `latest` is used only to acquire the small official checksum manifest in the component proof; blocked Wikipedia content bytes are not fetched.

## Rights decision

Ukrainian Wikipedia text is reviewed as `CC-BY-SA-4.0`. License-level reuse is broad, including commercial reuse/adaptation, but project admission is a separate decision. Wikimedia reuse obligations include contributor attribution, applicable additional attribution requirements for imported text, modification notices, license notice/link or copy, ShareAlike when triggered, and no incompatible downstream legal or technical restrictions.

The repository's incumbent candidate authority already marks `ua.wikipedia.ukwiki` `BLOCKED_BY_RIGHTS` / `REVIEW_REQUIRED`, with `allows_model_training=false`, because the project has no approved ShareAlike compliance policy for training artifacts/model outputs. No terminal successor policy resolving that blocker was found at cutoff. This worker therefore does not promote the license label into project-purpose permission.

Purpose decisions:

- model training: `REJECT`
- redistribution of a project-normalized Wikipedia snapshot: `NOT_ADMITTED`
- evaluation: `NOT_SEPARATELY_ADMITTED`

Retest requires an explicit owner-approved or terminal project ShareAlike compliance policy before materializing Wikipedia content.

## Privacy / PII

Wikipedia article text can contain personal information about living or otherwise identifiable people. Public availability does not remove privacy, publicity, moral-right, or other non-copyright risk. Because the rights gate fails before content materialization, the exact-content privacy/PII gate is `NOT_RUN_RIGHTS_GATE_FAILED`; it becomes mandatory on any future retest.

## Bounded LOCAL_FREE acquisition

The current exact-head workflow performs only a bounded upstream-metadata probe:

- maximum network metadata bytes: `1048576`
- resource: official Wikimedia SHA-1 manifest
- purpose: verify that the exact dated dump object is bound to the published SHA-1
- Wikipedia content payload bytes: exactly `0`

The previously considered 170 MB content probe is not current authority and is intentionally superseded by rights-first fail-close behavior. No raw SHA-256 or normalized SHA-256 is invented for content that was not materialized.

## Deterministic normalization contract

A future retest is pre-bound to `UKWIKI_WIKITEXT_NFKC_LINES_V1`: namespace `0`, non-redirect current revisions in dump order, Unicode NFKC, canonical newlines, removal of Unicode `Cf`, line-end trim, outer trim, UTF-8, canonical JSONL bound to page/revision IDs. The exact-head component validates this normalizer on a deterministic project-owned fixture only. It does not normalize blocked Wikipedia bytes.

Wikitext markup is to remain preserved at the rights-retest stage so imported attribution/source notices are not silently stripped before policy review.

## Family and dedup

Candidate family identity: `wikimedia:wikipedia:ukwiki`.

It is distinguishable from incumbent Ukrainian family `ua.rada.open-data.laws-texts`, but receives `independent_family_credit=0` while rejected. Existing exact/normalized/near-copy/fragment and lineage semantics are compatible with a future materialized candidate. Cross-family dedup is `NOT_RUN_RIGHTS_GATE_FAILED` and must run on exact normalized bytes before any future admission.

## Current registry boundary

The branch is stacked on DATA-287 exact green head `b0523ccbc4b957615aac849d476cfa851be87578`. The committed machine registry identity there is `917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c`. Ukrainian Wikipedia is not in the admitted snapshot set. Its historical candidate remains `REVIEW_REQUIRED`.

A stale prose registry identity in the DATA-287 PR body is not used over the exact machine registry bytes.

## Terminal root cause

`PROJECT_HAS_NO_APPROVED_SHAREALIKE_COMPLIANCE_POLICY_FOR_TRAINING_ARTIFACTS_OR_MODEL_OUTPUTS`

Consequences at this authority: no content materialization, no training snapshot, no redistribution admission, no evaluation reservation, no privacy/dedup execution on source bytes, and no model training.
