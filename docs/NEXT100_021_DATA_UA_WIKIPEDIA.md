# NEXT100-021 Ukrainian Wikipedia source-family evidence

Worker: `NEXT100-021-DATA-UA-WIKIPEDIA`

Terminal decision target: `RETEST`.

This change does not admit Ukrainian Wikipedia for model training, does not create a training snapshot, does not reserve evaluation material, and does not mutate the frozen corpus contract. It records exact upstream identity and runs a bounded analysis-only acquisition proof.

## Authoritative upstream and exact identity

Upstream is Wikimedia Foundation's dump service. The investigated immutable object is:

- dump family: `ukwiki`
- dump date: `20260801`
- object: `ukwiki-20260801-pages-articles1.xml-p1p194007.bz2`
- exact dated URL: `https://dumps.wikimedia.org/ukwiki/20260801/ukwiki-20260801-pages-articles1.xml-p1p194007.bz2`
- expected size: `166733586` bytes
- upstream SHA-1: `e6aa53cf981f53807ca59d4fc7ab8e1d97151461`

The current `latest` index was observed on 2026-08-26 to identify this dump generation, but acquisition uses the dated URL and never the mutable `latest` payload alias.

## Rights decision

Ukrainian Wikipedia text is treated as `CC-BY-SA-4.0` for this review. The license and Wikimedia Terms permit reuse, including commercial reuse, but impose obligations. The project must preserve attribution, applicable additional attribution requirements for imported text, modification notices, license notice/link or copy, and ShareAlike requirements where triggered.

Existing project candidate authority already marks `ua.wikipedia.ukwiki` as `REVIEW_REQUIRED` because there is no approved project ShareAlike compliance policy for training artifacts/model outputs. This worker does not override that decision by inference from the license label.

Purpose decisions remain independent:

- bounded acquisition for analysis: `ALLOWED_FOR_BOUNDED_ANALYSIS_PROBE`
- ephemeral analysis storage: `EPHEMERAL_ANALYSIS_ONLY`
- analysis: `ALLOWED`
- model training: `RETEST_REQUIRED`
- redistribution: `RETEST_REQUIRED`
- evaluation: `NOT_SEPARATELY_ADMITTED`

## Privacy / PII

Main-namespace article text can contain information about living or otherwise identifiable people. Public availability does not remove privacy, publicity, or other non-copyright risks. The bounded probe counts simple email-like and phone-like signals but this is not a substitute for the current project privacy gate. Exact normalized bytes must pass the project privacy/PII policy before any future training admission.

## Bounded LOCAL_FREE acquisition

The dedicated workflow downloads exactly one dated `pages-articles` shard with a hard `170000000` byte cap and verifies the published size and SHA-1. It computes an additional raw SHA-256 locally. Raw bytes are kept only in a temporary runner directory and are deleted before the evidence artifact is uploaded.

The normalization probe selects at most 128 current non-redirect `ns=0` pages and at most 2,000,000 canonical normalized bytes. It runs the normalization twice against the same exact raw object and requires byte-identical metadata/hashes.

No raw or normalized article text is uploaded. The artifact contains only source/revision identifiers, byte counts, hashes, privacy signal counts, and the terminal decision metadata.

## Deterministic normalization

Policy `UKWIKI_WIKITEXT_NFKC_LINES_V1`:

1. select MediaWiki namespace `0` pages;
2. exclude redirects;
3. select current revision text in dump order;
4. Unicode NFKC;
5. CRLF/CR to LF;
6. remove Unicode `Cf` format characters;
7. trim trailing whitespace per line;
8. trim outer whitespace;
9. UTF-8 encode;
10. hash canonical JSONL records bound to `page_id` and `revision_id`.

Wikitext markup is intentionally preserved during this rights retest so source/attribution notices are not silently discarded by an ad-hoc markup stripper.

## Family and dedup identity

Independent family: `wikimedia:wikipedia:ukwiki`.

This is distinct from incumbent Ukrainian family `ua.rada.open-data.laws-texts`. The record hashes are compatible with the existing exact/normalized/near-copy/fragment family audit semantics, but cross-family near-dedup is deliberately not claimed before rights admission. A future admission must run the current project dedup authority on the exact normalized candidate bytes.

## Current registry boundary

The branch is stacked on DATA-287 exact green head `b0523ccbc4b957615aac849d476cfa851be87578`. Machine registry identity is `917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c`. Ukrainian Wikipedia is not in the admitted snapshot set. A historical candidate entry exists and remains `REVIEW_REQUIRED`.

The prose identity previously stated in the DATA-287 PR body is not used when it conflicts with the committed machine registry; exact-head machine bytes and their successful component workflow are authoritative for this binding.

## RETEST blockers

1. `SHAREALIKE_TRAINING_ARTIFACT_AND_MODEL_OUTPUT_COMPLIANCE_POLICY_NOT_APPROVED`
2. `IMPORTED_TEXT_ADDITIONAL_ATTRIBUTION_PRESERVATION_NOT_PROVEN_FOR_NORMALIZED_CORPUS`
3. `EXACT_NORMALIZED_PRIVACY_PII_GATE_NOT_EXECUTED`
4. `CROSS_FAMILY_NEAR_DEDUP_NOT_EXECUTED`

No model training is executed by this worker.
