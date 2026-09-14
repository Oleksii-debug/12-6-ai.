# Common Pile v0.1 source-rights audit

Status: `CANDIDATE_SOURCE_RIGHTS_AUDIT`

Claim: `SWARM-746` / `D03|COMMON-PILE|SOURCE-RIGHTS-AUDIT|V1`

Base: `5020afd671a3885c1b738c8b4eafe7525f630546`

## Decision

This package does **not** authorize Common Pile v0.1, any Common Pile source family, or any
Common Pile byte for canonical 12-6 training. It records upstream source-specific rights and
provenance claims in a machine-checkable form so later D03 workers can review sources one at a
time without treating the Common Pile name, its repository MIT license, or an aggregate dataset
card as blanket training authority.

The registry therefore fixes all training-credit fields at zero, requires `REVIEW_REQUIRED` for
all 30 raw v0.1 source families, excludes the final-test surface, and rejects terminal/adopted
states.

## Bound upstream authority

- Release: Common Pile v0.1.
- Raw-data collection: `https://huggingface.co/collections/common-pile/common-pile-v01-raw-data`.
- Raw collection membership: exactly 30 dataset entries.
- Paper: *The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly Licensed Text*,
  arXiv `2506.05209`.
- Code repository: `https://github.com/r-three/common-pile`.
- Audited current code commit: `9457f04a14cb2355ab00023420369d46ffd4a395`.
- Audited current code tree: `b481d48569f509097be8c502c6b9f89a27094251`.
- Audited `sources/` tree: `59d9c98a7885903f34de985c0316475c5f935b6c`.
- Repository code license at the audited snapshot: MIT.

The code commit above is an immutable **current audit snapshot**. This package does not assert
that it was the exact commit used to cut v0.1. The registry makes that distinction executable.

## What the upstream paper says

The source families use materially different rights/provenance strategies. Examples include:

- per-document CC BY / CC BY-SA / CC0 filters for ArXiv papers and PubMed Central;
- CC0 metadata for ArXiv abstracts;
- repository-license filtering for GitHub Archive and The Stack V2;
- public-domain metadata or collection scope for BHL, pre-1929 books, Library of Congress,
  Project Gutenberg, legal/government sources, Ubuntu IRC, and most retained PEPs;
- explicit license-statement checks for DOAB and LibreTexts;
- public-domain / CC BY / CC BY-SA filters for PressBooks and OER Commons;
- per-wiki metadata plus license-laundering filtering for Wikiteam;
- manually curated CC BY YouTube channels;
- CC-marker plus high-volume-domain verification for Creative Commons Common Crawl;
- CC BY / CC BY-SA news-site selection and CC BY-SA Public Domain Review articles.

These are recorded as upstream claims and filtering descriptions, not as a 12-6 legal opinion.

## Collector mapping

At the audited code commit the `sources/` tree contains 25 collector directories. Several collectors
cover multiple raw v0.1 families, including `sources/arxiv` and `sources/wiki`. Three raw families
have no dedicated collector directory in the audited current tree: GitHub Archive, pre-1929 books,
and YouTube. Their registry rows are explicitly marked
`NOT_PRESENT_AT_AUDITED_CODE_COMMIT`; absence is not silently repaired or inferred.

## Fail-closed invariants

`src/twelve_six/common_pile_rights.py` rejects the registry when any of the following occur:

1. the source-family set is not exactly the 30 v0.1 raw dataset identities;
2. a mutable upstream code ref replaces the immutable commit;
3. the current audit snapshot is relabeled as the v0.1 release-cut commit;
4. package/repository licensing is promoted into blanket dataset authority;
5. global or source-local training authorization becomes true;
6. corpus bytes or optimized loss positions become nonzero;
7. source-specific review, provenance, rights signals, or collector status disappear;
8. final-test exclusion is weakened;
9. status becomes terminal/adopted;
10. semantic content changes without a new deterministic registry identity.

## Downstream admission handoff

A successor that wants to admit one source family must independently bind and retain:

1. an exact immutable source snapshot/version and acquisition recipe;
2. the applicable source/item license or public-domain evidence, including attribution,
   share-alike, notice, version, jurisdiction, and redistribution obligations where relevant;
3. provenance to the original rights holder/source rather than only a Common Pile package label;
4. privacy, terms-of-service, robots/access, quality, and sensitive-data review where applicable;
5. exact normalization/materialization identities and reproducibility evidence;
6. global exact/near deduplication and mirror/fork/fragment lineage against all selected sources;
7. evaluation reservation and decontamination before training split;
8. split/tokenizer/packing identities and a unique nonignored causal-loss-position ledger;
9. a separate project training-authorization decision.

Until those steps are terminal for a particular source, its `credited_bytes` and
`authorized_loss_positions` remain zero.

## Scientific and compute boundary

No corpus was downloaded or materialized. No tokenizer was fitted. No optimizer update or model
training was run. No final-test payload was accessed. No GPU/cloud resource was provisioned and no
paid compute was authorized. This work is a source-rights/provenance audit and validator only.
