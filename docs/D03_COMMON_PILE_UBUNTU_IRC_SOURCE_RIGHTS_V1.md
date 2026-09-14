# D03 Common Pile Ubuntu IRC source-rights/provenance V1

This package closes one narrow review boundary for the exact `common-pile/ubuntu_irc` candidate lineage. It is a project-policy source qualification, **not** a general legal opinion and **not** training authorization.

## Exact scope

The authority is bound to the Common Pile registry source key `ubuntu_irc`, audited upstream collector commit `9457f04a14cb2355ab00023420369d46ffd4a395`, collector path `sources/ubuntu`, source label `ubuntu-chat`, `Public Domain` metadata signal, and `https://irclogs.ubuntu.com/` origin. Non-Ubuntu IRC, another source label, another license/status signal, another origin, or another upstream collector revision is outside this decision.

The parent Common Pile registry remains intentionally `REVIEW_REQUIRED` and zero-credit. This source-specific authority is the independent review artifact that a downstream composition may require in addition to the unchanged parent registry.

## Published policy evidence

The review records three Ubuntu-hosted policy/help surfaces, accessed on 2026-09-11:

- `https://wiki.ubuntu.com/IRC/Guidelines` — published Ubuntu IRC guidelines state that Ubuntu channels are logged and their content is considered public domain.
- `https://help.ubuntu.com/community/InternetRelayChat` — the Ubuntu Community Help Wiki independently states that Ubuntu channel content, whether officially logged or otherwise, is considered public domain.
- `https://wiki.ubuntu.com/IRC/TermsOfService` — the current IRC terms state that participation in publicly logged Ubuntu channels agrees to public storage or processing of sent messages on `irclogs.ubuntu.com` or other external sites.

The Terms of Service is used only for the logging/storage-consent fact. This package does not reinterpret that text as an express copyright waiver. The two public-domain policy statements, the exact Ubuntu-hosted archive provenance, and the Common Pile collector's source-specific `Public Domain` signal are evaluated together as the project-policy basis for this narrow source qualification.

## Upstream and real-execution binding

The audited Common Pile collector README at the exact commit records the same Ubuntu-channel public-domain basis and emits `license = Public Domain` metadata for `ubuntu-chat` records from `irclogs.ubuntu.com`. The source-specific materializer already pins:

- dataset: `common-pile/ubuntu_irc`
- revision: `47d55b0534a62bf0766c621297165451969f3de9`
- file: `v0/documents/00007_ubuntu.jsonl.gz`
- source SHA-256: `75e38bffcaceb00ed9ce9a63b1d9e74a70f5582b3e3f763a7ec573c7fd6c1e60`

A separate Product execution authority at PR #1100 records two independent byte-identical LOCAL_FREE materializations. This rights package only exact-binds that lineage as a separate authority reference; it does not ingest, copy, or promote that execution evidence into training authority.

## Fail-closed rules

`python -m twelve_six.ubuntu_irc_rights <authority.json> <parent-registry.json>` validates the source-specific authority. The validator rejects parent-registry byte drift, upstream commit/blob substitutions, policy-anchor substitutions, non-Ubuntu origins, non-`ubuntu-chat` labels, non-`Public Domain` signals, blanket website-license inference, metadata-only authority, bool/int aliases, added override fields, or any attempt to widen the zero-credit truth boundary.

No raw IRC text, usernames, or author metadata is stored in this package.

## Scientific boundary

This qualification authorizes **zero** corpus bytes, **zero** optimized loss positions, no tokenizer fitting, no evaluation payload use, no optimizer update, no model training, and no final-test access. Downstream admission still requires exact real-execution authority plus current global dedup, reserved-evaluation decontamination, post-composition quality/privacy, balance/family caps, cluster-safe split, deterministic two-clean packing, and a positive exact unique-loss ledger.

Therefore:

- source-specific rights/provenance review: qualified for the exact scoped lineage;
- canonical training admission: **false**;
- legal conclusion claimed: **false**;
- paid compute used: **false**;
- foreign pretrained weights used: **false**.
