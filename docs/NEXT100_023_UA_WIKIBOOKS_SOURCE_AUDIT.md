# NEXT100-023 — Ukrainian Wikibooks source qualification

Worker: `NEXT100-023-DATA-UA-WIKIBOOKS`

## Verdict

`RETEST`

This is a terminal qualification result for the candidate as currently evidenced. It is **not** a training-source admission and it creates **zero** new independent source-family credit.

## Exact authoritative snapshot candidate

The bounded source is the official Wikimedia Foundation dump:

- database: `ukwikibooks`
- dump date: `2026-02-01`
- file: `ukwikibooks-20260201-pages-articles.xml.bz2`
- official dump status: `complete`
- official SHA-1: `6975ba549f822ea2394567743fdb3564c36e048a`
- deterministic extraction: all namespace-0 pages in that exact current-revision XML dump
- record identity: page ID + revision ID + revision timestamp + revision SHA-1 + normalized SHA-256

The source probe verifies the compressed dump checksum before emitting evidence. No live page is accepted as a revision identity.

## Rights decision

Wikimedia Foundation Terms of Use require contributors to license ordinary contributed text under CC BY-SA 4.0 and GFDL unless a project/feature exception applies. Reusers may reuse and redistribute hosted text under the applicable licenses, including commercial use, provided the relevant license terms are followed.

That does **not** imply that every record can safely be labeled dual-licensed. Imported text may be available under compatible CC BY-SA terms but not GFDL. Public-domain material can also exist and must remain distinguished rather than being relabeled as ordinary CC BY-SA contribution text.

For reuse/redistribution, the project must preserve applicable attribution, a license notice, modification indication where relevant, and any additional attribution requirements attached to imported text. A page URL or equivalent stable copy/author list can satisfy ordinary Wikimedia contributor attribution, but imported-source requirements still have to be retained.

Therefore:

- model training: license-level use is compatible subject to provenance and attribution compliance, but this worker does not grant project corpus admission;
- redistribution: license-level reuse is compatible only with applicable attribution/license/imported-text obligations, but this worker does not grant project redistribution admission;
- evaluation: `NOT_SEPARATELY_ADMITTED`.

Authoritative/legal references:

- `https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content`
- `https://uk.wikibooks.org/wiki/Вікіпідручник:Авторські_права`
- `https://dumps.wikimedia.org/ukwikibooks/20260201/`
- `https://dumps.wikimedia.org/ukwikibooks/20260201/ukwikibooks-20260201-sha1sums.txt`

The local Ukrainian Wikibooks copyright page is useful corroboration, but the current Wikimedia Foundation Terms of Use are controlling for the platform-wide licensing framework. The local page itself is old and contains legacy wording, so it is not used to weaken or simplify current obligations.

## Provenance boundary

Wikimedia Foundation is the platform/infrastructure host; the text is community-created and may include imports from external sources. The exact dump and revision IDs establish byte/revision provenance to the Wikimedia snapshot, not authorship provenance for every underlying passage.

The current pages-articles dump is insufficient by itself to prove that every imported-text attribution condition has been captured. Before admission, a page-level provenance/attribution pass must either materialize required attribution evidence or reject records whose obligations cannot be deterministically retained.

## Language and quality evidence

`.github/workflows/next100-023-ua-wikibooks-source-audit.yml` executes `tools/audit_next100_023_ua_wikibooks.py` on the exact official dump and emits:

- namespace-0 page count;
- redirect/non-redirect counts;
- total and distributional wikitext byte statistics;
- pages at least 500 and 2,000 bytes;
- Cyrillic-letter ratio;
- Ukrainian-specific-letter count and page coverage;
- exact normalized duplicate groups;
- revision-level provenance manifest without redistributing page text.

These statistics are descriptive corpus-quality evidence only. They are not a broad quality claim and they do not override existing project quality/privacy/dedup gates.

## Family lineage and dedup

Lineage namespace: `wikimedia.uk`.

Independent-family credit at this authority: `0`.

A distinct Wikimedia hostname/database is not evidence of an independent source family. Before any family credit is granted, the exact Wikibooks records must be compared with current Ukrainian Wikipedia and Wikisource candidates/authorities using:

1. exact and normalized content hashes;
2. near-copy/mirror detection under the incumbent project policy;
3. import/copy attribution and page-history evidence;
4. connected-component family collapse for mirrors, copied pages, translations sharing document lineage, and imported derivatives.

The result may ultimately be an independent educational family, a partially overlapping family with only unique records admitted, or a full collapse into another Wikimedia lineage. This worker does not predetermine that answer.

## RETEST blockers

1. Page-level imported-text attribution completeness is not yet proven.
2. Exact-dump quality/language evidence must be materialized by the dedicated workflow and bound to the exact head.
3. Cross-Wikimedia Wikipedia/Wikisource lineage and dedup authority is not yet terminal.
4. A successor source registry must explicitly admit the exact object after those gates pass.
5. Evaluation remains excluded without separate authority.

## Exit criteria

`ADMIT` is permitted only after all blockers above are closed while retaining the exact snapshot identity and all applicable attribution/license obligations. `REJECT` is appropriate if provenance/attribution cannot be made deterministic, quality gates fail irreparably for the bounded object, or cross-Wikimedia lineage leaves no useful unique training material.

LOCAL_FREE only. No model training.
