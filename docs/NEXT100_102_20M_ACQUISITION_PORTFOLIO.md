# NEXT100-102 — 20M data acquisition portfolio

## Decision

The current ~20M model mechanics are not the limiting factor. The limiting factor is a terminal, immutable, sufficiently diverse Research Corpus V1.

At the NEXT100-102 claim cut, NEXT100-063 PR #527 exposes a **nonterminal planning input** of:

- Ukrainian: 100,856 pre-successor-dedup source bytes;
- English: 144,151 bytes;
- code: 69,133 bytes;
- total: 314,140 bytes.

The frozen DATA-295 target remains 20,000,000 source bytes at 45% Ukrainian / 35% English / 20% code. Therefore the current pre-dedup planning gaps are:

- Ukrainian: 8,899,144 bytes;
- English: 6,855,849 bytes;
- code: 3,930,867 bytes;
- total: 19,685,860 bytes.

These are **source-capacity bytes**, not tokens and not optimized causal-loss positions.

## Why this package exists

The repository already has many source-specific workers. The missing control-plane piece is a deterministic acquisition envelope that prevents four recurring mistakes:

1. treating a promising URL or license page as admitted training data;
2. filling a quota by replaying or duplicating one source family;
3. allowing one large family to dominate a stratum;
4. jumping directly from source admission to tokenizer fitting or long training without successor global dedup, corpus identity and decontamination.

The machine configuration and validator make those mistakes fail closed.

## Frozen family geometry

DATA-295/NEXT100-069 remains authoritative for the mixture policy:

- maximum family share of the full 20M target: 25%;
- maximum family share of its own stratum: 60%;
- minimum independent families per stratum: 2;
- replay/duplication is forbidden as quota repair;
- model BPB or other downstream model results must not be used to retune this preregistered mixture.

At the 20M target this yields maximum single-family planning envelopes of:

- Ukrainian: 5,000,000 bytes;
- English: 4,200,000 bytes;
- code: 2,400,000 bytes.

So even in the most concentrated legal plan, the remaining gap cannot be solved by one new family in any stratum. NEXT100-102 deliberately plans more family slots than the bare minimum to reduce concentration risk before the balance/diversity retest.

## Research-only acquisition portfolio

### Ukrainian

Planning budget: 10.0 MB for an 8.899 MB current gap.

Priority lanes:

- multi-author public-domain Ukrainian literature, with exact author/edition/public-domain and transcription-platform rights per object;
- official legal/regulatory/public-information text from independently proven origins;
- prose-heavy open-data material from multiple agencies, with strict exclusion of identifiers, personal records and low-information tables.

The official Verkhovna Rada legislation portal currently states that site content is available under CC BY 4.0 unless otherwise noted. The national open-data portal likewise states CC BY 4.0 unless otherwise noted and allows reuse with source attribution. These facts are **research leads only**: they do not replace object-level provenance, quality, PII, reservation and dedup checks.

Research references:

- https://zakon.rada.gov.ua/laws/show/2811-20#Text
- https://data.gov.ua/pages/about

### English

Planning budget: 7.2 MB for a 6.856 MB current gap.

Priority lanes:

- document-level U.S. federal government works with employee/authorship provenance and third-party exclusion;
- technical prose from multiple independent federal agency origins;
- bounded expansion of the already terminal NIST family using the same document-specific rights discipline;
- additional independent public-domain or permissively licensed prose families.

17 U.S.C. 105 is useful screening evidence, not a blanket permission rule for everything hosted by the U.S. government. Contractor works, transferred copyrights, special statutory categories and embedded third-party material must fail closed unless separately resolved.

Research reference:

- https://uscode.house.gov/view.xhtml?req=(title:17%20section:105%20edition:prelim)

### Code

Planning budget: 4.2 MB for a 3.931 MB current gap.

Priority lanes:

- bounded first-party SQLAlchemy source under an exact MIT release/license binding;
- bounded expansion of the already represented Django family without exceeding the code-family cap;
- CPython-authored source with module-level third-party exclusions and exact applicable license evidence;
- multiple additional independent permissive Python families.

SQLAlchemy's current documentation publishes the MIT license for the software and documentation. NEXT100-102 still gives it zero capacity: an exact release, first-party allowlist, secret/privacy checks, parse checks, lineage-aware dedup and evaluation separation are required before source authority.

Research reference:

- https://docs.sqlalchemy.org/en/21/copyright.html

## Required execution order

The safe order is fixed:

1. exact source qualification;
2. source-registry convergence;
3. successor global cross-source/lineage-aware dedup;
4. balance/diversity retest;
5. immutable pre-decontamination candidate-corpus identity;
6. exact/near evaluation decontamination;
7. quality/privacy/split/two-clean-build verification;
8. exact post-pack unique-loss ledger;
9. tokenizer-fit authorization;
10. MODEL-341 ~20M training authorization.

No step later in this list may manufacture the missing evidence of an earlier step.

## 100M and 1B implication

The 20M corpus target is a proving-ground target, not a reusable proof that a 100M or 1B model has enough data. Once the learned 20M model passes its preregistered scientific gates, the next scale step must create a new data/compute scaling contract based on **unique post-dedup causal loss positions**, measured learning curves and explicit compute budget. The project should not move to 100M merely because the architecture can instantiate 100M parameters, and it should not move to 1B merely by multiplying layer width/depth.

For 100M+, the required engineering expansion includes distributed training qualification, checkpoint/recovery at the larger state size, throughput/memory profiling, larger tokenizer/corpus freeze, contamination controls, and matched-budget evaluation. For 1B, those become first-class infrastructure rather than optional optimizations.

## Truth boundary

NEXT100-102 performs no model training, tokenizer fitting, optimizer update, paid compute or final-test payload access. It admits no source and credits zero new training capacity. Its output is an executable acquisition plan that can safely feed parallel source-qualification workers without allowing planning evidence to masquerade as scientific or training authority.
