# AUDIT-B CulturaX UA/EN Rights and Provenance V1

Status: `CHANGES_REQUIRED`  
Worker: `SWARM-747`  
Lane: `AUDIT-B|CULTURAX|INDEPENDENT-VERIFY|UA-EN-RIGHTS-PROVENANCE-V1`  
Control: issue #723 / `SWARM-300-V2`  
Base main: `5020afd671a3885c1b738c8b4eafe7525f630546`  
Execution profile: `LOCAL_FREE`

## Audit question

Can the live `#720` CulturaX candidate be treated as canonical 12-6 training authority for Ukrainian and English now?

No. CulturaX is a high-value source-qualification candidate, but this audit finds no defensible basis for granting any 12-6 training credit from public availability, aggregate token counts, or package/dataset-card licensing alone. The live registry already says `canonical_training_authorized=false`; this package makes that boundary executable and defines what a later D03 acquisition worker must prove.

This is an independent AUDIT-B package. It does not patch D03 Product code, download corpus payloads, fit a tokenizer, train a model, touch Base weights, access benchmark/final-test payloads, or authorize paid compute.

## Live project authority

The audit is bound to:

- main SHA `5020afd671a3885c1b738c8b4eafe7525f630546`;
- `configs/research/open_source_reuse_registry_v2.json`;
- registry Git blob `d80a60357c56eacac135f948b8a72556bb849e5a`;
- component `CULTURAX`;
- decision `P0_UA_EN_COMPARISON_ACQUISITION`;
- `canonical_training_authorized=false`;
- the registry-wide rules that package-level dataset licensing is not training authority and source-level rights/provenance are required.

The validator recomputes the Git blob identity of the checked-out registry and fails closed on authority drift.

## Current upstream observation

The audited CulturaX card is pinned for evidence to README revision `6a8734bc69fefcbb7735f4f9250f43e4cd7a442e`:

`https://huggingface.co/datasets/uonlp/CulturaX/blob/6a8734bc69fefcbb7735f4f9250f43e4cd7a442e/README.md`

The card declares:

- CulturaX combines mC4 `3.1.0` with OSCAR `20.19`, `21.09`, `22.01`, and `23.01`;
- the corpus is described as 6.3 trillion tokens across 167 languages;
- English is exposed as `en/*.parquet`;
- Ukrainian is exposed as `uk/*.parquet`;
- declared English inventory is 3,241,065,682 documents / 2,846,970,578,793 tokens / 45.13%;
- declared Ukrainian inventory is 44,740,545 documents / 38,226,128,686 tokens / 0.61%;
- records retain `text`, `timestamp`, `url`, and `source`, where source is `mc4` or an `OSCAR-xxxx` lineage label;
- the card says CulturaX licensing strictly follows mC4 and OSCAR;
- the card warns that personal or sensitive information may remain;
- its gated access text requires the user to confirm downloading is legal for the current jurisdiction/use case and to agree not to attempt re-identification.

Those numbers are upstream descriptive statistics only. They are not 12-6 admitted bytes, tokenizer tokens, unique causal-loss positions, or Research Corpus V1 evidence.

## Rights and privacy finding

### mC4

Primary card: `https://huggingface.co/datasets/allenai/c4`

The mC4/C4 card identifies Common Crawl as the source. It states ODC-BY terms for the dataset and also states that use is bound by Common Crawl terms for the contained content. Therefore ODC-BY cannot be treated by this project as a blanket transfer of rights in every underlying webpage.

### OSCAR 23.01

Primary card: `https://huggingface.co/datasets/oscar-corpus/OSCAR-2301`

OSCAR explicitly distinguishes packaging/metadata/annotations from crawled text. Its card states that CC0 applies to the packaging, metadata, and annotations, while the OSCAR authors do not own the crawled text. The card also exposes jurisdiction/copyright-use caveats. Treating the CC0 metadata label as ownership of source text would be an audit failure.

### Common Crawl

Terms observed: `https://commoncrawl.org/terms-of-use`  
Last updated: `2024-03-07`

The terms state that Crawled Content can be subject to separate terms from content owners, that use must comply with applicable law and third-party rights, and that privacy-related restrictions apply. This is incompatible with a project rule that would infer universal training authority from Common Crawl availability alone.

### CulturaX

CulturaX inherits those source families and explicitly points users back to both mC4 and OSCAR licensing. It also preserves source URL/timestamp/source metadata and warns about personal or sensitive material. The correct project posture is therefore source-aware, provenance-preserving qualification, not blanket admission.

This report is a project engineering/audit decision, not legal advice. A later acquisition must re-check the actual terms, jurisdiction, and intended use at execution time.

## UA/EN feasibility decision

`HIGH_FOR_SOURCE_QUALIFICATION`, but `ZERO_CREDIT_PENDING_SOURCE_LEVEL_QUALIFICATION`.

Both English and Ukrainian are clearly present upstream. English has a much larger declared inventory, while Ukrainian is still large enough to be strategically relevant to the project's UA capacity bottleneck. That makes CulturaX worth a D03 acquisition/qualification package.

It does **not** make CulturaX currently training-ready. This audit accessed no corpus payload and therefore credits:

- admitted source bytes: `0`;
- admitted tokenizer tokens: `0`;
- authorized unique causal-loss positions: `0`;
- bulk download: `false`;
- corpus payload access: `false`.

## Required successor contract

Before any CulturaX byte receives canonical training credit, the D03 successor must:

1. Pin an immutable CulturaX dataset revision and enumerate exact selected `en`/`uk` files.
2. Record exact file identity, size, and checksum before preprocessing.
3. Retain `source`, `url`, and `timestamp` per record into immutable intake manifests.
4. Report mC4 and each OSCAR release separately rather than collapsing inherited authority.
5. Perform source-level rights/terms classification; do not infer text ownership from CulturaX, ODC-BY, or CC0 packaging metadata.
6. Re-check current CulturaX, mC4, OSCAR, Common Crawl, jurisdiction, and use-case terms.
7. Run privacy/PII risk processing, with evidence and quarantine/removal accounting; do not claim perfect PII elimination without evidence.
8. Apply the reserved-evaluation firewall and decontamination before training eligibility.
9. Apply exact/near cross-source deduplication and terminal no-replay accounting.
10. Compute capacity and unique causal-loss positions only from locally verified, rights-approved, decontaminated terminal artifacts.
11. Freeze immutable manifests and two-clean-build determinism evidence before any Research Corpus V1 promotion.

AUDIT-B should independently re-check the exact resulting artifacts. D03 owns acquisition implementation; this audit does not.

## Executable fail-closed checks

`tools/validate_culturax_ua_en_audit.py` rejects, among other things:

- `canonical_training_authorized=true`;
- any nonzero 12-6 source-byte/token/unique-loss credit;
- public/package licensing being treated as blanket source training authority;
- OSCAR CC0 metadata packaging being treated as crawled-text ownership;
- loss of CulturaX `source/url/timestamp` provenance;
- erasure of the personal/sensitive-data caveat;
- benchmark/final-test access or training use;
- foreign weights/tokenizers/teacher data entering canonical Base lineage;
- paid compute/model-training/tokenizer-fit claims;
- live registry component drift or registry blob drift;
- removal of required successor decontamination/dedup/rights/privacy gates.

`tests/test_culturax_ua_en_audit.py` supplies positive and adversarial cases for those boundaries.

## Audit verdict

`CHANGES_REQUIRED`

The required change is **not** “change the registry to authorize CulturaX.” The required work is a separate, exact D03 acquisition/qualification package that proves source-level authority, provenance retention, privacy handling, evaluation decontamination, dedup/no-replay, deterministic materialization, and terminal capacity. Until then the only scientifically defensible project training credit is zero.
