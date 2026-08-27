# NEXT100-063 terminal source registry V4

## Decision

`CONVERGED_TERMINAL_SOURCE_AUTHORITY_VECTOR_REQUIRES_SUCCESSOR_GLOBAL_DEDUP_NOT_CORPUS_FREEZE`

V4 advances the source-authority layer only. It consumes two exact successful machine authorities that became available after V3 without converting source capacity into corpus, tokenizer, loss-position, training, or quality claims.

## V4 deltas

### Accepted-only CPython capacity

NEXT100-037 admitted the CPython tutorial source with 14 accepted DATA-228 chunks and two `pii_phone` rejections. The later exact accepted-only materialization at head `8f0cbc16f9a920ca9ab3e3061b53fbfec8838d77` completed workflow `33005689174` successfully and retained artifact `9620571005` (`sha256:5c04e12f1100fd4012efc1cf693f213d1d7c9ababee2a16367897377cde60379`).

V4 therefore credits exactly **15,540** accepted UTF-8 bytes, not the 17,901-byte full normalized source. The remaining **2,361** normalized bytes stay uncredited. The materialized comparison payload is 15,566 bytes because comparison separators are not training-capacity credit.

### Exact Gutenberg terminal source

The exact NEXT100-033 Gutenberg realization at head `3f4ad26e1e8f3406a1274418cf5f485814ce3032` completed workflow `32998859164` successfully and retained artifact `9618402768` (`sha256:63fa5d9b403432074193e290beb0473b5a1f7b74de1ac30bad71b9ec8405e006`). NEXT100-107 seals this machine result under authority identity `1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b`.

V4 credits exactly **1,672,110** normalized bytes across three admitted records while assigning exactly **one** independent family: `en.project-gutenberg.public-domain-books`. Evaluation remains unauthorized by this source authority.

## Exact pre-successor-global-dedup vector

- Ukrainian: 100,856 numeric bytes / 4 families.
- English: 1,838,293 numeric bytes / 5 families.
- Code: 106,031 numeric bytes / 5 families.
- Total numeric source capacity: **2,045,180 bytes**.
- Total normalized source envelope: **2,047,541 bytes**.
- Uncredited normalized bytes: **2,361 bytes**.
- Independent families: **14**.
- Research Corpus V1 20,000,000-byte acquisition-planning gap: **17,954,820 bytes**.
- Authorized balanced no-replay causal-loss positions: **0**.

These are source-authority planning units before successor global dedup. They are not token counts, packed loss positions, epochs, FLOPs, or evidence of a learned 20M model.

## Fail-closed downstream boundary

V4 does not freeze Research Corpus V1. The next authority must run global exact/near/fragment/lineage dedup over the composed exact record graph, then reserved-evaluation decontamination, post-composition quality/privacy and mixture checks, cluster-safe split/packing, two clean byte-identical builds, and the post-pack unique causal-loss ledger.

Long training remains `BLOCKED`. Paid compute remains `NOT_AUTHORIZED`. No tokenizer fit, optimizer update, learned-20M claim, or 100M/1B promotion is introduced by V4.
