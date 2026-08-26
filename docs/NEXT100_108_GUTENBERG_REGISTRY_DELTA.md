# NEXT100-108 Gutenberg registry delta

## Purpose

Consume the exact-green Gutenberg source evidence from NEXT100-107 into a successor delta over the current fail-closed NEXT100-063 V2 source vector, without editing the concurrently owned canonical registry files.

Parent V2 is bound at head `33017f1e344534841b31df5a6e0bfaf5b7cb2bcc`, registry identity `448dd61ed3e0d78d0bca9e202529a79c02811fd67beebe4833373d0c2ab0c0a7`: 303,374 pre-global-dedup source bytes / 11 independent families.

The additive source is one English lineage family, `en.project-gutenberg.public-domain-books`, with three exact records and 1,672,110 normalized UTF-8 bytes, backed by dedicated run `32998859164 = success` and terminal report `ADMIT`.

## Successor source vector

- UK: 100,856 bytes / 4 families
- EN: 1,822,753 bytes / 4 families
- code: 51,875 bytes / 4 families
- total: 1,975,484 bytes / 12 families
- Research Corpus V1 20,000,000-byte planning-floor gap: 18,024,516 bytes

This is almost 9.9% of the legacy 20 MB source-byte planning floor, but source bytes remain neither tokenizer tokens nor optimized causal-loss positions.

## The critical result: English is no longer the immediate limiter

Under the frozen 45/35/20 UK/EN/code source-mixture planning rule, the stratum-only no-replay ceilings are:

- UK: floor(100,856 / 0.45) = 224,124 total bytes
- EN: floor(1,822,753 / 0.35) = 5,207,865 total bytes
- code: floor(51,875 / 0.20) = 259,375 total bytes

Therefore the current balanced source ceiling remains **224,124 bytes**, limited by UK; code is the next limiter. Adding more English before UK/code expansion has sharply diminishing value for the immediately usable balanced pool.

For the 20 MB source-floor mix itself, current raw stratum gaps are:

- UK: 8,899,144 bytes
- EN: 5,177,247 bytes
- code: 3,948,125 bytes

This gives the acquisition controller a concrete priority: grow multiple rights-clear Ukrainian and code families first while continuing English diversification at lower urgency.

## Family-cap boundary

Gutenberg is 84.64% of the successor candidate inventory and 91.74% of the English inventory. That does not invalidate source admission, because family caps are downstream selection constraints. It does mean the whole Gutenberg family cannot be consumed in one 45/35/20 training mix under the existing <=25% global and <=60% within-stratum family caps. Downselection is mandatory; replay/duplication is prohibited.

## Downstream gates

Global exact/near/fragment/lineage dedup remains next. Evaluation decontamination, post-composition quality/privacy, family-cap selection, split/packing, two clean builds, unique-loss accounting, tokenizer fit and long training remain blocked.

No corpus identity, shard identity, learned checkpoint, optimizer update, paid compute or training authorization is claimed.
