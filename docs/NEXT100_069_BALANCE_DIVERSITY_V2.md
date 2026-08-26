# NEXT100-069 — Corpus Balance and Diversity Gate V2

Worker: `NEXT100-069-BALANCE-DIVERSITY-V2`

Execution class: `LOCAL_FREE`. No model training. No BPB or other model result is read or used to retune the mixture.

## Frozen policy

The DATA-295 policy remains unchanged:

- Ukrainian: 45% / 9,000,000 source bytes at the 20M target.
- English: 35% / 7,000,000 source bytes.
- Code: 20% / 4,000,000 source bytes.
- At least 2 independent families in every stratum.
- No family may exceed 25% of total selected bytes.
- No family may exceed 60% of its own stratum.
- Replay or duplication cannot satisfy quota.

## Consumed terminal source vector

This authority is stacked on NEXT100-065 head `90065ffc97a5133e76cacdee6991eb171c4ea2ba` and its refreshed terminal-source rule: only an exact current authority head whose dedicated admission/qualification workflow completed successfully receives capacity.

Conservative unique source-authority capacity after NEXT100-065 lineage dedup:

- Ukrainian: 90,044 bytes across 2 families.
- English: 84,793 bytes across 1 family.
- Code: 69,133 bytes across 4 families.
- Total: 243,970 bytes across 7 independent families.

No capacity-collapsing duplicate edge exists in the consumed vector. Same-origin sibling objects collapse only for family independence, not byte capacity.

## Family balance

Available-pool family shares, before any deterministic subsampling:

- `ua.rada.open-data.laws-texts`: 88,565 bytes; 36.301594% global; 98.357470% of UA.
- `ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv`: 1,479 bytes; 0.606222% global; 1.642530% of UA.
- `en.standardebooks.manual`: 84,793 bytes; 34.755503% global; 100% of EN.
- `github:encode/httpx`: 8,161 bytes; 3.345083% global; 11.804782% of code.
- `github:psf/requests`: 1,542 bytes; 0.632045% global; 2.230483% of code.
- `github:django/django`: 54,156 bytes; 22.197811% global; 78.335961% of code.
- `github:Kludex/starlette`: 5,274 bytes; 2.161741% global; 7.628774% of code.

Hill-q2 effective family counts on unique-byte shares:

- UA: 1.0333898867.
- EN: 1.0.
- code: 1.5775237133.
- global: 3.2947591213.

Whole-pool concentration is not a deletion rule. Rada, Standard Ebooks, and Django would need deterministic capping/subsampling in a composed mixture. The decisive hard blocker is English family count: 1 < 2. Therefore the maximum non-replayed fixed 45/35/20 mixture satisfying every family rule is currently 0 bytes.

## Acquisition gaps to the frozen 20M target

Exact unique source-byte gaps against the consumed terminal vector:

- UA: 8,909,956 bytes.
- EN: 6,915,207 bytes.
- code: 3,930,867 bytes.
- total: 19,756,030 bytes.

At the 20M target, the effective per-family ceilings are 5,000,000 UA bytes, 4,200,000 EN bytes, and 2,400,000 code bytes. English requires at least one additional terminal independent family merely to pass the two-family hard minimum. If existing exact snapshots are frozen and every gap byte must come only from newly acquired families, at least two new families per stratum are required by the byte ceilings. This second statement is not imposed when an existing terminal family later expands with additional unique admissible objects.

## Unique loss positions

A full current-vector loss-position count is deliberately not inferred from source bytes. NEXT100-064 generalized multi-source loss accounting is absent.

DATA-294 certifies only the incumbent text subset:

- Rada UA: 88,564 unique causal loss positions.
- Standard Ebooks EN: 84,791.
- certified partial total: 173,355.

The current EN stratum therefore has an exact 84,791 loss positions because no additional EN source is terminal in NEXT100-065. Full UA is unknown because the 1,479-byte Wikisource object is not in DATA-294. All 69,133 code bytes are also outside DATA-294. Exactly 70,612 current source bytes remain unledgered for causal loss-position accounting.

## Verdict

`FAIL_RETAIN_45_35_20_POLICY_ACQUIRE_MORE_DATA_NO_BPB_RETUNING`

The mixture is not retuned. The next admissible progress is corpus acquisition and terminalization, especially an independent English family, followed by a refreshed cross-source dedup registry and generalized exact loss-position ledger.
