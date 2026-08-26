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

## Dedup-certified source vector

NEXT100-065 config blob `c1e05f09490e25f6fed765dfb70d900717528f4d` remains byte-identical at the latest observed registry head `efc278cec0e4773eb4ff405bf4b4d24ee63b5d13`. Its last terminal-success source-vector cut contains 11 objects and seven independent families.

Conservative globally dedup-certified unique source capacity:

- Ukrainian: 90,044 bytes across 2 families.
- English: 84,793 bytes across 1 family.
- Code: 69,133 bytes across 4 families.
- Total: 243,970 bytes across 7 independent families.

No capacity-collapsing duplicate edge exists in that certified vector. Same-origin sibling objects collapse only for family independence, not byte capacity.

## Final concurrency refresh

NEXT100-064 appeared during the mandatory final refresh at head `0f4189f78b23b0aa5540fd024be0959b2fb29926`, cutoff `2026-08-26T18:29:51Z`. It independently records KMu Secretariat source `ua.kmu.portal.secretariat-news.bounded-six` as terminal training-admitted. The KMu exact head `40950a950b60921fd856af2719e1ae2486d9e892` has a successful dedicated source-rights workflow and declares 9,153 normalized UA bytes.

That KMu terminalization postdates the last terminal-success NEXT100-065 source-vector cut. Therefore it is not added to the globally dedup-certified unique-byte numerator until a successor NEXT100-065 convergence incorporates it. The live terminal-admitted pre-successor-global-dedup inventory is consequently:

- Ukrainian: 99,197 source-authority bytes across 3 families.
- English: 84,793 bytes across 1 family.
- Code: 69,133 bytes across 4 families.
- Total: 253,123 bytes across 8 families.

These 253,123 bytes are not presented as globally dedup-certified unique capacity. The exact conservative unique-byte answer remains the 243,970-byte lower bound above until KMu survives successor global dedup.

## Family balance

On the globally dedup-certified 243,970-byte vector, before deterministic subsampling:

- `ua.rada.open-data.laws-texts`: 88,565 bytes; 36.301594% global; 98.357470% of UA.
- `ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv`: 1,479 bytes; 0.606222% global; 1.642530% of UA.
- `en.standardebooks.manual`: 84,793 bytes; 34.755503% global; 100% of EN.
- `github:encode/httpx`: 8,161 bytes; 3.345083% global; 11.804782% of code.
- `github:psf/requests`: 1,542 bytes; 0.632045% global; 2.230483% of code.
- `github:django/django`: 54,156 bytes; 22.197811% global; 78.335961% of code.
- `github:Kludex/starlette`: 5,274 bytes; 2.161741% global; 7.628774% of code.

Hill-q2 effective family counts on certified unique-byte shares:

- UA: 1.0333898867.
- EN: 1.0.
- code: 1.5775237133.
- global: 3.2947591213.

If KMu's 9,153 bytes survive successor global dedup unchanged, the live terminal-admitted family shares would yield Hill-q2 counts UA 1.2409063312, EN 1.0, code 1.5775237133, global 3.5302436319. KMu itself would be 3.616029% global and 9.227094% of UA; Rada would still dominate UA at 89.281934%.

Whole-pool concentration is not a deletion rule. Rada and Standard Ebooks exceed the 25% global limit; Rada, Standard Ebooks, and Django exceed 60% of their current strata. Deterministic subsampling/capping is required in a composed training mixture. The decisive hard blocker remains English family count: 1 < 2. Therefore the maximum non-replayed fixed 45/35/20 mixture satisfying every family rule remains 0 bytes.

## Acquisition gaps to the frozen 20M target

Exact gaps against the globally dedup-certified unique-byte lower bound are:

- UA: 8,909,956 bytes.
- EN: 6,915,207 bytes.
- code: 3,930,867 bytes.
- total: 19,756,030 bytes.

If KMu survives successor global dedup unchanged, the conditional gaps become UA 8,900,803 and total 19,746,877 bytes; EN and code are unchanged.

At the 20M target, the effective per-family ceilings are 5,000,000 UA bytes, 4,200,000 EN bytes, and 2,400,000 code bytes. English requires at least one additional independent terminal family merely to pass the two-family hard minimum. If current exact snapshots are frozen and every gap byte must come only from newly acquired families, the byte ceilings require at least two new family bins per stratum; same-family expansion can change that topology.

## Unique loss positions

NEXT100-064 now exists, but its terminal verdict is `BLOCKED_NO_TERMINAL_POSTPACK_CORPUS_MATERIALIZATION`. It explicitly reports the exact post-pack one-pass maximum as `null`; current authorized training exposure is 0. No source bytes are relabelled as loss positions.

Its historical pre-build diagnostic is 183,056 causal targets: 173,355 text and 9,701 code. That diagnostic is explicitly not training authority and is before current evaluation reservations, global dedup, split, and packing, so it is not substituted for the requested current exact loss-position capacity.

DATA-294 remains useful only as a certified historical partial: Rada UA 88,564 and Standard Ebooks EN 84,791, total 173,355. Full current UA/code/post-pack totals remain unavailable until an exact terminal corpus materialization reaches reservations, global dedup, split, and packing.

## Verdict

`FAIL_RETAIN_45_35_20_POLICY_ACQUIRE_MORE_DATA_NO_BPB_RETUNING`

The mixture is not retuned. Blocking conditions are: only one terminal English family, no terminal post-pack corpus materialization for exact loss-position accounting, KMu awaiting successor global-dedup convergence, and the remaining 20M unique-byte acquisition gap.
