# DATA-295 — Bounded UA/EN/code mixture policy for a 20M research corpus

Worker: `DATA-295-BALANCE-POLICY-20M-V1`

Execution boundary: `LOCAL_FREE` only. This is a preregistration and inventory calculation. It performs no model training and uses no model-result tuning.

## Decision

The primary top-level mixture remains the pre-existing preregistered `45% Ukrainian / 35% English / 20% code` allocation. DATA-295 does not use DATA-34 or any later BPB result to retune those weights.

The new hard constraint is source-family diversity:

- at least 2 independent training-eligible families in each of Ukrainian text, English text, and code;
- no family may exceed 25% of the total selected corpus;
- no family may exceed 60% of its own stratum;
- a mirror, fork, vendored copy, generated derivative, or duplicated document does not count as a new family;
- if a family would overflow a cap, subsample that family or acquire another independent family; never replay another document to fill the quota.

At a 20,000,000 source-byte-token target, the selected allocation is:

| stratum | target share | unique source-byte-token target |
|---|---:|---:|
| Ukrainian text | 45% | 9,000,000 |
| English text | 35% | 7,000,000 |
| code | 20% | 4,000,000 |
| total | 100% | 20,000,000 |

This preserves continuity with the incumbent top-level mixture while preventing a single publisher/project family from silently becoming the corpus.

## Terminal inputs consumed at the cutoff

### DATA-229 text registry

Exact head: `90bc0b7f8b696ec35202532b13edf6ab29a662fe`.

Dedicated `DATA-229 Real Snapshot Registry` run `32957147036` completed successfully. Registry identity: `1357a343eb4ea973950d8991913109cbea53fe4fa891f0be9745ab497eb59486`.

Training-eligible normalized UTF-8 text bytes:

- `ua.rada.open-data.laws-texts`: 88,565 Ukrainian bytes;
- `en.standardebooks.manual`: 84,793 English bytes across two objects;
- total: 173,358 bytes.

DATA-229 had zero code sources at its own cutoff.

### DATA-227 code admission

Exact head: `8ebdb2e132ed7bae5245e9d4c140752640ab9885`.

Dedicated `DATA-227 External Real Code Admission V2` run `32956209865` completed successfully and retained artifact `9602093542`, digest `sha256:080f073327020cb3bbb05c7348f658223804684d23012d9b66ab9b798c4fed5d`.

It admits exactly two independent identity-preserved code objects after rights/D03/dedup gates:

- `github:encode/httpx`: 8,161 bytes;
- `github:psf/requests`: 1,542 bytes;
- total: 9,703 bytes.

No exact or >=0.85 near-duplicate pair was reported between the two admitted code objects.

### Explicit exclusions

DATA-228 head `46a70c990dab6ff72bb84ddb54cff1156b491b40` is not consumed because its dedicated immutable-source probe run `32957120454` is terminal failure.

No terminal DATA-230 research-corpus authority is consumed. No durable terminal DATA-278 English large-source expansion authority was discoverable at this cutoff, so DATA-295 does not invent its output.

The cross-authority sum below is therefore a policy input, not a new converged corpus identity.

## Current unique material

| stratum/family | unique training-eligible source bytes |
|---|---:|
| Ukrainian — Rada | 88,565 |
| English — Standard Ebooks manual | 84,793 |
| code — httpx | 8,161 |
| code — Requests | 1,542 |
| total | 183,061 |

Current family counts are `UK=1 / EN=1 / code=2`.

Therefore the full DATA-295 family gate is **not activatable** today: Ukrainian and English each need at least one additional independent family. The family-constrained training budget is consequently `0` until that gate passes. This is deliberate fail-closed behavior rather than permission to let Rada or Standard Ebooks dominate.

## Feasible top-level policies from current unique material

The following comparison ignores the family-count gate only to quantify acquisition pressure. It uses unique source bytes once, never document duplication or replay. Budgets are rounded down to deterministic 100-byte granularity, or 200 bytes for the half-percent candidate.

| policy | UA/EN/code | current modality-only max no-replay source-byte budget | realized allocation at that ceiling | 20M target allocation |
|---|---|---:|---|---|
| continuity | 45/35/20 | 48,500 | 21,825 / 16,975 / 9,700 | 9.0M / 7.0M / 4.0M |
| symmetric text | 40/40/20 | 48,500 | 19,400 / 19,400 / 9,700 | 8.0M / 8.0M / 4.0M |
| balanced | 45/40/15 | 64,600 | 29,070 / 25,840 / 9,690 | 9.0M / 8.0M / 3.0M |
| text equal | 45/45/10 | 97,000 | 43,650 / 43,650 / 9,700 | 9.0M / 9.0M / 2.0M |
| scarcity relief | 47.5/47.5/5 | 178,400 | 84,740 / 84,740 / 8,920 | 9.5M / 9.5M / 1.0M |

All five are mathematically feasible at a non-zero top-level source-byte budget. Only `45/35/20` is selected. The larger ceilings of lower-code candidates are not evidence that code should be reduced; they merely expose that code is currently the tightest inventory constraint. Selecting a lower code share just to consume more of today's text would be data-availability tuning, not the intended corpus design.

## 20M acquisition gap under the selected policy

Against the current cross-authority inventory, the minimum additional unique source bytes needed before dedup/family-cap losses are:

- Ukrainian: `8,911,435`;
- English: `6,915,207`;
- code: `3,990,297`.

These are lower-bound acquisition gaps. Actual acquisition must exceed them because some candidate bytes will be rejected by rights, privacy, quality, exact/near dedup, reserved evaluation splits, or family caps.

At 20M, the hard family caps mean:

- no family may contribute more than 5,000,000 total bytes;
- within Ukrainian, the 60% stratum cap is 5,400,000, so the stricter 5,000,000 global cap applies;
- within English, the 60% stratum cap is 4,200,000;
- within code, the 60% stratum cap is 2,400,000.

Thus a compliant 20M corpus necessarily needs at least two real independent families in every stratum, even before any stronger diversity study is attempted.

## Rights, quality, privacy, and dedup gates

A source contributes only if it is pinned to an exact immutable revision and passes the existing explicit training-rights and D03 gates. Public availability, an SPDX string, or a familiar publisher name is not self-authorization.

Mirrors/forks cannot manufacture family diversity. Generated, vendored, build, minified, privacy-sensitive, ambiguous-license, exact-duplicate, and near-duplicate material is excluded under the existing authorities. No missing bytes are backfilled by copying accepted documents.

## No-replay accounting boundary

The quantities above are source-byte/source-token ceilings under the canonical byte-token view. They are **not** relabelled as exact optimized causal loss targets.

Before any training run may claim a no-replay optimized-target budget, the final admitted corpus must publish an exact post-split, post-dedup, post-packing source-position ledger that proves every optimized causal position is used at most once. Padding is never counted as data.

Current selected-policy state:

- 20M ready: **NO**;
- full family-constrained no-replay budget: **0** because UK/EN family-count gates fail;
- modality-only ceiling before the family gate: **48,500 source bytes**;
- exact optimized-loss-target budget: **UNPUBLISHED** pending the final position ledger.

## Unblock condition

Admit at least one additional independent training-eligible Ukrainian family and one additional independent training-eligible English family, retain the two-or-more code-family requirement, then rebuild a deterministic converged registry and recompute unique bytes after all rights/D03/dedup/family-cap losses. The `45/35/20` weights remain frozen until a separately preregistered mixture experiment is authorized; model outcomes from the current corpus must not retroactively change DATA-295.
