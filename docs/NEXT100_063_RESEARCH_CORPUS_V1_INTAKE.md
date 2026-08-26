# NEXT100-063 — Research Corpus V1 intake convergence

## Decision

`PREDECONTAMINATION_CANDIDATE_FROZEN`

This worker composes the smallest exact-byte successor intake that resolves the DATA-301 `G09` independent-family blocker without replay or source aliases. It does **not** claim a final Research Corpus V1, shards, training capacity, or a learned model.

## Exact parent

- DATA-301 head: `8820ba1b255f6bb95c7db0531fd846078a1aae01`
- DATA-301 evidence identity: `939065abeefff8aed924415589608ff3fc721fe4b0a57fc200146a4b6a137e81`
- parent state: `TERMINAL_BLOCKED`

## Terminal source composition

1. DATA-287 external snapshot registry V2 — 5 records / 183,061 normalized bytes.
2. NEXT100-026 KMu Secretariat — 6 Ukrainian records / 9,153 normalized bytes.
3. NEXT100-034 NIST technical series — 3 English records / 59,358 normalized bytes.

All three source authorities are bound to exact successful dedicated workflow runs in the machine manifest.

## Frozen candidate

- records: `14`
- unique normalized SHA-256 identities: `14`
- normalized source bytes: `251,572`
- UK: 97,718 bytes / 7 records / 2 independent families
- EN: 144,151 bytes / 5 records / 2 independent families
- code: 9,703 bytes / 2 records / 2 independent families
- pre-decontamination candidate identity: `fb2b95bdc93301fc4456e06650722024794a46ac692883ffc8e74be131cca1e9`

The exact DATA-300/DATA-301 `2/2/2` family minimum is therefore satisfied at the pre-decontamination intake boundary. This removes the old structural family-count blocker only. It does not imply balanced final capacity.

## Deferred terminal source

NEXT100-037 CPython documentation remains a valid terminal source authority, but its training-eligible unit is 14 accepted chunks while exact per-chunk byte lengths/content materialization are not sealed in that source authority. It is deliberately not counted in this exact-byte intake. A later successor may add it after exact chunk materialization and cross-source dedup.

## Required next gates

The candidate identity is intended to unblock the next decontamination worker. Before any training authorization, the composed candidate still requires exact/normalized/fragment/near-copy evaluation decontamination, cross-source dedup, quality/privacy reruns, cluster-safe split, deterministic shards, two clean byte-identical builds, post-pack unique-loss accounting, and balanced no-replay capacity authorization.

Source bytes are never relabeled as optimized causal positions. `authorized_unique_loss_positions` remains exactly `0`.

## Safety boundary

`LOCAL_FREE` only. No long training, paid compute, optimizer campaign, final-test access, corpus-release claim, or learned-20M claim was performed.
