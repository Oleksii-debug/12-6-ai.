# NEXT100-063 — Research Corpus V1 terminal-source convergence

Status: `TERMINAL_SOURCE_INTAKE_READY_RECORD_MATERIALIZATION_REQUIRED`.

This package closes one specific gap between terminal DATA-301 and the first meaningful ~20M Base campaign: it creates a single fail-closed intake authority for newer source qualifications that were not allowed to mutate the frozen DATA-300 v2 inventory.

## What changed

The old DATA-301 source-authority cardinality was `uk=1, en=1, code=2`. NEXT100-063 consumes only source qualifications whose dedicated source workflows completed successfully:

- NEXT100-026 KMu Secretariat — one new UA family, 6 exact records, 9,153 normalized bytes;
- NEXT100-022 Ukrainian Wikisource — one new UA edition lineage, 1 exact 1,479-byte snapshot;
- NEXT100-037 CPython documentation — one new EN family, 17,901 normalized source bytes, but only 14 accepted chunks are training-eligible and their exact accepted payload bytes are intentionally not invented here.

At the source-authority layer the resulting family counts are `uk=3, en=2, code=2`. This resolves the old minimum-two-families cardinality prerequisite without replay, aliases, mirrors or duplicated family credit.

EVAL-303 is also bound as the current nonempty immutable selection-validation authority (10 records: UA 8 / EN 2 / code 0), so the historical DATA-301 empty-selection blocker is no longer carried forward.

## What this does not authorize

This is not Research Corpus V1 freeze or release. It emits no corpus identity, no shard identity and no post-pack unique-loss count. Authorized training exposure remains exactly zero. Tokenizer fitting, long training and paid compute remain blocked.

Source normalized bytes are not causal loss positions. In particular, CPython's 17,901 normalized source bytes cannot be counted as training payload until the exact 14 accepted chunks are materialized and identity-bound.

## Required successor order

1. Materialize exact accepted records for every added source.
2. Freeze one exact pre-decontamination record inventory and identity.
3. Run reserved evaluation decontamination: exact, normalized, fragment, near-copy, mirror and code-copy checks.
4. Rerun quality, privacy and lineage-aware global dedup over that exact inventory.
5. Recompute the frozen 45/35/20 mixture under family-share caps with no replay.
6. Produce cluster-safe split/shards, then two independent byte-identical clean builds.
7. Materialize the post-pack unique nonignored causal-loss ledger; that ledger is the only legal no-replay training ceiling.
8. Independently close the D05 checkpoint-corruption retest and requalify TRAIN-344 on exact MODEL-341 before a long campaign.

No model training or Base-weight mutation is performed by NEXT100-063.
