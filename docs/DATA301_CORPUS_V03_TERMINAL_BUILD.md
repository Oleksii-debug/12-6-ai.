# DATA-301 Corpus V03 Terminal Build

Worker: `DATA-301-CORPUS-V03-TERMINAL-BUILD`

Execution profile: `LOCAL_FREE`

Base authority: DATA-300 v2 at commit `8ea7f830e50a23754d189dd4134f4afad76a7ee9`, contract identity `07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5`.

## Verdict

`TERMINAL_BLOCKED`

No corpus identity or shard identity is emitted. The corpus remains not built, not frozen, not terminal and not release-ready.

The exact DATA-300 v2 inventory cannot satisfy the terminal DATA-295 diversity gate: Ukrainian has one independent family, English has one, and code has two, while the frozen policy requires at least two independent families in every stratum. The family-constrained no-replay budget is therefore zero. No replication, replacement sampling, replay, alias inflation or padding-as-data is allowed to repair the deficit.

The same frozen contract also lacks a nonempty terminal selection-validation authority, lacks passing exact Wave-3 quality and privacy reruns, lacks a full five-source post-split unique-loss ledger, and consequently has no authorized pair of clean deterministic shard builds.

Changing the exact source inventory is not permitted under this DATA-300 identity. Such a change requires a successor DATA-300 contract.

## Exact candidate inventory

| Stratum | Source objects | Independent families | Normalized unique prebuild bytes |
| --- | ---: | ---: | ---: |
| uk | 1 | 1 | 88,565 |
| en | 2 | 1 | 84,793 |
| code | 2 | 2 | 9,703 |
| total | 5 | 4 | 183,061 |

Per family: `ua.rada.open-data.laws-texts` = 1 object / 88,565 bytes; `en.standardebooks.manual` = 2 objects / 84,793 bytes; `github:encode/httpx` = 1 object / 8,161 bytes; `github:psf/requests` = 1 object / 1,542 bytes.

DATA-298 terminal prebuild evidence reports zero duplicate matches and zero duplicate-discount bytes. The frozen contract still requires exact/near dedup to rerun on the exact materialization before a build may be promoted.

## One-pass optimized loss-position accounting

No terminal full five-source Wave-3 ledger exists, so no nonzero full-corpus one-pass loss capacity is claimed.

The terminal DATA-294 ledger covers only the three DATA-229 text objects: 173,358 normalized bytes and 173,355 unique optimized causal targets, split as 88,564 Ukrainian and 84,791 English; code is outside that ledger and is reported as zero. Ledger identity: `9a1cd57c52459bdc6e4bb2d46047a47713e10d9a5be7b0a4b86f041ba6f62bd0`.

## Pipeline status

The requested ordering is preserved: rights → normalization → quality → privacy → exact/near dedup → evaluation decontamination → cluster-safe split → balance → deterministic sharding.

Rights are terminal-pass for all five training objects and normalization is bound to their exact hashes. Quality and privacy are hard-blocked because no passing exact Wave-3 reruns exist. DATA-298 dedup is prebuild-only. Evaluation decontamination has preflight authority but no final build result. Cluster-safe split, balance materialization and deterministic sharding are therefore not reached. Two clean builds are not performed because doing so would manufacture an invalid product after hard prebuild gates have already failed.

## Product Trainer streaming proof

DATA-301 binds `src/twelve_six/training/trainer.py` Git blob `8fb5e9ce4c5417986ad1f086ebc16cd7538a151e`. The dedicated validator parses `Trainer.run` and requires its source contract to remain `batches: Iterable[Batch]`, directly iterate `for batch in batches`, preserve the `max_steps` boundary guard, and avoid `len`, indexing, `list`, `tuple` or `sorted` materialization of the supplied iterable. This is a fail-closed source-level proof that a one-shot generator can be streamed through the Product Trainer API.

## Terminal evidence identity

DATA-301 evidence identity: `939065abeefff8aed924415589608ff3fc721fe4b0a57fc200146a4b6a137e81`.

This identity is the SHA-256 of canonical JSON after omitting the evidence identity and identity-scope fields.
