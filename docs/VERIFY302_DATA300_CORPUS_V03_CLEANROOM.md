# VERIFY-302 — DATA-300 Corpus V03 Clean-room Verification

## Run identity

- Worker: `VERIFY-302-CORPUS-V03-CLEANROOM`
- Timestamp: `2026-08-26T14:07:14Z` / `2026-08-26T17:07:14+03:00` Europe/Uzhgorod
- Execution profile: `LOCAL_FREE`
- Model training: **not executed**
- Verification branch: `verify302/corpus-v03-cleanroom-20260826`
- Frozen DATA-300 head: `8ea7f830e50a23754d189dd4134f4afad76a7ee9`
- DATA-300 contract identity: `07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5`
- DATA-301 comparison head: `8820ba1b255f6bb95c7db0531fd846078a1aae01`

## Clean-room input boundary

DATA-301 output bytes were not used as build input. Canonical terminal source artifacts were reacquired independently before DATA-301 was consulted as a comparison target:

- DATA-213 artifact `9600107886`, ZIP SHA-256 `927e10ccb83f919d57a8b78e3ea9cae72f9d4e0232401043234a35eec150074d`
- DATA-227 artifact `9602093542`, ZIP SHA-256 `080f073327020cb3bbb05c7348f658223804684d23012d9b66ab9b798c4fed5d`

The artifacts were expanded into two separate clean roots with no shared mutable build cache. Text normalization was re-executed from raw canonical snapshot bytes using the retained DATA-181 extraction/normalization semantics; code normalization was verified as strict UTF-8 identity and both Git blob SHA-1 identities were recomputed.

## Independent source rebuild result

Exact five-object inventory reproduced:

- EN Standard Ebooks typography: 48,002 normalized bytes, SHA-256 `154fb4034929714087e75150d678bf65049ddac32e79dcdf97162c8972c2be83`
- EN Standard Ebooks metadata: 36,791 normalized bytes, SHA-256 `94eb2f529922d125b3bd40691778886f4d5d80b128b925d0274fb3d94646ec5a`
- UK Verkhovna Rada law text: 88,565 normalized bytes, SHA-256 `72c301db0b2539f3f7a73c9c15e2e425700a6b758a1114f1a861e2d60c704c50`
- code encode/httpx: 8,161 bytes, SHA-256 `2c61b3ac94d1dcebcde0c6f519554d2d7917247fbaa0a97002db4ef69e70ff28`, Git blob SHA-1 `6f479a0885f723b7395843d41164a87041820776`
- code psf/requests: 1,542 bytes, SHA-256 `4c7d8d132c9898fc7d715e473f3ac74785ddc4ab96d2c9240f87835dc6d981ff`, Git blob SHA-1 `0466a7d347db4ed34a37db51b75fc8e80bc06055`

Totals: 183,061 normalized unique prebuild bytes. By stratum: UK 88,565; EN 84,793; code 9,703. Independent family counts: UK 1; EN 1; code 2.

Two independent prebuild evidence trees were byte-identical:

- `source-inventory.jsonl` SHA-256, A/B: `486d2319a655e763e2da74a9c8ea796525b590fbdedc70f9823330b9e3333ffd`
- `source-rebuild.json` SHA-256, A/B: `3739e21c32a2bc01cdc81e57e5e31c6c2aceb17e030c164dc9581fde42926160`

## Pipeline execution and fail-closed stop

- rights: PASS from terminal purpose-specific source authorities
- normalization: PASS, independently re-executed from raw snapshots
- quality: HARD BLOCKED because no passing exact Wave-3 DATA-32 rerun exists
- privacy: HARD BLOCKED because no passing exact Wave-3 DATA-33 rerun exists
- exact/near dedup: whole-object exact precheck PASS; full Wave-3 result not authorized
- evaluation decontamination: no final build result authorized
- cluster-safe split: NOT REACHED
- balance: HARD FAIL because family counts `uk=1,en=1,code=2` violate the frozen minimum of two independent families per stratum
- deterministic sharding: NOT REACHED

The pipeline therefore stops before split or shard materialization. Creating train/selection/final membership, a corpus identity, or shards would violate the frozen DATA-300 hard gates.

## DATA-301 comparison

DATA-301 exists at `8820ba1b255f6bb95c7db0531fd846078a1aae01` and independently reports `TERMINAL_BLOCKED`.

Comparison result:

- document/candidate inventory: MATCH — 5 sources, 183,061 normalized unique bytes, matching stratum bytes and family counts
- split membership: MATCH — neither side materialized splits
- corpus identity: MATCH — `null` on both sides
- shard SHA set: MATCH — empty on both sides because sharding was not reached
- unexplained differences: none

This does **not** establish corpus reproducibility because no valid corpus product exists. It establishes reproducibility of the terminal blocker and confirms that DATA-301 did not have a valid corpus/shard identity that clean-room reconstruction could compare byte-for-byte.

## Exact tests executed

Local clean-room commands executed twice against independent roots:

- raw snapshot byte-size and SHA-256 verification for all five objects
- DATA-181 text extraction + first-50,000-character strict normalization re-execution for all three text objects
- normalized byte-size/SHA-256 comparison against frozen identities
- retained normalized artifact comparison as evidence only, never build input
- strict UTF-8 identity verification for both code objects
- Git blob SHA-1 recomputation for both code objects
- exact whole-object normalized SHA collision check
- deterministic A/B evidence-tree SHA comparison and byte `cmp`

Repository regression test added: `tests/test_verify302_corpus_v03_cleanroom.py`.

## Changed artifacts

- `reports/verify302/data300_corpus_v03_cleanroom_verification.json`
- `tests/test_verify302_corpus_v03_cleanroom.py`
- this run report

## NOT TESTED / not reachable

- train/selection/final split membership, because hard pre-split gates fail
- full Wave-3 quality and privacy passes, because no terminal passing exact reruns exist
- full exact/near dedup and evaluation-decontamination product over an authorized materialization
- cluster-safe split
- full five-source unique-loss ledger
- deterministic shard tree and per-shard SHA-256
- Product Trainer execution; no model training was permitted or performed

## Blocker and verdict

**VERDICT: TERMINAL_BLOCKED_MATCHES_DATA301_NO_CORPUS_PRODUCT_TO_COMPARE.**

Primary immutable blocker: the frozen DATA-300 v2 inventory has only one independent UK family and one independent EN family, while DATA-295 requires at least two independent families per stratum and replay is forbidden. The same contract also lacks a nonempty terminal selection-validation authority, passing exact Wave-3 quality/privacy reruns, and a full five-source unique-loss ledger.

A source-inventory change cannot be made inside frozen DATA-300 v2. The next valid action is a successor DATA-300 contract that admits sufficient independent UK/EN families and closes the remaining hard authorities before any corpus materialization attempt.

## Ownership

This worker owns only VERIFY-302 clean-room verification evidence and its regression test. It does not modify DATA-300 source inventory, DATA-301 terminal evidence, training code, model state, or source snapshots.
