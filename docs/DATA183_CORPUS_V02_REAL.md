# DATA-183 — Corpus V0.2 Real Candidate

Status: `CANDIDATE_UA_EN_REAL_PROJECT_CODE` when the workflow succeeds.

Authority: `LOCAL_FREE_CANDIDATE_NOT_CORPUS_FREEZE_OR_REPRESENTATIVENESS_PROMOTION`.

DATA-183 is a convergence layer on the accepted DATA-110 corpus path. It does not replace the rights gate, normalization/materialization, quality, privacy, exact dedup, DataTrove 0.10.0 near dedup, D06 decontamination, cluster-safe split, deterministic sharding, Product packer, or Trainer.

## Live-state boundary

At the DATA-183 bootstrap point, no live branch named DATA-181 or DATA-182 and no indexed DATA-181 commit supplied an admitted external code source. DATA-110 does reacquire bounded rights-approved real Ukrainian and English sources and intentionally rejects any external language/modality outside `uk` and `en`.

Therefore DATA-183 must not claim a full external UA/EN/code V0.2 corpus. The valid fallback is real external UA/EN plus project-authored code, with `EXTERNAL_REAL_CODE_UNAVAILABLE` and representativeness blockers retained in the machine report.

## Added evidence

`src/twelve_six/data183_corpus_v02_real.py` runs DATA-110's two full clean builds and requires identical corpus identity and shard hashes. It then independently reopens every retained shard with hash verification and records:

- canonical report origin classes `EXTERNAL_REAL` and `PROJECT_AUTHORED`, while preserving DATA-110 shard spellings for lineage;
- zero normalized train-validation overlap under an independent NFKC/newline/whitespace audit normalization;
- unique retained record IDs, with no document replication mechanism added to hit a token target;
- exact one-finite-TRAIN-pass autoregressive optimization-target token supply by origin, exact `source_id` source family, stratum, and source-family/origin pair, computed through Product `iter_packed_examples` with `cross_document=false`;
- real retained-shard streaming through Product packing into three committed CPU `Trainer.train_microbatch` updates, one each for Ukrainian, English, and code;
- the upstream DATA-110 rights, policy, dedup/decontamination, split, and deterministic-build evidence;
- explicit absence of a full V0.2 representativeness, production-readiness, or universal semantic-cleanliness claim.

The token-supply metric is corpus supply for one finite pass. It is deliberately distinct from a training run's repeated optimized-token budget.

## Required successful evidence

A successful workflow must produce `data183-evidence/corpus-v0.2-real-candidate.json` with a valid self-hash and all of the following true:

- real external Ukrainian survived all gates;
- real external English survived all gates;
- project-authored code survived all gates;
- normalized train-validation overlap is exactly zero;
- DATA-110 clean build A and B have identical corpus identity;
- build A and B have identical shard path/hash identities;
- actual Trainer streaming passed on CPU under LOCAL_FREE authority;
- `documents_duplicated_to_reach_token_target=false`;
- `full_v0_2_claim=false`.

Until an independently rights-approved external code source is admitted through the corpus pipeline, `external_real_code_present=false`, status remains `CANDIDATE_UA_EN_REAL_PROJECT_CODE`, and full V0.2 representativeness remains blocked.

## Non-claims

This release candidate does not claim corpus representativeness, universal benchmark cleanliness, production readiness, model intelligence, alignment, instruction following, or paid-compute authorization.
