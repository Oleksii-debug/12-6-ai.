# D03 clean-retained + PEP + LoC global dedup composition

This package composes the already-qualified clean retained DATA-232 carrier with
the independently materialized PEP public-domain and selected Library of Congress
candidates, then delegates duplicate matching to the canonical indexed incumbent
V3 executor merged through PR #1459.

It does **not** create another matcher, grant corpus credit, authorize tokenizer
fit, authorize optimizer exposure, or execute training.

## Exact clean retained authority

The base carrier is the PR #2183 clean current-main handoff, exact Product head
`22cdc8696a11cb3490538d6060fe41f9b4d67a95`. V10 accepts only the deterministic
qualified outputs:

- `training_records.jsonl` SHA-256
  `3458afe0380ea45d328ad3f004b21845a188ca69f2f7a83c24999e9e53268e53`;
- `training_handoff.json` SHA-256
  `80bcf2dd28f0d13795ceea01b358c7149b636f55f17b29575d14313b5cee99ee`;
- 257 retained records from 244 distinct physical source IDs;
- 5,601,716 retained payload bytes;
- materialization identity
  `7061d74db13bf45a9a7a1266ebe50feab8e7d22c32fba7a81dd91c2be4135ade`;
- composition-preflight identity
  `1b3adfffab2a9d65af78e88b805ef221714a0cc94fca4055105665a6a155ce95`.

The handoff is revalidated against its self-hash and matcher projection. Every
training record is cross-bound to the projection by record ID, physical source
ID, family, modality, UTF-8 byte count and payload SHA-256. Matcher source IDs
are record-specific; stable-origin IDs preserve the 244 physical-source origin
count without falsely collapsing all records from one physical source as copies.

## Exact candidate authorities

PEP remains bound to 86 records / 4,987,775 normalized UTF-8 bytes and candidate
SHA-256
`65d1cf23354efc5a66d7132b67602a7af1715e18a0fb77e729595866c25d33c3`.

LoC remains bound to 28 records / 4,499,908 normalized UTF-8 bytes and candidate
SHA-256
`f87201d71eb8c3f2510cdf2c9b3ce1a32b83d1a11198e8a1ff0ad67d0a3fec20`.

Both candidate/report byte identities are checked before semantic parsing. JSON
and JSONL parsing reject duplicate keys; rows are cross-bound to report counts,
byte totals, source identities and the zero-credit boundary.

## Canonical matcher execution

V10 no longer executes V9 or accepts caller-supplied matcher callbacks. The
previous private-V9 execution seam identified by AUDIT2064-001 is removed.

Production execution uses:

- `twelve_six.data.incumbent_dedup_indexed_execution`;
- wrapper Git blob `f75336008839198b6d46bea4954f120e2d81613c`;
- core Git blob `af7be7909501ea9d76604ebed084cec32fbd9456`;
- the exact historical incumbent V3/V1/DATA-232 runtime closure attested by that
  executor;
- explicit fail-closed budgets for candidate pairs, index postings and pair
  expansions.

The runner loads V3 only from a clean worktree at terminal V7 head
`d3333ec1b4a508df232a5aefccd6686adda745fb`. The indexed executor then attests
the exact V3, V1 and DATA-232 source/runtime closure before matching. V10
re-attests immediately before incumbent report verification.

Duplicate clusters are produced only by canonical V3 semantics. Survivor
representative selection is a local deterministic projection of those clusters:
largest declared capacity, then lexicographically smallest source ID. It cannot
create new duplicate relations.

## Execution

The runner consumes job-local bytes; raw corpus text is not committed:

```bash
python tools/run_d03_expanded_global_dedup_v10_pep_loc.py \
  --v7-root /path/to/clean-v7-d3333ec1-worktree \
  --clean-training-records /path/to/training_records.jsonl \
  --clean-training-handoff /path/to/training_handoff.json \
  --pep-candidate /path/to/pep-candidate.jsonl \
  --pep-report /path/to/pep-report.json \
  --loc-candidate /path/to/loc-candidate.jsonl \
  --loc-report /path/to/loc-report.json \
  --output-report /tmp/expanded-v10-report.json \
  --output-survivors /tmp/expanded-v10-survivors.json
```

Optional `--max-candidate-pairs`, `--max-index-postings`, and
`--max-pair-expansions` tighten the canonical work bounds. Exhausting any bound
fails closed.

Before dedup, the exact authority-bound composition is 371 source objects and
15,089,399 payload bytes, with 358 stable origins.

## Scientific boundary

A V10 PASS means only that the exact clean retained, PEP and LoC inputs were
authenticated and globally deduplicated under the canonical incumbent matcher,
with text-free report/survivor identities produced.

It does not complete reserved-evaluation decontamination, canonical
quality/privacy, family caps, cluster-safe split, deterministic packing,
two-clean corpus reproducibility, tokenizer fit, positive unique-loss accounting
or optimizer execution.

Therefore after a V10 PASS:

- `AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0`;
- `tokenizer_fit_authorized=false`;
- `optimizer_updates=0`;
- `model_training_executed=false`;
- `learned_weights_created=false`;
- final-test outcomes remain unread;
- paid compute remains unused;
- foreign pretrained weights remain forbidden.
