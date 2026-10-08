# Plan 2 Section 12 — immutable LOCAL_FREE tokenizer fit fixture

This is repository-controllable Section 12 engineering qualification, **not**
physical-corpus fit authorization, learned-model training or production release.

The tool `tools/plan2_tokenizer_fit_freeze_v1.py` reuses the accepted Plan-2 S11
byte/BPE architecture proposal, S10's family-isolated synthetic train partition,
the existing S0 byte tokenizer and canonical `TokenizerIdentity` / compatibility
guard. No shadow training, split or checkpoint authority is added. It fits only
the S10 *fixture* train rows, never validation/test, and never widens S10's
`tokenizer_fit_authorized=false`.

Deterministic frequency/lexicographic-tie byte-BPE learns at most 96 fixture
merges. Bytes 0–255 retain raw meanings; four distinct special-token IDs
256–259 are frozen; merges occupy IDs from 260. Version, exact ordered merge
table, normalization=none, UTF-8 fallback, complete vocab hash and config hash
form a frozen `TokenizerIdentity`. Fixture ModelSpec vocabulary size is
explicitly exact and intentionally much smaller than the *future target* 32,768.

Immutable evidence includes parent S10 split SHA, S11 architecture manifest SHA,
train-only record digest, policy Git blob, ordered merges, exact special IDs,
vocabulary/config fingerprints, frozen model-vocabulary binding and explicit
`production_release_authorized=false`. Restart or clean rebuild must reproduce
byte-identical manifests; mutated output, policy, special IDs, vocabulary,
checkpoint identity and unauthorized corpus promotions must fail closed.
Importing `FrozenBPE` cannot itself authorize its use with training/inference.

LOCAL_FREE positive/negative pytest, compiler, CLI restart and clean rebuild
are required. The current *physical* S9 candidate has only one independent
source, has no released three-way split and cannot authorize physical tokenizer
fit. That external readiness is deferred to the appropriate physical corpus
acceptance; do not misreport this fixture as the full production tokenizer.
No material paid compute is executed.
