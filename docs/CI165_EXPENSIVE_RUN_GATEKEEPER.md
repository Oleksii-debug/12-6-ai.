# CI-165 Expensive Run Gatekeeper

`SWARM_WORKER_ID: CI-165-EXPENSIVE-RUN-GATEKEEPER`

## Purpose

Every long learned-model training phase must fail closed before it consumes meaningful runner time unless a cheap launch gate has produced a valid machine envelope for the exact checkout and launch request.

The gate is preflight only. It does not instantiate model weights, create an optimizer, load a checkpoint, run a forward pass or train a token.

## Launch request

Schema: `12-6.expensive-run-launch-request.v1`.

A request binds:

- workflow/config identity;
- purpose/dependency profile path and ID;
- Python/tool/module requirements;
- focused test selectors;
- exact `ModelSpec`, semantic hash and parameter count;
- tokenizer version/config/vocab identities;
- corpus manifest path and corpus identity;
- critical imports;
- checkpoint output path and minimum disk headroom;
- positive optimizer-step/token/wall budget;
- whether GPU visibility is actually required.

MILESTONE-150 has explicit requests for 100K, 500K and 1M in `configs/launch/`.

## Check order

The historical SCALE-141 failure was `python -m pytest: No module named pytest` after runtime setup. CI-165 therefore checks required modules/commands first. Only after this passes does the gate validate the dependency profile/locks and import project runtime modules to construct metadata-only `ModelSpec` and tokenizer identities.

GPU visibility is checked only when the request says `requires_gpu=true`.

## Launch envelope

Schema: `12-6.expensive-run-launch-envelope.v1`.

The envelope records the exact Git SHA, canonical request SHA-256, purpose/base profile file identities, every referenced lock hash, Python executable/version, test selectors, budget, storage probe, corpus identity, ModelSpec identity/count, tokenizer identities, critical imports, conditional GPU evidence and `training_performed=false`.

`envelope_sha256` is the SHA-256 of the complete canonical unsigned envelope. This is an integrity signature, not a cryptographic identity signature.

A long entrypoint revalidates the envelope against the current Git SHA, canonical launch request, expected workflow/scale binding, purpose profile/lock identities and Python version. Missing, stale, tampered or differently bound envelopes are rejected.

## MILESTONE-150 migration

`src/twelve_six/milestone150_entrypoint.py` requires `TWELVE_SIX_LAUNCH_REQUEST` and `TWELVE_SIX_LAUNCH_ENVELOPE` for `phase1` and `resume`. Non-training `prepare`, `verify-scale`, `finalize` and report validation remain callable without an envelope.

The exact-head workflow creates and verifies all three envelopes after DATA-25 prepare and before the first training phase, then passes the matching request/envelope pair into each rung.

## Historical regression

`tests/test_ci165_launch_gate.py` adversarially removes `pytest` from module discovery and asserts that gate creation fails with `required module unavailable: pytest` before any heavyweight project-contract function can run and before any envelope can be written.

## Boundaries

No paid compute is authorized by this gate. A PASS envelope is launch-integrity evidence only; it does not grant stage promotion, intelligence, production-readiness, alignment or instruction-following claims.
