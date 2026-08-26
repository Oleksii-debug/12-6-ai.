# EVAL-291 — immutable external-real English selection-validation authority

Worker: `EVAL-291-EN-SELECTION-VALIDATION-V1`

## Scope

This authority admits English data for model/tokenizer-family and checkpoint selection validation only. It does not create training data, tokenizer-fit data, or final-test data.

The authority is based on the two terminal source families admitted by DATA-227 at exact head `8ebdb2e132ed7bae5245e9d4c140752640ab9885`: `github:encode/httpx` and `github:psf/requests`. EVAL-291 reserves different exact documentation objects from the same pinned upstream revisions. The DATA-227 training objects are named in the config and are forbidden from reuse.

## Preserved final-test boundary

EVAL-291 is based on EVAL-233 head `b5512b4648cb09dd052b08884dc53f291e1ce935`. It binds only the RECOVER-174/EVAL-233 authority metadata and identities. The preserved final-test payload is not a builder input, is not copied into the EVAL-291 namespace, and its outcomes are not inspected or allowed to influence construction.

The final-test authority source IDs are `en.standardebooks.manual` and `ua.rada.open-data.laws-texts`. EVAL-291 uses neither.

## Exact selection objects

- HTTPX `docs/advanced/timeouts.md` at upstream commit `b5addb64f0161ff6bfe94c124ef76f6a1fba5254`, Git blob `aedcfb627f5139901638ebf8de30216c2640591e`, raw SHA-256 `48959a587ecc6430f9051ab1b1c76fcebc3f3e95ec01199281aab813a2dcfaa1`.
- Requests `docs/user/authentication.rst` at upstream commit `5460f467b02e49471c0fd6cfc9ca0adab6351f98`, Git blob `76be9cccafc0ce28a698672a128eac9d1f9bbe15`, raw SHA-256 `5878d62d3929b057f8a6008f21641ae9fae0515b5ed6174da7ac763e36ccdae6`.

The committed source snapshots are byte-identical to these upstream Git objects.

## Rights and redistribution evidence

HTTPX is bound to its exact pinned BSD-3-Clause license object and carries that exact license snapshot.

Requests is bound to its exact pinned Apache-2.0 license Git object. This authority carries a complete Apache-2.0 compliance copy plus the exact upstream NOTICE object. The EVAL-291 decision is purpose-specific: these exact documentation objects are approved and reserved for selection validation. The project deliberately prohibits their use for training, tokenizer fitting, and final testing even though the upstream licenses grant broader reuse rights.

## Deterministic rebuild

`python -m twelve_six.eval291_en_selection_validation build --repo-root .`

The build is offline. It reads only the committed EVAL-291 config, source snapshots, and rights evidence; emits canonical `en.jsonl` and the authority manifest; and produces identical bytes on every rebuild. `verify` fails closed on source-byte mutation, rights-evidence mutation, training-object reuse, final-test payload references, or purpose-boundary changes.

Committed authority identity: `727f229c091f86748a4eee9ea5aec72bb65347b68d6b687fabbf33166b0eca1e`.

Committed selection JSONL SHA-256: `df20e3d3ec75208399039a283487c3b8958c80ec3f119cee278fcc09948c6bfb`.
