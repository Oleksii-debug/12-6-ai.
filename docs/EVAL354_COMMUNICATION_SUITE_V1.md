# EVAL-354 Communication Suite V1

## Purpose

EVAL-354 freezes a small, project-authored evaluation suite for future post-Base
assistant behavior. It is intentionally separate from canonical Base raw-language-model
diagnostics and from communication-training data.

The suite tests only five narrow behaviors:

- instruction adherence;
- basic dialogue consistency;
- refusal to invent results when a required tool observation is unavailable;
- formatting compliance;
- use of supplied conversational/context evidence.

It does not measure broad intelligence, general knowledge, safety completeness,
professional competence, open-domain factuality, or production readiness.

## Immutable identity

Terminal V1 binds these exact identities:

- `cases.jsonl` SHA-256: `65e2d28ef4adb442d636ec286e34ce7b43b21d7d2612325d1ca96e310863be8f`;
- suite semantic identity SHA-256: `7a94f1c2dd9cb31a571dd8383ed1994936abc6e6003b0b53829778ae19ba2ba7`;
- provenance: `PROJECT_AUTHORED`;
- foreign model output admitted: `false`.

The committed `manifest.json` binds the exact bytes of `cases.jsonl` with SHA-256 and
also binds its own semantic metadata through `suite_identity_sha256`. Any fixture-byte
change fails closed. After terminalization, a substantive fixture change requires a
successor version rather than silently changing V1 evidence.

All V1 records declare `PROJECT_AUTHORED`. Foreign model output is not admitted as
fixture content. Reference responses are project-authored scorer controls only; a
successful reference-control run is not model evidence.

## Separation from Base

EVAL-354 uses `evidence/post_base/eval354`. Canonical Base evidence remains
`evidence/base`.

The manifest hard-codes:

- `base_raw_lm_diagnostics = false`;
- `training_eligible = false`;
- `broad_intelligence_claim_authorized = false`;
- scope `fixture_behavior_only`.

The evaluator imports no model, trainer, or checkpoint implementation, performs no
generation, and has no optimizer or weight-update path. Base raw-LM metrics must not
be written into or inferred from EVAL-354 results.

## Fixture structure

V1 contains 12 cases:

- instruction adherence: 2;
- dialogue consistency: 2;
- refusal to invent unavailable tool results: 3;
- formatting: 2;
- context handling: 3.

Each case includes conversational messages, explicit tool availability state, a
deterministic expectation, and a project-authored reference response used only to test
scorer mechanics.

The two additional adversarial cases remain inside the original V1 scope. One asks the
assistant to fabricate a CRM observation while the tool is unavailable; the expected
behavior is to reject the fabricated value. The other introduces a later user
correction followed by a contradictory assistant assertion; the expected behavior is
to preserve the user's corrected context.

Exact-output checks are used where the prompt itself demands exact output. JSON
formatting is parsed structurally and rejects extra keys. Missing, duplicate, or extra
response case IDs fail closed.

## Determinism

For an identical immutable suite and identical response mapping, scoring is required
to return byte-equivalent structured content after canonical JSON serialization.
The validator evaluates the same reference-control mapping twice and fails if the two
structured results differ.

## Running future candidates

A future candidate runner should generate one response per case without exposing
`reference_response` to the candidate. Write JSONL records shaped as:

```json
{"case_id":"instruction.exact_ack","response":"ACK"}
```

Then run:

```bash
python -m twelve_six.post_base.communication_eval \
  --manifest tests/fixtures/post_base/communication_eval/v1/manifest.json \
  --responses path/to/candidate-responses.jsonl \
  --output evidence/post_base/eval354/<candidate-id>.json
```

The result always carries `claim_scope=fixture_behavior_only`,
`broad_intelligence_claim=false`, and `base_raw_lm_diagnostic=false`.

## LOCAL_FREE boundary

The suite and validator are Python-stdlib-only at runtime. Tests require only the
repository development dependencies. No external LLM, remote evaluator, paid API,
GPU, or paid compute is required or authorized by EVAL-354.
