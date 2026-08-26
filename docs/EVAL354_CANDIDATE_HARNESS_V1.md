# EVAL-354 Candidate-Response Harness V1

## Scope

This harness connects the frozen EVAL-354 communication suite to deterministic candidate generation without changing the EVAL-354 fixture bytes or their terminal identity.

Frozen suite identity remains:

- suite: `eval354-communication-suite-v1`;
- suite identity SHA-256: `7a94f1c2dd9cb31a571dd8383ed1994936abc6e6003b0b53829778ae19ba2ba7`;
- cases SHA-256: `65e2d28ef4adb442d636ec286e34ce7b43b21d7d2612325d1ca96e310863be8f`.

The harness supports the incumbent EVAL-354 behaviors: exact output, structured JSON equality, forbidden fabrication checks, context consistency, and instruction adherence. Scoring remains delegated to `communication_eval.py`; the harness only supplies deterministic candidate responses and binds the result envelope.

## Candidate request firewall

`CommunicationCase` contains scorer-only material, including the expectation and the project-authored `reference_response`. Candidate adapters never receive that object directly.

They receive `CandidateRequest` with exactly:

- `case_id`;
- conversational `messages`;
- `tool_state`.

Expectation values and reference responses are therefore absent from the adapter-visible request.

## Deterministic generation

Every case is generated twice with the same request. Any byte difference fails closed before scoring. Mapping fixtures must also bind exactly the EVAL-354 case ID set; missing or extra response IDs fail closed.

Supported candidate kinds are:

- `deterministic_mock` — mechanics evidence only;
- `learned_base_adapter_plumbing` — Base-adapter plumbing evidence only, never a quality claim;
- `post_sft_model` — reserved for future post-SFT fixture-level comparison.

No foreign model, remote judge, API call, paid compute, or training run is required.

## Immutable result envelope

Result schema:

`12-6.post-base.communication-eval-harness-result.v1`

Each result binds:

- suite ID and immutable suite identity;
- candidate ID, candidate kind, deterministic-generation flag, and generator identity SHA-256;
- a response-set SHA-256 derived from ordered per-case response hashes;
- per-category pass/fail counts;
- per-case category, response SHA-256, UTF-8 byte length, pass/fail, and deterministic scorer reason;
- a result identity SHA-256 over canonical JSON excluding only the identity field itself.

Raw candidate responses are not embedded in the comparison result. This makes the artifact stable for later candidate comparison while avoiding a convenient response corpus.

The envelope hard-codes:

- `evaluation_use = evaluation_only`;
- `suite_training_eligible = false`;
- `candidate_outputs_training_eligible = false`;
- `final_test_training_reuse_authorized = false`;
- `raw_responses_embedded = false`;
- `reference_responses_exposed_to_candidate = false`;
- `broad_quality_claim_authorized = false`;
- `base_raw_lm_diagnostic = false`;
- `claim_scope = fixture_behavior_only`.

The validator fails closed if these boundaries are weakened or if response/result hashes no longer match.

## Deterministic mock control

The project-authored scorer-control mapping is used only to prove harness mechanics. It is not candidate quality evidence and is not communication training data.

Pinned control identities:

- candidate ID: `eval354-mock-control-v1`;
- generator identity SHA-256: `518ec14daca039a708b3ce9c12b485c8717e9ab4271ad355dc97280cbcc44027`;
- response-set SHA-256: `fb23106c7c5349df73c34f2f915ce20ff9ef52e6c6e2035f22e57801e5412d64`;
- result identity SHA-256: `078f0d454a095cced4635cfe3770a61876dcd48355df5f465f4c600ae5d042ae`.

A learned Base adapter may later be wrapped with `CallableResponder`, but any such run must use candidate kind `learned_base_adapter_plumbing` and remains plumbing-only evidence. It must not be reported as communication quality or post-SFT quality.

## Library use

```python
from twelve_six.post_base.communication_eval import load_suite
from twelve_six.post_base.communication_harness import (
    CandidateDescriptor,
    CallableResponder,
    LEARNED_BASE_ADAPTER_PLUMBING,
    run_candidate_harness,
)
```

The adapter callable accepts a `CandidateRequest` and returns one string. The caller must bind a stable generator/checkpoint/config identity as SHA-256.

## CLI use for deterministic response fixtures

```bash
python -m twelve_six.post_base.communication_harness \
  --manifest tests/fixtures/post_base/communication_eval/v1/manifest.json \
  --mock-responses path/to/project-owned-responses.jsonl \
  --candidate-id candidate-name \
  --output evidence/post_base/eval354/candidate-name.json
```

The CLI accepts the same two-field JSONL response records as the incumbent EVAL-354 scorer: `case_id` and `response`.

## Training firewall

EVAL-354 fixtures and harness outputs are evaluation-only. Neither the fixture expectations, reference responses, candidate outputs, per-case hashes, nor evaluation results authorize ingestion into SFT, preference, reward, or Base training datasets. A future final-test reservation must remain outside training by the same rule.
