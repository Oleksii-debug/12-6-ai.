# S0 first-party inference acceptance evidence

This document describes the additive D05/D07 acceptance layer implemented in
`src/twelve_six/inference/acceptance.py`.

It is deliberately an evidence composition layer. It does **not** add another
model implementation, tokenizer, checkpoint loader, generation loop, sampler,
or HTTP server. It reuses the canonical surfaces already present in the S0
successor lineage:

- D01 `TwelveSixDecoder` / `ModelSpec` through the first-party adapter;
- D04 `ByteTokenizer` identity through the first-party adapter;
- D05 verified checkpoint loading and manifest identities;
- D07 `generate`, `compare_backends`, and the local raw-completions server.

## Why this exists

The predecessor work already proves the individual pieces and the D04 exact
candidate evaluator proves real train -> checkpoint -> reload generation. What
was still missing was a compact, independently consumable D05/D07 artifact that
binds one verified checkpoint identity to the user-facing inference surfaces.

`python -m twelve_six.inference.acceptance` closes that evidence gap without
changing model or serving semantics.

## Fail-closed acceptance requirements

A report is emitted only when all of these checks pass:

1. The checkpoint loads twice through
   `twelve_six.inference.first_party:load_first_party_backend`.
2. Both loads expose identical checkpoint, Git, ModelSpec, parameter-count,
   context, tokenizer, dataset, run-manifest, step, and token-count identities.
3. `compare_backends` obtains exact `atol=0`, `rtol=0` logit parity for every
   requested probe and identical greedy token/decode behavior.
4. Greedy generation is identical across the two independently loaded
   backends.
5. Seeded sampling is identical on repeat execution and after reload.
6. A prompt that exceeds the declared context window is rejected rather than
   truncated or silently accepted.
7. A real loopback HTTP server is started with the already loaded backend.
8. `/healthz` and `/v1/models` expose the requested model identity.
9. Greedy and seeded `/v1/completions` results match the offline completion
   semantics exactly, excluding nondeterministic response id/time fields.
10. Wrong-model, chat-completion, and oversized-context requests fail closed.
11. The completed report is canonical-JSON hashed. A validator rejects schema,
    authority, PASS-state, identity, parity, or hash tampering.

Any failed requirement raises `InferenceAcceptanceError`; the CLI exits nonzero
and no PASS report is emitted.

## Privacy boundary

Prompt text is used transiently for inference but is not persisted in the
acceptance report. Each prompt is represented by:

- SHA-256 of its UTF-8 bytes;
- UTF-8 byte count;
- prompt token count;
- privacy-safe hashes/counts for greedy and sampled outputs.

Generated text and generated token ids are also represented by SHA-256 rather
than raw content. The existing HTTP server does not log request bodies.

## Report authority

The schema is `12-6.inference-acceptance.v1` and the authority string is
`LOCAL_FREE_INFERENCE_EVIDENCE_NOT_PROMOTION`.

A successful artifact proves only the named local inference acceptance checks
for the exact checkpoint identity captured in that artifact. It does not grant
release or promotion authority and does not replace AUDIT-A, AUDIT-B, D10, or
COORD decisions.

The report explicitly records:

- `raw_base_completion_semantics=true`;
- `hidden_prompt_or_chat_template=false`;
- `promotion_authority=false`;
- `windows_nvda_live_execution=false`.

The last field is important for the current repository. The physical GitHub
repository name ends with a period (`12-6-ai.`), which remains a separate native
Windows checkout/runtime acceptance problem. This evidence collector is
platform-neutral Python and must not be cited as a live Windows/NVDA PASS.

## CLI

Example against an existing verified checkpoint:

```text
python -m twelve_six.inference.acceptance ^
  --checkpoint PATH_TO_CHECKPOINT ^
  --prompt "12-6" ^
  --seed 20260825 ^
  --max-new-tokens 8 ^
  --output inference-acceptance.json
```

On POSIX shells the same arguments can be written on one line or continued with
backslashes. The command refuses to overwrite an existing output path.

When `--output` is omitted, one JSON report is written to stdout. Operational
PASS/FAIL diagnostics go to stderr when a file is requested. This keeps the
surface usable from plain terminals and automation without a TUI or GUI.

## Test coverage

`tests/test_inference_acceptance.py` covers:

- exact reload/logit/generation/HTTP parity on a deterministic backend;
- prompt-content absence from serialized evidence;
- identity mismatch rejection;
- logit divergence rejection;
- report hash tamper rejection;
- a real canonical `TwelveSixDecoder` checkpoint serialized by D05, loaded twice
  by the first-party adapter, and accepted through the full collector including
  loopback HTTP.

The real-checkpoint fixture is intentionally random-init and tiny. It tests the
acceptance mechanism itself. Real trained+reloaded S0 generation remains bound
to the exact-candidate D04 evidence workflow; this layer consumes trained
checkpoints when supplied and does not make a new training-quality claim.
