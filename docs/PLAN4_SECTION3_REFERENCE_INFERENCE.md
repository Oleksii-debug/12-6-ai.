# Plan 4 / Section 3 — Canonical reference inference

## Semantic authority and boundaries

`tools/inference_runtime.py` is the one reference tokenizer→model→decode
adapter for this Plan-4 component. It **reuses** `TwelveSixDecoder.generate`
from Plan 3 for the actual next-token operation; no second sampler/logit
filter/token selector is defined here. Serving, gateways, optimized backends,
signed checkpoint publication and product packaging have distinct later
Sections and are not claimed as completed here.

The reference adapter accepts the frozen `TokenizerProtocol`; the current
qualified reference fixture uses `ByteTokenizer` / a tiny randomly
initialized `TwelveSixDecoder` in evaluation mode. The tokenizer vocab must
equal the model's vocab. Zero-length prompts and context overflow are refused
instead of padding, clipping, truncating or selecting a different tokenizer.
It does not initiate training, download weights, or use paid compute.

### GenerationConfig

* `max_new_tokens` is exact (0..32768), subject to available model context.
* `strategy="greedy"` forbids silently ignored temperature, top-k and seed;
  explicitly uses temperature 1, top-k None, seed None.
* `strategy="sample"` requires a nonboolean bounded seed and finite positive
  temperature; top-k, when set, is a positive integer no larger than vocab.
  A request-private `torch.Generator` is seeded deterministically.
* `decode_errors="replace"` is the explicit default for arbitrary emitted
  byte tokens; `strict` fails on incomplete/invalid UTF-8. Stream events
  carry **cumulative** text rather than fragile UTF-8 byte-fragment deltas:
  replacement glyphs may be revised when later bytes complete a character.
  Token IDs are always canonical.

### Identity, streaming, cancellation and drift

`ReferenceInference.start(prompt, config)` binds actual bytes of every
model state tensor (SHA-256), ModelSpec SHA-256, tokenizer semantic
identity, generation config and the prompt/token IDs. These bind a stable
`request_id` across streamed and nonstreamed calls using identical inputs.
A restart with the same loaded weights produces the same IDs and tokens.

`ReferenceInference.generate` drains the **same** single-use state machine
returned by `ReferenceInference.stream`; each TOKEN event provides sequence,
token ID, cumulative token IDs/text and full request identity. The last
COMPLETED event matches the nonstream GenerationResult including its
result SHA-256. For a cancelled request, a terminal CANCELLED event keeps the
exact emitted prefix and identity but has a **different** result digest; no
remaining token is generated or falsely marked complete. A completed session
cannot be cancelled or replayed. An interrupted live stream is not a
checkpoint; new invocation with identical model/config/seed deterministically
replays the whole request.

Model.eval() is mandatory, state bytes are rechecked on each new request and
between stream tokens; drift or nonfinite weights fail closed rather than
silently changing model provenance. This is a trusted local reference runtime,
not concurrent arbitrary-code isolation or production lock/load management.
Serving process isolation, capacity and hot swaps are handled in Plan-4
Sections 6–9, not by this component.

## Evidence

Focused automated qualification:

```sh
python -m pytest -q tests/test_inference_runtime_plan4.py \
  tests/test_evaluation_harness_plan4.py tests/test_evaluation_vault_plan4.py
python -m py_compile tools/inference_runtime.py
ruff check tools/inference_runtime.py tests/test_inference_runtime_plan4.py
```

Coverage: reference greedy/sampled logits-to-token parity; deterministic
seed replay; stream/nonstream terminal equality; zero-token requests;
cancellation before/during generation; single-use and incomplete session
failure; restart parity with reloaded model weights; weight drift before/during
generation; nonfinite state; mode/seed/temperature/top-k/unknown-policy
negative cases; prompt/context/vocab constraints; UTF-8 strict/replace
semantics; causal prefix violation. The independent Plan-4 workflow binds
these tests on PR/main. Existing Section1 sealed vault is independent and
not bypassed. No champion, training performance, OS sandbox, model gateway
or whole-product acceptance is asserted.
