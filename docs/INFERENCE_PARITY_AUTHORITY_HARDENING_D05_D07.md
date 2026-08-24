# D05/D07 inference parity authority hardening

Status: **EXPERIMENTAL / parity infrastructure only**.

This package hardens the existing backend-neutral D07 parity harness. It does not implement a second model architecture, a new sampler, a checkpoint loader, an HTTP server, or an alternative runtime.

## Defect

Before this change, a parity report could be structurally green without comparing any model logits. The clearest case was `max_new_tokens=0`: `compare_backends()` accepted the value, executed zero numerical steps, accumulated no failures, and therefore returned `passed=True` with `steps_compared=0`.

Related Python/runtime edge cases could reach the same authority gap:

- `max_new_tokens=True` was accepted as integer `1` because `bool` subclasses `int`;
- two backends with the same invalid `max_context_tokens=True` could satisfy the equality check and leave a prompt with no numerical capacity;
- a prompt that exactly filled the shared context window could complete parity with zero logits calls;
- identical empty logit vectors were accepted by the vector comparator until later greedy selection;
- identical invalid EOS contracts were not rejected centrally;
- backend `encode()` results outside the declared `list[int]` protocol were consumed without an explicit parity-contract failure;
- token IDs or EOS identities outside the runtime logit vocabulary were not independently rejected by the parity harness.

A zero-step PASS is unsuitable as evidence for canonical-vs-converted inference equivalence.

## Strengthened contract

`compare_backends()` now requires:

- a non-empty list/tuple of string prompts;
- `max_new_tokens` to be a positive integer and never a boolean;
- finite non-negative real `atol`/`rtol`, without bool or text coercion;
- both backends to expose a positive integer `max_context_tokens`;
- each EOS identity to be `None` or a non-negative integer;
- backend `encode()` to return `list[int]` containing non-negative, non-boolean token IDs;
- non-empty equal-length logit vectors for every numerical step;
- input tokens and shared EOS identity to fit the runtime logit vocabulary;
- at least one real numerical parity step for a prompt that otherwise had no failure.

`ParityReport.passed` independently requires `prompts_compared > 0` and `steps_compared > 0` in addition to an empty failure set, so even a manually reconstructed report cannot claim a vacuous PASS through the normal report property/JSON serialization.

Existing strict behavior remains: prompt-token equality, context/EOS equality, NaN rejection, non-matching infinity rejection, explicit atol/rtol logit comparison, greedy-token equality, and decoded-output equality.

## Ownership / collision boundary

This change is limited to:

- `src/twelve_six/inference/parity.py`;
- `tests/test_inference_parity_authority.py`;
- this document.

It deliberately does not edit concurrent D05/D07 surfaces:

- first-party atomic checkpoint loading;
- replay/acceptance/retained evidence;
- HF-style export and export adapters;
- generation/sampling/OpenAI request semantics;
- local HTTP server transport;
- Windows CLI transport.

Downstream alternative-backend or export parity workers may consume the stronger harness without treating this package itself as proof that any external runtime is compatible.

## Truth boundary

A green parity harness only means the requested backend pair performed non-vacuous comparisons under the declared tolerances. It does **not** by itself prove Transformers, vLLM, GGUF, llama.cpp, GPU, Windows/NVDA, distributed, or cross-hardware parity.

No audit verdict or S0 promotion authority is created here. Canonical Base remains random-initialized and pretraining-only. No foreign pretrained weights, instruction/alignment/refusal/ethics/personality/domain-specialization behavior, materially paid compute, CANDIDATE, AUDITED_CANDIDATE, or STABLE status is introduced.
