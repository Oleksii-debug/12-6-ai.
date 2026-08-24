# D05/D07 inference backend conformance gate

## Why this exists

The repository already has a backend-neutral D07 generation API and a parity harness.
Parity is intentionally comparative: it can prove that a candidate backend agrees with
a reference backend. It should not be the first place malformed backend mechanics are
discovered.

`python -m twelve_six.inference.conformance` is the single-backend gate before parity,
conversion acceptance or serving evidence. It does not duplicate model architecture,
tokenization, generation, sampling, checkpoint loading, parity or HTTP code. It calls
the existing `InferenceBackend`, loader, canonical greedy sampler and generation path.

## Contract proved

For privacy-safe probe prompts the gate requires:

- a structurally valid D07 `InferenceBackend`;
- positive, non-boolean `max_context_tokens`;
- `eos_token_id` is `None` or a non-negative, non-boolean integer;
- `encode()` returns a non-empty `list[int]` with deterministic repeated output;
- `decode()` returns deterministic text for repeated token IDs;
- `next_token_logits()` returns one non-empty finite numeric vocabulary vector;
- repeated logits calls keep vocabulary width and stay within configured absolute
  repeat tolerance (zero by default);
- vocabulary width is stable across all probes;
- every encoded token ID and optional EOS is inside the inferred logits vocabulary;
- the existing canonical greedy sampler returns an in-vocabulary token;
- at least one probe leaves room for, and successfully executes, one real call through
  the existing D07 `generate()` path.

The gate hashes prompt text, prompt token IDs, decoded text, per-probe float64-hex logits
and generated text. Literal prompt or generated text is not written into the report.

## CLI

Canonical first-party checkpoint:

```text
python -m twelve_six.inference.conformance CHECKPOINT --json
```

Alternative backend:

```text
python -m twelve_six.inference.conformance CHECKPOINT \
  --backend-loader package.module:load_backend \
  --prompt "probe one" \
  --prompt "probe two" \
  --repeat-atol 0 \
  --json
```

Plain mode emits one ASCII-compatible PASS line with backend type, inferred vocabulary,
context, probe count and report hash. JSON mode emits
`12-6.inference-backend-conformance.v1`.

A nonzero `repeat_atol` is an explicit mechanics tolerance for repeated calls to the
same backend in the same runtime. It is not the reference-vs-candidate tolerance used
by the parity harness and does not authorize cross-hardware reproducibility claims.

## Fail-closed report boundary

`validate_conformance_report()` validates exact report/probe/generation/check schemas,
numeric ranges, digest syntax, vocabulary/EOS/token relationships, one-step generation
semantics and the canonical whole-report SHA-256. Rehashing a report after changing
`parity_proven`, `checkpoint_identity_proven` or `promotion_authority` still fails.

The report deliberately states:

- `parity_proven=false` — use D07 parity for comparison;
- `checkpoint_identity_proven=false` — use the D05 verified loader/evidence for artifact
  identity;
- `promotion_authority=false` — this gate has no D06/D10/audit authority.

## Relationship to active late-wave work

This package is additive and intentionally does not edit the currently active
first-party loader, generation/sampling/OpenAI request contract, server, Windows
transport, retained checkpoint evidence, acceptance evidence or HF export surfaces.
It is intended to become the reusable preflight for future Transformers/vLLM/GGUF/
llama.cpp or other backend adapters before a parity claim is attempted.

## Base and compute truth boundary

Canonical early Base remains random-initialized and pretraining-only. This package adds
no foreign pretrained weights, instruction/alignment/refusal/ethics/personality/domain
behavior and runs no materially paid compute. A conformance PASS is mechanics evidence
only; it is not model quality, serving compatibility, Windows/NVDA live execution,
AUDIT-A/B PASS, CANDIDATE or STABLE evidence.
