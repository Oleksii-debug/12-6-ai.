# D05 first-party inference replay evidence

This package adds a durable evidence layer around the existing D01 + D04 + D05 -> D07 first-party inference path. It does not replace or fork the canonical generation, sampling, checkpoint, tokenizer, model, or server implementations.

## Purpose

The current S0 convergence lineage already proves real training, strict checkpoint save/load and resume, verified first-party reload, greedy/seeded generation, stop/context semantics, direct-vs-reloaded parity, and a local raw `/v1/completions` server. The remaining evidence problem is that those facts are spread across test logs and stage reports and do not provide one independently replayable inference transcript suitable for later conversion/backend comparisons.

`python -m twelve_six.inference.evidence` therefore records a self-hashed `12-6.first-party-inference-evidence.v1` envelope containing:

- exact D05 checkpoint ID and checkpoint Git SHA;
- D01 ModelSpec identity and parameter count;
- D04 tokenizer config, vocabulary and version identities;
- exact context/vocabulary runtime contract;
- dataset and run-manifest identities carried by the checkpoint;
- checkpoint step/tokens-seen and runtime device;
- generation config for every probe;
- prompt UTF-8 hash and exact prompt token IDs, but no literal prompt field;
- generated token IDs and output UTF-8 hash;
- stop reason;
- per-logits-call input-token hash, logit count, exact float64-hex logit fingerprint and greedy token;
- per-probe self-hash plus whole-envelope SHA-256;
- explicit non-promotion/raw-Base truth boundary.

The trace backend only delegates to the existing `generate()` path and fingerprints `next_token_logits()` calls. Sampling and stopping semantics remain owned by the canonical D07 implementation.

## Collection

Example for a verified checkpoint:

```text
python -m twelve_six.inference.evidence collect \
  --checkpoint CHECKPOINT_DIR \
  --prompt "12-6" \
  --prompt "Base" \
  --max-new-tokens 8 \
  --sample-seed 23 \
  --output inference-evidence.json
```

For every prompt, collection records a greedy probe. Supplying `--sample-seed` adds a seeded-sampling probe through the same canonical generator.

The JSON artifact contains exact prompt token IDs. For S0 raw-byte tokenization those token IDs can reveal prompt content to a determined reader, even though literal prompt text is omitted. Do not use secret or private prompts in evidence intended for publication.

## Independent replay

```text
python -m twelve_six.inference.evidence replay \
  --checkpoint CHECKPOINT_DIR \
  --evidence inference-evidence.json
```

Replay fails closed if any of these differ:

- checkpoint/runtime diagnostics;
- self-hash or per-probe hash;
- truth-boundary claims;
- generation-config schema;
- prompt token/hash round trip;
- generated tokens;
- stop reason;
- step count/input hash;
- exact logit fingerprint or greedy token;
- decoded output hash.

`--json` returns a minimal machine-readable PASS report after successful replay.

## Intended downstream use

This evidence is the canonical first-party replay anchor for future Transformers/vLLM/GGUF/llama.cpp or other backend work. Alternative backends should still use the existing tolerance-aware `twelve_six.inference.parity` harness; this D05 artifact proves what exact first-party checkpoint/run was used as the reference and detects reference drift before numerical parity comparison.

## Truth boundary

This package does not claim cross-hardware bitwise reproducibility. Device is recorded and exact replay currently requires the same declared runtime identity. It does not make Transformers, vLLM, GGUF or llama.cpp compatibility claims. It does not authorize paid compute, alter Base behavior, introduce instruction/chat/refusal/personality semantics, or grant AUDIT/CANDIDATE/STABLE promotion authority.

Canonical Base remains random-initialized and pretraining-only for this run.
