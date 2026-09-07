# First-party S0 inference and generation

This package binds the accepted S0 lane contracts without reimplementing model architecture or checkpoint serialization:

- D01: `ModelSpec` and `TwelveSixDecoder`.
- D04: canonical `s0-byte-v1` tokenizer and vocabulary identities.
- D05: verified checkpoint format, checksums, lineage and load semantics.
- D07: `InferenceBackend`, generation, sampling, stop/context semantics and CLI.
- D01 convergence runtime: `S0TorchInferenceBackend` is reused as the model/tokenizer adapter rather than duplicated.

The canonical loader is:

```text
twelve_six.inference.first_party:load_first_party_backend
```

`twelve-six-generate` now uses that loader by default. An explicit `--backend-loader MODULE:CALLABLE` remains available for controlled alternative-backend and conversion testing.

## Loading contract

`load_first_party_backend(CHECKPOINT)` performs the following fail-closed sequence before inference:

1. verify the checkpoint manifest checksum, every recorded artifact size/SHA-256, manifest identities and checkpoint ID;
2. reconstruct and validate `ModelSpec` from the checkpoint identity;
3. require the semantic `ModelSpec` hash and parameter count to match the manifest;
4. require a bound training `context_length`, when present, to equal `ModelSpec.max_seq_len`;
5. instantiate the canonical D04 byte tokenizer and require its config hash, vocabulary hash, version and vocabulary size to match the checkpoint/model contract;
6. construct `TwelveSixDecoder` from the verified spec and load weights with strict state-dict compatibility;
7. leave training RNG state untouched (`restore_rng=False`) and switch the model to evaluation mode.

The S0 byte tokenizer has no EOS/BOS token. Generated token IDs are therefore the exact output identity. User-facing decode uses the accepted runtime adapter's UTF-8 replacement policy so arbitrary random/pretraining bytes do not crash a Windows/NVDA console; this rendering policy does not alter logits or selected token IDs.

## CLI

Greedy completion:

```text
twelve-six-generate --checkpoint CHECKPOINT --prompt "12-6" --greedy --max-new-tokens 32
```

Seeded sampling:

```text
twelve-six-generate --checkpoint CHECKPOINT --prompt "12-6" --sample --temperature 0.8 --top-k 32 --top-p 0.95 --seed 17
```

Stdin and JSON diagnostics:

```text
type prompt.txt | twelve-six-generate --checkpoint CHECKPOINT --json
```

The interface uses ordinary stdin/stdout/stderr only: no TUI, cursor addressing, ANSI control requirement, mouse interaction or GUI. JSON includes generated token IDs plus privacy-safe checkpoint/model/tokenizer identities. Prompt text is not echoed in diagnostics.

## Deterministic parity

The parity harness is available without another console-script dependency:

```text
python -m twelve_six.inference.parity \
  --reference-checkpoint CANONICAL \
  --reference-backend-loader twelve_six.inference.first_party:load_first_party_backend \
  --candidate-checkpoint CANDIDATE \
  --candidate-backend-loader PACKAGE:LOAD_CANDIDATE \
  --prompt "12-6" \
  --max-new-tokens 8 \
  --atol 1e-6 \
  --rtol 1e-5 \
  --json
```

It compares context/EOS contracts, prompt token IDs, per-step logits, greedy token IDs and decoded output. NaN, incompatible infinities, shape drift, greedy divergence and decode drift fail. Sampling is tested for seed repeatability separately; conversion parity is intentionally based on deterministic logits/tokens rather than assuming bit-identical sampled trajectories after small numerical perturbations.

## Local OpenAI-compatible server handoff

`src/twelve_six/inference/openai_compat.py` is a dependency-free request/response seam for a future local HTTP layer exposing `POST /v1/completions`. It supports one raw text prompt, `max_tokens`, `temperature`, `top_p`, `seed`, and stop strings and returns the standard text-completion fields plus exact token usage.

It intentionally does not implement `/v1/chat/completions`, role messages, system prompts, instruction templates, hidden prefixes, refusal behavior or other post-training semantics. `messages`, streaming, `n != 1`, echo and logprobs are rejected rather than silently approximated. A later local web server can call `completion_response()` and supply request-specific response IDs/timestamps without changing Base completion semantics.

## Evidence boundary

`tests/test_first_party_inference.py` performs a LOCAL_FREE CPU integration fixture using the canonical S0 10,140-parameter `ModelSpec`: it trains a real S0 model for one optimizer step, saves a D05 checkpoint, reloads through the first-party adapter, and checks logits, greedy output, seeded sampling, stop/context behavior, deterministic parity, CLI/stdin/JSON, corruption/incompatibility rejection and raw completion handoff.

That test is real train -> checkpoint -> verified reload -> generation execution, but its temporary checkpoint is a CI test artifact, not a promoted or durable trained S0 release checkpoint. No STABLE, external-serving, Windows/NVDA live-run, vLLM, Transformers, GGUF or llama.cpp parity claim is made by this package.
