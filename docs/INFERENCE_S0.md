# S0 inference and local test contract

D07 owns the user-facing generation boundary. S0 deliberately uses a plain Python CLI and does not claim to be a chat product.

## Backend boundary

A checkpoint adapter must provide `eos_token_id`, positive integer `max_context_tokens`, `encode(text)`, `decode(token_ids)`, and `next_token_logits(input_ids)`. The CLI loads that adapter through `--backend-loader MODULE:CALLABLE`; the callable receives the local checkpoint `Path`.

`max_context_tokens` is a required capability, not an advisory value. D07 rejects prompts that already exceed the backend context window and stops with `context_limit` before it would call the backend with a full/overflowing context. This keeps the generic harness fail-closed for the D01 `max_seq_len` boundary without copying D01 model logic.

Once D01/D04/D05 accepted surfaces are composed, D07 should add a first-party adapter that obtains this value from the accepted ModelSpec, verifies the tokenizer/checkpoint identities, and makes that adapter the normal path. The generic loader remains useful for conversion comparison and compatibility tests.

## Commands

After `pip install -e .`, a backend can be exercised with:

```text
twelve-six-generate --checkpoint PATH --backend-loader PACKAGE.MODULE:load_backend --prompt "text" --greedy --max-new-tokens 32 --seed 0
```

Sampling is explicit:

```text
twelve-six-generate --checkpoint PATH --backend-loader PACKAGE.MODULE:load_backend --prompt "text" --sample --temperature 0.8 --top-k 20 --top-p 0.95 --seed 7
```

`--stop` and `--stop-token-id` are repeatable. Text stop strings are removed by default; use `--keep-stop-string` to retain them. `--json` emits a single machine-readable object on stdout. Human diagnostics go to stderr and never echo the prompt.

When `--prompt` is omitted, stdin is consumed, which works cleanly in PowerShell/CMD pipelines and with screen-reader workflows.

## Windows / NVDA

The S0 interface is keyboard-only text I/O: normal command-line arguments, stdin, stdout, and stderr. There are no mouse-only controls, cursor-positioned terminal widgets, or custom GUI elements. A later web/desktop frontend should consume a stable local API rather than embedding model logic.

## Compatibility roadmap

Transformers-compatible export is preferred as soon as D01/D05 can express the architecture safely. vLLM/OpenAI-compatible serving comes after that model path is accepted by the backend. llama.cpp/GGUF is later and requires architecture/converter support plus canonical-vs-converted logits/generation tolerance checks.

## NOT TESTED in this package

The generic context guard is unit-tested, but no real composed 12-6 checkpoint is loaded in this branch. Real D01+D04+D05 adapter loading, canonical-logit comparison, Transformers loading, vLLM, OpenAI-compatible serving, KV-cache behavior, and GGUF/llama.cpp remain NOT TESTED until those accepted surfaces are composed.
