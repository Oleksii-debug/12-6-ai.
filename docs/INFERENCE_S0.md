# S0 inference and local test contract

D07 owns the user-facing generation boundary. S0 deliberately uses a plain Python CLI and does not claim to be a chat product.

## Backend boundary

A checkpoint adapter must provide `eos_token_id`, `encode(text)`, `decode(token_ids)`, and `next_token_logits(input_ids)`. The CLI loads that adapter through `--backend-loader MODULE:CALLABLE`; the callable receives the local checkpoint `Path`. This keeps D07 independent of the final D01 model class and D05 serialization shape while those lanes are still being implemented.

Once D01/D05 publish the canonical checkpoint/model loader, D07 should add a first-party adapter and make it the normal path. The generic loader remains useful for conversion comparison and compatibility tests.

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

No real 12-6 checkpoint exists in this branch, so real checkpoint loading, canonical-logit comparison, Transformers loading, vLLM, OpenAI-compatible serving, KV-cache behavior, and GGUF/llama.cpp are not claimed here. Those require committed D01/D05 integration surfaces.
