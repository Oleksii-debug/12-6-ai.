# S0 local raw completions server

This server is a transport layer over the existing verified D01+D04+D05 -> D07 first-party inference path. It does not implement another model, tokenizer, checkpoint format, sampler, or prompt policy.

## Start

The checkpoint is verified and loaded before the listening socket is created. An incompatible or corrupt checkpoint therefore fails before the server becomes reachable.

```text
python -m twelve_six.inference.server --checkpoint CHECKPOINT --host 127.0.0.1 --port 8000 --json-diagnostics
```

Default bind is `127.0.0.1:8000`. Non-loopback addresses are rejected unless `--allow-non-loopback` is supplied explicitly. The S0 server is intentionally serialized rather than a throughput/concurrency claim.

Startup diagnostics are plain stderr text or one JSON object with `--json-diagnostics`. When the first-party backend exposes diagnostics, the JSON object includes privacy-safe checkpoint, Git, ModelSpec, tokenizer/vocabulary, context and run identities. Prompt text is never included in server logs or startup diagnostics.

## Endpoints

`POST /v1/completions` uses the existing `openai_compat.completion_response()` contract. Supported request fields are one string `prompt`, `max_tokens`, `temperature`, `top_p`, `seed`, and string/list `stop`. The response has the OpenAI text-completion shape with `id`, `object=text_completion`, `created`, `model`, `choices`, and token `usage`.

Example request:

```text
curl -sS http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d "{\"prompt\":\"12-6\",\"max_tokens\":8,\"temperature\":0,\"seed\":17}"
```

Additional local diagnostics endpoints:

- `GET /healthz`
- `GET /v1/models`

The request body is UTF-8 JSON only and is bounded to 1 MiB by default (`--max-request-bytes` can change the bound). Invalid JSON, unsupported media types, oversized requests and generation/configuration errors return structured JSON error objects rather than partial completions.

## Raw Base semantics

This is pretraining-only Base completion. The HTTP layer passes the exact supplied `prompt` to the existing generation path. It adds no system text, role marker, chat template, instruction, refusal, ethics, personality, domain-specialization prefix, or other hidden behavior.

The server deliberately rejects `/v1/chat/completions`. The underlying completion contract also rejects `messages`, `stream=true`, `n != 1`, `echo=true`, and `logprobs` instead of silently approximating unsupported semantics.

Greedy mode is selected by `temperature: 0`; positive temperature uses the existing seeded sampler. Stop strings, context-window limits and finish reasons are inherited from the canonical D07 generation contract.

## Accessibility and platform boundary

The server and diagnostics use ordinary command-line arguments, stdout/stderr and JSON/HTTP only. There is no TUI, cursor addressing, ANSI control requirement, GUI, or mouse-only interaction.

This is code-level Windows/NVDA-friendly I/O, not a live Windows/NVDA PASS. The repository currently has a separately documented Windows checkout blocker because the physical GitHub repository name ends in a period. This package does not hide or repair that repository-identity blocker.

## Evidence boundary

Focused transport tests exercise real loopback HTTP requests and prove raw prompt preservation, deterministic seeded request behavior, stop semantics, OpenAI-shaped completion/model responses, structured errors, body limits, explicit chat/streaming rejection, and loopback-by-default binding.

The integrated parent candidate already proves the strict first-party checkpoint load/reload and real S0 generation path. This server package consumes that path without weakening D05 verification. It does not claim external/public serving hardening, authentication, TLS, streaming, batching, KV-cache performance, vLLM, Transformers, GGUF/llama.cpp parity, paid compute, AUDIT PASS, CANDIDATE, or STABLE status.
