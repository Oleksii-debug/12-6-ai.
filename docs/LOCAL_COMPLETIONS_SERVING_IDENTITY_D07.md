# D07 local completion serving identity

## Problem

The S0 local completion server already loads a verified first-party checkpoint before binding a socket and preserves raw pretraining-Base semantics. Before this change, HTTP clients could identify only the configured `model_name` such as `12-6-base`. Exact checkpoint lineage was available through first-party backend diagnostics, but only startup diagnostics exposed it.

That creates an evidence gap for long-running or automated local tests: two different valid checkpoints can be served under the same model name, while `/healthz`, `/v1/models`, completion responses, and errors do not tell a client which exact checkpoint/runtime identity the server captured.

## Server-lifetime identity

`CompletionHTTPServer` now captures backend diagnostics exactly once during construction, before serving requests. Only a fixed privacy-safe field allowlist is retained:

- backend kind;
- checkpoint ID and Git SHA;
- ModelSpec identity and parameter count;
- vocabulary/context limits;
- tokenizer version/config/vocabulary identities;
- dataset and run-manifest identities;
- checkpoint step/tokens seen;
- device.

The selected object is encoded as canonical JSON and SHA-256 hashed. The resulting `serving_fingerprint` is immutable for the lifetime of that server object even if a backend later returns different diagnostics.

This is a serving-provenance fingerprint, not a new checkpoint format and not an alternative to the D05 checkpoint ID.

## HTTP exposure

When a backend exposes diagnostics, every JSON HTTP response includes:

- `X-12-6-Serving-Fingerprint: <sha256>`;
- `X-12-6-Checkpoint-ID: <checkpoint_id>` when checkpoint ID is exact lowercase SHA-256.

`GET /healthz` additionally returns the complete allowlisted `serving_identity` and `serving_fingerprint`.

`GET /v1/models` keeps the configured OpenAI-compatible model ID and adds a `metadata` object containing the fingerprint and checkpoint ID.

`POST /v1/completions` keeps the existing response shape; the exact served identity is carried by headers rather than changing completion semantics.

Generic test/alternative backends that do not implement `diagnostics()` retain the prior HTTP contract and receive no synthetic checkpoint identity.

## Why capture once

The server is constructed only after first-party checkpoint load succeeds. Capturing one diagnostic object at server construction means health/model/completion/error responses all refer to the same server-lifetime provenance. A later mutation of an object, path, or diagnostic producer cannot silently alter the identity advertised by an already-running server.

This is intentionally stronger than evaluating `backend.diagnostics()` on every request, which could make one server appear to change checkpoint identity over time without an explicit restart.

## Scope and truth boundary

This package does not change D01 architecture, D02 training, D03 data, D04 tokenization/packing/evaluation, D05 serialization, D07 sampling/generation semantics, D08 environment locks, or D10 promotion authority.

It does not add chat/system/instruction behavior, authentication/TLS, public-server hardening, streaming, batching, KV-cache acceleration, or alternative backend parity. The server remains a minimal local/raw-Base handoff. The new headers and model metadata are 12-6 diagnostic extensions, not claims of complete OpenAI API compatibility.

Canonical Base remains random-initialized and pretraining-only. No paid compute or promotion authority is introduced.
