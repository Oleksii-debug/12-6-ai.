# Plan 5 / Section 1 — Context management

## Contract

- Python entrypoint: `twelve_six_agent_runtime.context.build_context`.
- Versioned provider-neutral envelope: `12-6.agent-context.v1`.
- Trusted caller supplies `ContextEntry(entry_id, source_id, source_kind, recency, priority, content, critical)`; generated/retrieved content must never self-declare trusted source identity or `critical`.
- Critical content may be pinned only for verified `system` / `owner` sources. Every protected entry is retained **byte for byte**, or the operation fails closed with `ContextError`.
- Entries are selected by descending priority, then descending recency, then entry ID. The final selected context is ordered by increasing recency/ID for reproducible replay.
- `max_bytes` bounds the actual canonical UTF-8 JSON envelope; `max_entries` bounds included records. Noncritical records may be omitted, never silently paraphrased by a model.
- `omitted_count` and `omitted_sha256` attest discarded entries in deterministic ID order. Source identity stays with every retained item.
- Invalid identities, duplicate entry IDs, invalid priorities, type confusion and an untrusted critical bit reject the entire assembly.
- No model, tokenizer, checkpoint, GPU, external service, Nika product state or paid compute is required.

## Existing-work audit and boundaries

PR #155 contains an older model-neutral tool/executor runtime on a separate, stale lineage. It does **not** contain a bounded context authority. Section 1 adds only the missing context module beside that existing namespace; it does not fork tool execution, turn old PR #155 into new Plan-5 closure evidence, or interfere with other plans.

## Autonomous qualification

Focused check: `PYTHONPATH=src pytest -q tests/test_agent_context_section1.py`.
The six test cases cover 400-turn compaction, critical invariant retention, byte cap, model-independent reproducibility, priority/recency, malformed and adversarial identities, fail-closed insufficient budget, and empty/oversized inputs.

The fixture validates Plan-5-owned component behavior, **not** production model quality or whole-product integration. Terminal closure requires integration and readback on `main`.
