# NEXT100-083 Hypothesis Memory/RAG Integration

Worker: `NEXT100-083-HYPOTHESIS-MEMORY-RAG`

Execution boundary: `LOCAL_FREE`.

## Consumed authorities

- Hypothesis search V1: `postbase256/hypothesis-search-v1-20260826` at
  `ea1d8fff0d3235660dffe7ba411e192df83f5e1d`.
- Converged SQLite/BM25 memory/RAG:
  `postbase358/memory-rag-convergence-20260826` at
  `976adda1cfe981d7b6363d267854759bee802006`.

The integration branch is based on the converged memory/RAG head and imports the terminal
POSTBASE-256 hypothesis-search implementation unchanged.

## Contract

`HypothesisMemoryRAG` performs deterministic lexical retrieval through
`LexicalRetriever.retrieve(..., use_embedding_adapter=False)` and binds retrieved
memory evidence to an active hypothesis through a caller-supplied deterministic
relation resolver.

Each immutable binding preserves:

- memory ID;
- human-readable source identity;
- full structured provenance;
- memory store;
- timestamp;
- memory version;
- integrity/content hash;
- verification state;
- supersedes and superseded-by lineage;
- BM25 lexical score;
- confidence;
- conflict evidence;
- support/contradiction relation;
- memory state used for score eligibility.

No semantic relation is inferred by a model. The relation resolver is a local,
deterministic function supplied by the caller.

## Fail-closed stale-memory rule

Before scorecard calculation, preferred-hypothesis selection, binding export, or
integration export, all active bindings are revalidated against SQLite.

A binding is removed from the active score overlay when its memory becomes:

- superseded;
- deleted;
- rejected/invalidated;
- integrity-invalid;
- identity-mismatched against the originally bound version/hash/provenance/store.

A stale binding is never silently reactivated. A memory that later becomes eligible
again must be retrieved and explicitly rebound.

This avoids leaving a stale positive contribution inside the immutable hypothesis
search score history.

## Selection semantics

The underlying POSTBASE-256 heuristic score is retained as `base_score`.
Active memory evidence contributes a deterministic bounded overlay:

- support: positive delta;
- contradiction: negative delta;
- irrelevant: no binding.

The resulting adjusted score is used only by the integration selector. This keeps
hypothesis-search history intact and avoids conflicts with independent verifier
integration.

## Deterministic fixture

`tools/run_next100_083_hypothesis_memory_rag.py` starts with a deliberately wrong
preferred hypothesis:

- wrong: `addition before multiplication`, score `0.85`;
- correct: `multiplication before addition`, score `0.60`.

A synthetic verified SQLite memory is retrieved by BM25. A deterministic resolver
classifies it as contradiction for the wrong hypothesis and support for the correct
hypothesis. The adjusted ranking must therefore select the correct hypothesis.

Tests also prove that superseded, deleted, and integrity-failed memories stop
contributing without another retrieval call, and that an installed embedding adapter
is never invoked.

## Explicit exclusions

- no embeddings required or invoked;
- no network retrieval;
- no external model judge;
- no paid service;
- no private scratch reasoning in traces.
