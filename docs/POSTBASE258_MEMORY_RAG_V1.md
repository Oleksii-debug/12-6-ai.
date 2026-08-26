# POSTBASE-258 — First-party memory/RAG v1

Status: CANDIDATE IMPLEMENTATION, LOCAL_FREE.

This package adds a non-neural long-term memory substrate. It never imports the model or Torch and therefore cannot update Base weights. It uses only Python's standard library at runtime.

## Stores

Five physically separate SQLite tables represent verified facts/observations, user/project memory, research documents, hypotheses, and experiment results. Every record carries provenance, a timezone-aware timestamp, semantic version number, confidence, verification state, a SHA-256 content/provenance/version hash, and bidirectional supersession links.

## Retrieval contract

`LexicalRetriever` is the mandatory first path. It performs deterministic BM25-style lexical scoring and resolves ties by `memory_id`. The output is a tuple of `EvidenceObject` values with their provenance and integrity metadata. The reasoning boundary receives these evidence objects directly; no untraceable prompt blob is constructed.

An `EmbeddingAdapter` protocol exists only as a pluggable reranker. It is disabled by default and retrieval fails closed if embedding reranking is requested without an adapter.

## Conflict, invalidation, deletion

Conflict detection is deliberately structured rather than LLM-inferred: active records with the same normalized `claim_key` and different `claim_value` values produce `ConflictEvidence`. Superseded records remain directly auditable but are excluded from ordinary retrieval. Invalidation rejects a record without retraining; deletion removes it and cleans supersession links.

## POSTBASE-255 boundary

`Postbase255ReasoningAdapter` accepts `Sequence[EvidenceObject]`. `MockPostbase255Adapter` proves the handoff deterministically in tests and records evidence IDs plus hashes. It does not emulate or pre-empt POSTBASE-255 reasoning policy.

## Verification

Run without network access or third-party packages:

`PYTHONPATH=src python tests/test_memory_rag_postbase258.py -v`

The test corpus contains only synthetic public fixture strings; no secrets or raw private data are ingested.
