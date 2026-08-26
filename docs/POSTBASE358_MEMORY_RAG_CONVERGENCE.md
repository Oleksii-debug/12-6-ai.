# POSTBASE-358 — Memory/RAG convergence

Status: CONVERGENCE CANDIDATE, LOCAL_FREE.

This worker converges the accepted SQLite/BM25-style POSTBASE-258 memory substrate onto the incumbent POSTBASE-253 post-Base communication-consumption boundary. It does not mutate, copy over, fine-tune, or relabel Base model weights.

## Provenance and integrity

Every memory record carries structured `Provenance`, a timezone-aware timestamp, version, confidence, verification state, and a SHA-256 digest over content + provenance + version. Required provenance identifiers must be non-empty. Reads and retrieval recompute the digest and fail closed with `MemoryIntegrityError` if stored content, source identity, source version, locator, or version no longer matches the recorded digest.

## Conflict detection

Conflict detection remains deterministic and non-neural. Active records are grouped by Unicode-NFKC/case/whitespace-normalized `claim_key`. Distinct normalized claim values produce a structured `ConflictEvidence` object with deterministic `memory_id` ordering. Equal values that differ only by normalization are not reported as contradictions.

## Supersession and deletion

Superseded records are excluded from normal retrieval while remaining directly auditable. The database stores the predecessor's pre-supersession verification state so deleting the final successor can restore the predecessor exactly. Deleting a middle node rewires predecessor/successor links directly, preserving a coherent lineage. Explicit invalidation wins over later successor deletion and is never silently undone.

The added `supersession_base_verification` column is migrated into pre-convergence SQLite tables with `ALTER TABLE`; existing rows and content hashes are retained.

## Deterministic lexical retrieval

`LexicalRetriever` is the mandatory first path. Token and claim normalization are Unicode NFKC + casefold. BM25 scoring is local and deterministic for a fixed database/query. Equal scores are ordered by `memory_id`. Optional embedding reranking remains disabled by default and fails closed when requested without an adapter.

## Evidence-object handoff

Retrieval returns immutable `EvidenceObject` values containing source provenance, integrity hash, verification state, lexical score, conflict evidence, and supersession links. `feed_reasoning()` hands that typed sequence directly to the post-Base reasoning protocol; no opaque concatenated prompt blob is constructed. The deterministic mock verifies IDs, hashes, and source IDs at the handoff boundary without implementing reasoning policy.

## Privacy, compute, and model boundary

Tests use only synthetic project-authored fixture strings such as `fixture-public-001`; no private user data, secrets, production memory, or external corpora are admitted. Runtime uses Python stdlib (`sqlite3`, hashing, regex/math/dataclasses) and makes no network call. No Torch/model import exists in the memory package, and this worker changes no model/checkpoint/weight path.

## Verification

Focused local command:

`PYTHONPATH=src python tests/test_memory_rag_postbase358.py -v`

Expected convergence suite: 13 tests PASS.

Source candidate: POSTBASE-258 head `030b50464cd71ca77a406f3fba0e211390d5e67f`.

Incumbent post-Base boundary used as branch base: POSTBASE-253 head `f6463424b5f53152fce6e6053b705f94e03f9f06`.
