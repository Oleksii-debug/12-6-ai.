# NEXT100-044 — FastAPI external-real code source admission

Worker: `NEXT100-044-CODE-FASTAPI`

Execution profile: `LOCAL_FREE`

## Scope

This authority qualifies one bounded FastAPI source-code family for model-training and redistribution use. It is stacked on terminal DATA-227 and does not modify the incumbent HTTPX/Requests code objects or DATA-300's frozen source inventory.

Pinned upstream:

- canonical repository: `fastapi/fastapi`
- exact commit: `49033471594ea5d99a80abdf1043231b7791ee49`
- root tree: `c2a8e5511f6a7174e9b5fb72a513288128011287`
- family: `github:fastapi/fastapi`
- independent family credit: exactly 1

Exact selected first-party Python implementation objects:

1. `fastapi/sse.py` — Git blob `c31334835032570d8244526a623ac249ffc77284` — 7,083 bytes.
2. `fastapi/exceptions.py` — Git blob `d7065c52fe20220e12b7d20db4da7cbeadaf171a` — 7,453 bytes.
3. `fastapi/datastructures.py` — Git blob `1da784cf0927ed55ec6abeb051d89a6ce1e90630` — 5,321 bytes.

Total bounded source bytes: 19,857.

`fastapi/encoders.py` is deliberately excluded because the exact file explicitly marks portions as “Taken from Pydantic v1 as is” and “Adapted from Pydantic v1”. This authority avoids that mixed-provenance fragment rather than treating the FastAPI root license alone as sufficient evidence for it.

Tests, fixtures, docs, `docs_src`, generated/vendor material, metadata-only files and evaluation objects are excluded. In particular, no test fixture containing credentials, mock secrets, or private endpoints is part of this admission.

## Rights

The exact root `LICENSE` is pinned by Git blob `3e92463e6bd522a2a21e5f0a80d8089d6c4be20d` (1,086 bytes) and verified as the MIT License.

The reviewed MIT grant permits use, copy, modification, merging, publication, distribution, sublicensing and sale of copies. Under the project's explicit model-training rights policy, the three exact enumerated code objects are approved for model-training use and redistribution. Redistribution must retain the FastAPI copyright notice and MIT permission notice in all copies or substantial portions.

This is a purpose-specific project decision. Public GitHub availability alone is not treated as training authority.

Evaluation use is `NOT_ADMITTED`.

## Immutable acquisition and validation

The dedicated exact-head workflow reacquires only the pinned license and three pinned raw GitHub objects. Before any object can be marked admitted it verifies:

- canonical upstream repository identity, non-fork and non-mirror state;
- exact upstream commit/tree policy binding;
- exact source Git blob SHA-1 and byte size;
- exact MIT license Git blob SHA-1, byte size and grant/notice markers;
- strict UTF-8 identity-preserving code normalization;
- Python `ast.parse` validity;
- absence of private keys, GitHub/OpenAI/AWS token patterns, JWT-like material and credential-like literal assignments;
- absence of email-like personal data;
- absence of URL endpoints resolving to loopback/private/link-local/reserved IP space, localhost or `.local`;
- absence of file-local alternate-license markers;
- content-addressed snapshot materialization outside Git.

A finding in any required gate is terminal rejection for that object; the workflow does not silently scrub or repair source code.

## Deduplication and family identity

All three FastAPI paths collapse to exactly one source family: `github:fastapi/fastapi`.

The authority fails closed on:

- internal exact SHA-256 duplicates;
- internal 5-token-shingle Jaccard similarity `>= 0.85`;
- exact raw SHA-256 collision with the current DATA-287 registry;
- 5-token-shingle Jaccard similarity `>= 0.85` against the incumbent DATA-227 code objects:
  - `encode/httpx@b5addb64f0161ff6bfe94c124ef76f6a1fba5254:httpx/_content.py`;
  - `psf/requests@5460f467b02e49471c0fd6cfc9ca0adab6351f98:src/requests/_internal_utils.py`;
- pre-existing `github:fastapi/fastapi` family identity in the bound registry.

Vendored or generated files cannot create family credit or capacity.

## Evaluation boundary

The review binds EVAL-303 head `5e5a1de3b594cee5612e63d3d4c2a70499740ac7` and its terminal EVAL-292 code component:

- EVAL-292 head: `2cbe2f2d9c74984baa69e49e520e2280fc76421b`
- dedicated run: `32967204390` = success
- code selection set identity: `9fd52e879c388f06f0b103afa02d68678388867c81cfb0f27ddbf0ca18867054`
- current code evaluation records: 0

Therefore the bounded FastAPI inventory has zero overlap with the current code selection-validation set. Every FastAPI object is explicitly `benchmark_material=false`, `held_out=false`, `reserved_for_evaluation=false`, and `evaluation_use=NOT_ADMITTED`.

This admission does not grant future evaluation use. Any later non-empty code evaluation authority must rerun reserved-object decontamination before corpus promotion.

## Authority state and promotion boundary

The source-specific verdict is `ADMIT` only when the dedicated `NEXT100-044 FastAPI Code Admission` workflow completes successfully at the exact PR head and a final live-registry concurrency check finds no conflicting FastAPI authority or duplicate selected blob identity.

A green source admission does not mutate DATA-300's frozen five-object candidate inventory. A successor source/corpus registry must explicitly consume this terminal authority and rerun its applicable quality, privacy, dedup, decontamination, diversity and release gates.

No model training is performed by this worker.
