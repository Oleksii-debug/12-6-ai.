# NEXT100-042 Django Code Source Authority

Worker: `NEXT100-042-CODE-DJANGO`

Execution boundary: `LOCAL_FREE`

Status at branch construction: `PENDING_EXACT_HEAD_RUN`

The only terminal decision this lane may emit is based on the exact-head workflow `.github/workflows/next100-042-django-code-source-admission.yml`, followed by the mandatory late live-registry conflict check. A branch or PR existing by itself is not terminal evidence.

## Bounded upstream identity

Canonical upstream: `https://github.com/django/django`

Release tag: `6.0.8`

Annotated tag object: `22ab6cfe6bfcca9f5fbc1d14fe1b7cb3cf89f375`

Pinned commit: `b2821457e8bfa9e9066837b9f9b3a3955224ab19`

Pinned tree: `7811eb2a3bfbf7a606ae26212d59e3102048581a`

The annotated upstream release tag must resolve to the exact commit above and GitHub must report the release-tag signature as verified. The upstream repository must remain canonical, non-fork, and non-mirror.

## Exact selected implementation objects

Exactly three Python implementation files are in scope. They are one source family, not three independent families.

| Source ID | Path | Git blob SHA-1 | Bytes | Role |
| --- | --- | --- | ---: | --- |
| `code.django.core.signing` | `django/core/signing.py` | `d6be4a5d804916e3d7705018f485c4c374d90cc3` | 9,594 | signing and timestamped serialization implementation |
| `code.django.db.transaction` | `django/db/transaction.py` | `1710d1ef17497c4aef3f4b65498dd539d0ee0050` | 12,506 | transaction and atomic-context implementation |
| `code.django.urls.resolvers` | `django/urls/resolvers.py` | `6c681f9d8d32a35ca512cf3d08b68ab5bc358777` | 32,056 | URL resolution and resolver graph implementation |

Total bounded raw source bytes: `54,156`.

Documentation, tests, generated or vendored assets, and version/bootstrap metadata such as `django/__init__.py` are excluded. No unlisted Django file is admitted by this authority.

## License and rights boundary

Primary Django terms are pinned as `LICENSE`, Git blob `5f4f225dd282aa7e4361ec3c2750bbbaaed8ab1f`, 1,552 bytes, BSD-3-Clause. The license permits redistribution and use in source and binary forms, with or without modification, subject to retention of the copyright notice, conditions and disclaimer, plus the non-endorsement restriction.

Upstream also states that Django includes code from the Python standard library. Therefore `LICENSE.python`, Git blob `2fc28e876fa1fc8c536f46b81789e2bb51b323c0`, 14,256 bytes, is retained in the evidence bundle as a conservative compliance boundary. This authority does not infer that every selected file is Python-derived; it avoids under-compliance where upstream does not provide a per-file mapping for that repository-wide notice.

Project purpose decisions for these exact pinned objects:

- model training: `ALLOWED`;
- redistribution: `ALLOWED`, subject to the retained notices and conditions above;
- evaluation use: `NOT_SEPARATELY_AUTHORIZED`;
- evaluation reservation: `false`.

Public availability is not the rights basis. The project decision relies on the reviewed permissive license grants and the exact-object provenance gates.

## Technical admission gates

For every selected object the verifier reacquires raw bytes from the pinned commit and requires exact byte count and Git blob SHA-1. It computes raw SHA-256 identities, requires strict UTF-8 identity preservation, parses the exact decoded text with Python `ast.parse`, rejects NUL bytes, and runs fail-closed private-key/token/key and email/PII-like literal heuristics.

Deduplication is performed over all three proposed Django objects plus the two incumbent DATA-227 code objects (`github:encode/httpx` and `github:psf/requests`). Exact duplicates are raw SHA-256 equality. Near-duplicates use 5-token shingle Jaccard with threshold `0.85`. Any exact or near collision rejects this admission.

Family identity is `github:django/django`, with family credit exactly `1`. The canonical upstream must not be a fork or mirror and must not collide with an incumbent repository identity.

## Fail-closed terminal rule

`ADMIT` is valid only if all of the following are simultaneously true at the exact project head: upstream tag/commit/tree identity passes; both license objects pass; all three source blob identities pass; strict UTF-8 identity and AST parsing pass; privacy/secret scans are clear; dedup produces zero exact and zero near collisions; the source family is independent from the incumbent registry; training and redistribution remain explicitly allowed; and evaluation remains neither separately authorized nor reserved.

Immediately before the final verdict, the live code registry must be re-read. A newly admitted identical blob, duplicate Django family, or conflicting evaluation reservation invalidates terminality until reconciled.
