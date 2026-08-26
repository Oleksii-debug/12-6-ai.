# NEXT100-045 — Starlette external-real code source authority

Worker: `NEXT100-045-CODE-STARLETTE`

## Candidate decision

`ADMIT`, contingent only on the dedicated exact-head verifier remaining green and the mandatory final live-registry refresh showing no newly sealed reservation or duplicate identity.

This authority is training-only and `LOCAL_FREE`. It does not execute training and it does not grant evaluation use.

## Immutable upstream boundary

- canonical repository: `Kludex/starlette`
- stable GitHub repository id: `138597372`
- historical transfer alias: `encode/starlette`
- release: `1.6.0`
- exact commit: `4f250d6b814587e20c5365f0a5f0c4d42bcb929f`
- license: `BSD-3-Clause`
- license path: `LICENSE.md`
- license Git blob: `d16a60ec5b9963ef86b35a52ac92227014618e6c`
- license raw SHA-256: `dcb95677a02240243187e964f941847d19b17821cf99e5afae684fab328c19bf`

Selected implementation objects only:

1. `starlette/_utils.py`
   - Git blob: `b9b8639b2106fd552db7526c3f12aa429b430252`
   - raw SHA-256: `7cb67fa5195ca7ec0fe33122c9e5ce225edfa3297fcc82f9026e84f9c56583cb`
   - bytes: `2970`
2. `starlette/convertors.py`
   - Git blob: `72b1cf9fdfb6d7f5be61f648a9ae016deb45012b`
   - raw SHA-256: `175aec7b701a70df6bb1f2674deb839e36cde75affa2e1dcdd62f26248e45ffa`
   - bytes: `2304`

No repository-wide or tag-wide authority is created. Docs, tests, benchmarks, GitHub metadata, generated/build/minified/binary material, and evaluation-reserved objects are outside this admission.

## Rights decision

The pinned BSD-3-Clause license permits redistribution and use in source and binary forms, with or without modification. Under the project's purpose-specific source policy this is sufficient for model-training use of the exact admitted code objects. Redistribution remains conditional on retaining/reproducing the copyright notice, license conditions, and disclaimer as applicable; endorsement using the copyright-holder or contributor names is not permitted without prior written permission.

Decision:

- training: `ALLOW`
- redistribution: `ALLOW_WITH_NOTICE`
- evaluation use: `NOT_AUTHORIZED_BY_THIS_AUTHORITY`

## Family identity

Family id: `github:Kludex/starlette`, bound additionally to stable GitHub repository id `138597372` so the historical `encode/starlette` transfer alias cannot create a second family.

FastAPI's dependency on Starlette is not a lineage-collapse condition by itself. Under DATA-288 policy, collapse is required for mirrors, forks, vendored/generated derivatives, sibling files in one canonical upstream, and other shared-source lineages. Starlette is a separately maintained canonical upstream repository, not a FastAPI fork or vendored subtree. Independent-family credit is therefore allowed only if the object-level cross-source dedup gate also passes.

## Verification contract

`tools/validate_next100_045_starlette.py` independently fetches the exact immutable upstream bytes and enforces:

- raw SHA-256 and Git-blob identity for every selected source and the license;
- Python AST parse validity;
- bounded private-key/token/credential and privacy/private-endpoint scans;
- exact and normalized-exact dedup;
- Python lexical-skeleton five-token shingle Jaccard and containment checks against the terminal HTTPX and Requests code objects;
- rejection at Jaccard `>=0.85` or containment `>=0.90`;
- explicit collision checks against the known evaluation-reserved HTTPX and Requests objects;
- no training execution.

The dedicated workflow is `.github/workflows/next100-045-starlette-code-source.yml` and emits a hash-bound JSON evidence artifact.

## Terminal sealing rule

The final verdict must be issued only after a second live GitHub registry check immediately before sealing. If a concurrent FastAPI, Starlette, or evaluation-reservation authority appears, its exact selected identities must be compared before this authority can remain `ADMIT`.
