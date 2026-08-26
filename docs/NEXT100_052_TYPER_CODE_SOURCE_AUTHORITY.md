# NEXT100-052 — Typer external-real code source authority

Worker: `NEXT100-052-CODE-TYPER`

## Candidate decision

`ADMIT`, bounded to one exact implementation object and subject only to the dedicated exact-head verifier remaining green plus the mandatory final live-registry refresh showing no newly sealed duplicate or evaluation reservation.

This authority is training-only and `LOCAL_FREE`. It does not execute training and it does not grant evaluation use.

## Immutable upstream boundary

- canonical repository: `fastapi/typer`
- stable GitHub repository id: `229937405`
- release: `0.27.1`, published 2026-08-03
- exact commit: `fe2aa0e2f9c853de378e60ca24ec3b256144decf`
- release commit is GitHub-verified
- license: `MIT`
- license path: `LICENSE`
- license Git blob: `a7694736cf37716aafec14b24aa8d6316ebe07a3`
- license raw SHA-256: `58992cebcf8dfb6e40c4e2112ed12126c243666dca3912a3d78b7ecac4859d49`

Selected implementation object only:

1. `typer/utils.py`
   - Git blob: `addf9334d4210a9cddc9e5608ae446417d372eb0`
   - raw SHA-256: `c272750d65c114c9f12c768c37f1c3627b712d16f2d0afabbb1eb76a89072272`
   - bytes: `7599`

No repository-wide or release-wide authority is created.

## Rights decision

The pinned MIT license grants unrestricted dealing in the Software, including use, copying, modification, publication, distribution, sublicensing, and sale, subject to including the copyright and permission notice in all copies or substantial portions. Under the project's purpose-specific source policy this grant is sufficient for model-training use of the exact admitted object. Redistribution remains conditioned on preservation of the required MIT notice.

Decision:

- training: `ALLOW`
- redistribution: `ALLOW_WITH_NOTICE`
- evaluation use: `NOT_AUTHORIZED_BY_THIS_AUTHORITY`

## Lineage and family decision

Family id: `github:fastapi/typer`, stable repository id `229937405`.

The family is independent only after object-level exclusions and dedup; repository ownership alone is not used to manufacture independence.

- Click: Typer is deliberately Click-lineage software. Typer 0.27.1 contains `typer/_click/**` plus a Click license file. That entire subtree is excluded from Typer capacity and is treated as Click-derived/vendored lineage. The selected `typer/utils.py` object remains eligible only if exact whole-tree and near-code comparison against pinned Click 8.4.2 stays below the project rejection thresholds.
- FastAPI: Typer and FastAPI share the `fastapi` GitHub organization and Sebastián Ramírez as primary maintainer, but they are separate non-fork repositories. Shared governance is not source equivalence. The selected object remains independently creditable only if exact and near-code comparison against pinned FastAPI 0.141.1 stays clear.
- Pydantic: upstream `typer/_typing.py` explicitly states that it was copied and reduced from Pydantic 1.9.2. It is excluded from this authority and cannot inflate Typer-family capacity.

This authority therefore admits no Click-vendored object and no explicitly copied Pydantic object.

## Local deterministic pre-check

Before materialization, the exact `typer/utils.py` bytes fetched through the connected GitHub source were independently checked locally:

- byte count reproduced: `7599`
- Git blob SHA-1 reproduced: `addf9334d4210a9cddc9e5608ae446417d372eb0`
- raw SHA-256 reproduced: `c272750d65c114c9f12c768c37f1c3627b712d16f2d0afabbb1eb76a89072272`
- Python `ast.parse` validity: `PASS`
- private-key/AWS/GitHub/Slack/OpenAI/credential-URL/literal-credential patterns: no hits
- email/IPv4/private-endpoint privacy patterns: no hits

The same identity and scan are re-executed from immutable upstream bytes by CI rather than trusting this prose record.

## Verification contract

`tools/validate_next100_052_typer.py` fetches immutable upstream bytes and enforces:

- byte count, raw SHA-256, and Git-blob identity for the selected source and license;
- Python AST parse validity;
- bounded private-key/token/credential and privacy/private-endpoint scans;
- exact and normalized-exact dedup against terminal HTTPX and Requests code objects;
- five-token Python lexical-skeleton Jaccard and containment checks, rejecting at Jaccard `>=0.85` or containment `>=0.90`;
- exact selected-blob search across the complete pinned Click and FastAPI trees;
- near comparison against analogous pinned Click and FastAPI implementation paths;
- explicit known evaluation-reservation collision checks;
- no training execution.

The dedicated workflow is `.github/workflows/next100-052-typer-code-source.yml` and uploads hash-bound JSON evidence.

## Explicit exclusions

Excluded from this authority: `typer/_click/**`, `typer/_typing.py`, bootstrap/version metadata, every other unselected `typer/**` file, docs, docs sources, tests, scripts, GitHub metadata, generated/build/minified/binary material, and all evaluation-reserved objects.

## Terminal sealing rule

The final verdict is sealed only after the dedicated verifier is green and a second live GitHub registry check is performed immediately before sealing. If concurrent Click, FastAPI, Typer, or evaluation-reservation authority appears, its selected object identities must be compared before this authority can remain `ADMIT`.
