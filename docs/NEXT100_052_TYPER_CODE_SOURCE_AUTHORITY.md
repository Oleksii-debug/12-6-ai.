# NEXT100-052 — Typer external-real code source authority

Worker: `NEXT100-052-CODE-TYPER`

Execution profile: `LOCAL_FREE`

## Terminal decision

`RETEST`

This qualification attempt does **not** admit Typer into training capacity. Current credit is `0 files / 0 bytes / 0 independent families`.

The source-rights review, immutable identity checks, local parse/privacy/secret pre-checks, and late Click/FastAPI lineage exclusions are favorable. The blocking gate is executable: the dedicated exact-head verifier has not run to completion under the repository-wide GitHub Actions backlog. The latest source-binding head before this terminal prose seal is `adc41698489f7a2b35a0aef62d41337db4c08dd2`; dedicated workflow run `32999335822` is `QUEUED`, not success evidence.

A successor may change this exact candidate to `ADMIT` only after the dedicated verifier executes successfully against the then-current exact authority head and the live source/evaluation registries are refreshed again. This terminal attempt itself remains `RETEST`.

No model training, optimizer update, paid compute, or evaluation use was performed.

## Immutable upstream boundary

- canonical repository: `fastapi/typer`
- stable GitHub repository id: `229937405`
- release: `0.27.1`, published 2026-08-03
- exact commit: `fe2aa0e2f9c853de378e60ca24ec3b256144decf`
- release commit: GitHub-verified
- license: `MIT`
- license path: `LICENSE`
- license Git blob: `a7694736cf37716aafec14b24aa8d6316ebe07a3`
- license raw SHA-256: `58992cebcf8dfb6e40c4e2112ed12126c243666dca3912a3d78b7ecac4859d49`
- license bytes: `1086`

Bounded candidate inventory, not presently admitted:

1. `typer/utils.py`
   - Git blob SHA-1: `addf9334d4210a9cddc9e5608ae446417d372eb0`
   - raw SHA-256: `c272750d65c114c9f12c768c37f1c3627b712d16f2d0afabbb1eb76a89072272`
   - bytes: `7599`
   - language: Python

No other Typer object is authorized by this attempt.

## Rights decision

The exact MIT license grants use, copying, modification, publication, distribution, sublicensing, and sale, subject to inclusion of the copyright and permission notice in copies or substantial portions. Under the project's explicit purpose-specific source policy this is sufficient legal authority for model-training use of the exact bounded candidate **if** the technical admission gates pass.

Purpose decisions for the exact candidate:

- acquisition/storage/analysis: `ALLOW`
- model training: `ALLOW_IF_TECHNICAL_GATES_PASS`
- redistribution: `ALLOW_WITH_NOTICE`
- evaluation: `NOT_AUTHORIZED_BY_THIS_AUTHORITY`

The terminal `RETEST` is not a rights rejection; it is a missing executable-proof result.

## Source-family and lineage decision

Candidate family: `github:fastapi/typer`, stable repository id `229937405`.

The intended family decision is `INDEPENDENT_WITH_LINEAGE_EXCLUSIONS`, not automatic independence from repository naming.

### Click boundary

Typer 0.27.1 contains `typer/_click/**` and a Click license file. That entire subtree is treated as Click-derived/vendored lineage and contributes zero Typer-family capacity.

The live NEXT100-046 Click authority was refreshed at branch head `5d7bd3d869e38f5f13ce8a82ad6fa19e3865a6df`. Its exact admitted candidate objects are:

- `src/click/decorators.py` — blob `db6a45ebbaedfdcde339397bbfe936c9440de180`
- `src/click/formatting.py` — blob `c4aa2de571a1eb17bfeb1853e315d28bc968e74c`
- `src/click/shell_completion.py` — blob `468ee7720d934396d0a309067d800ea819af7da2`

The Typer candidate blob `addf9334d4210a9cddc9e5608ae446417d372eb0` is not an exact identity match to those objects. The updated verifier also binds Click 8.4.2 commit `b2e30a175449cfda909ee4fbf4a29a6a071cad53`, performs a complete-tree exact-blob scan, and near-compares the Typer object with Click core/decorators/formatting/shell-completion/types/utils. Those expanded executable comparisons have not completed, so no family credit is granted yet.

### FastAPI boundary

Typer and FastAPI share the `fastapi` GitHub organization and primary maintainer, but are separate non-fork repositories. Shared governance does not prove source equivalence and does not itself collapse families.

The live NEXT100-044 FastAPI authority was refreshed at branch head `fbbdd3a5fc5207c39708e133400ed0305767b9f9`. It pins upstream FastAPI commit `49033471594ea5d99a80abdf1043231b7791ee49` and selects:

- `fastapi/sse.py` — blob `c31334835032570d8244526a623ac249ffc77284`
- `fastapi/exceptions.py` — blob `d7065c52fe20220e12b7d20db4da7cbeadaf171a`
- `fastapi/datastructures.py` — blob `1da784cf0927ed55ec6abeb051d89a6ce1e90630`

The Typer candidate blob is not an exact identity match to those objects. The updated verifier binds those live FastAPI objects and also performs complete-tree exact and lexical near comparisons. That executable comparison remains uncompleted at terminalization.

### Pydantic copied-source boundary

`typer/_typing.py` explicitly states that it was copied and reduced from Pydantic 1.9.2. It is excluded from this authority and contributes zero Typer-family capacity.

## Local deterministic pre-check

The exact `typer/utils.py` bytes acquired through the connected GitHub source reproduced:

- bytes: `7599`
- Git blob SHA-1: `addf9334d4210a9cddc9e5608ae446417d372eb0`
- raw SHA-256: `c272750d65c114c9f12c768c37f1c3627b712d16f2d0afabbb1eb76a89072272`
- Python `ast.parse`: `PASS`
- private-key/AWS/GitHub/Slack/OpenAI/credential-URL/literal-credential patterns: no hits
- email/IPv4/private-endpoint privacy patterns: no hits

These are useful pre-checks but do not replace the dedicated immutable upstream verifier for terminal admission.

## Dedup contract

`tools/validate_next100_052_typer.py` is stdlib-only and fail-closes on:

- exact source/license byte count, raw SHA-256, and Git-blob reconstruction;
- Python AST parse validity;
- bounded secret, credential, email, private-IP, and private-endpoint scans;
- raw exact and normalized exact duplicate matches;
- five-token Python lexical-skeleton Jaccard `>=0.85`;
- five-token lexical-skeleton containment `>=0.90`;
- full pinned Click/FastAPI tree exact-blob identity matches;
- near-code collisions with live-authority-selected and analogous Click/FastAPI files;
- evaluation-reservation collision;
- any training execution.

The manifest now binds terminal DATA-287 HTTPX/Requests objects plus the current NEXT100-046 Click and NEXT100-044 FastAPI selected objects. Because the updated verifier is still queued, its dedup result is not represented as terminal PASS.

## Evaluation firewall

Candidate role: `TRAINING_ONLY` if later admitted.

Evaluation use: `NOT_AUTHORIZED`.

The live EVAL-289 code-evaluation-rights-reservation branch was refreshed at head `1c870e5e02bf48891ca599b0b3f3bfe6e84425bc`; that authority has zero active reserved code objects. No current Typer evaluation reservation was found. Any future reservation of the exact candidate object blocks training use and requires a new authority.

## Live registry seal

Immediately before terminalization:

- DATA-287 external snapshot registry head remained `b0523ccbc4b957615aac849d476cfa851be87578`; its canonical code baseline remains HTTPX + Requests.
- NEXT100-046 Click authority head: `5d7bd3d869e38f5f13ce8a82ad6fa19e3865a6df`.
- NEXT100-044 FastAPI authority head: `fbbdd3a5fc5207c39708e133400ed0305767b9f9`.
- EVAL-289 reservation head: `1c870e5e02bf48891ca599b0b3f3bfe6e84425bc`, zero active reserved code objects.
- concurrent code-source branches exist for other independent families; open candidate branches are not silently composed into DATA-287. A later registry-convergence authority must recompute global cross-source family and near-dedup accounting before simultaneous ingestion.

## Exact blocker and successor condition

Dedicated workflow: `.github/workflows/next100-052-typer-code-source.yml`.

Latest source-binding run before this seal: `32999335822` on `adc41698489f7a2b35a0aef62d41337db4c08dd2`.

Observed terminalization state: `QUEUED`.

Therefore this attempt cannot claim parse/dedup/lineage/evaluation checks as one completed exact-head proof and cannot grant training capacity.

A successor `ADMIT` requires, on one exact head:

1. dedicated Typer verifier `SUCCESS`;
2. immutable evidence report with `verdict=ADMIT` and `failures=[]`;
3. source/license identities reproduced exactly;
4. secrets/privacy/AST gates terminal PASS;
5. Click/FastAPI exact and near-lineage gates terminal PASS;
6. immediate late re-read of DATA-287 or its successor plus all newly terminal code-source authorities;
7. evaluation-reservation recheck with no selected-object collision.

Terminal result for NEXT100-052 current attempt: **RETEST; zero capacity credit.**
