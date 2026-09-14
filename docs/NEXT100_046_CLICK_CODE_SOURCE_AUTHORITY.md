# NEXT100-046 Click Code Source Authority

SWARM_WORKER_ID: `NEXT100-046-CODE-CLICK`

Execution profile: `LOCAL_FREE`

Authority type: external real code source admission for bounded base-model training.

## Terminal ruling

`ADMIT` the exact three-file Click-native implementation snapshot defined below, provided the dedicated exact-head verifier succeeds and the mandatory final live code/evaluation-registry recheck finds no new collision or reservation. This authority grants no evaluation use and does not admit the whole repository.

Any change to upstream repository identity, release/tag resolution, commit, path, Git blob, license, rights decision, lineage, evaluation reservation, or live-registry collision invalidates this authority and requires a new admission.

## Immutable upstream boundary

- Canonical upstream: `https://github.com/pallets/click`
- GitHub repository id: `19103692`
- Source family: `github:pallets/click`
- Release: `8.4.2`
- Annotated tag object: `c6b2d71ee056a96b8e6e06e6c29f67c1a766f8e4`
- Exact commit: `b2e30a175449cfda909ee4fbf4a29a6a071cad53`
- `src/click` tree: `abb3201d1d4b3e1c3cf2ff958d1b95e8ea650fe8`
- Selected-inventory identity: `sha256:37e0abc2386f1e762ddba23417cd6e6df6c707c12ae63f625f48a5d8b4288a3e`

The annotated release tag is unsigned. That is recorded as a provenance fact, not treated as a rights grant or a blocker; byte identity is instead bound through the exact commit/tree/blob chain.

## License and permitted purposes

Exact license object:

- path: `LICENSE.txt`
- Git blob: `d12a849186982399c537c5b9a8fd77bf2edd5eab`
- bytes: `1,475`
- SPDX: `BSD-3-Clause`

The exact reviewed BSD-3-Clause grant permits redistribution and use in source and binary forms, with or without modification, subject to retention of the copyright notice, conditions and disclaimer and the non-endorsement condition. Under `policy://12-6/data/explicit-model-training-evidence-v1`, this is sufficient authority for acquisition, analysis, storage and model-training use of the exact bounded objects. Public accessibility or SPDX labeling alone is not the basis.

Training decision: `ALLOWED`.

Redistribution decision: `ALLOWED_WITH_LICENSE_CONDITIONS`.

Redistributed source or transformed bundles must preserve the applicable BSD notice, conditions and disclaimer. Pallets or contributor names may not be used for endorsement without permission. This authority does not claim that verbatim source reproduced by a model is free of the underlying license obligations.

## Admitted code capacity

Only these exact first-party implementation objects count:

| Path | Git blob SHA-1 | Bytes | Role |
| --- | --- | ---: | --- |
| `src/click/decorators.py` | `db6a45ebbaedfdcde339397bbfe936c9440de180` | 19,709 | command, option, argument and context decorator implementation |
| `src/click/formatting.py` | `c4aa2de571a1eb17bfeb1853e315d28bc968e74c` | 10,444 | terminal help and formatter implementation |
| `src/click/shell_completion.py` | `468ee7720d934396d0a309067d800ea819af7da2` | 22,618 | shell completion protocol and renderer implementation |

Total admitted candidate capacity: `52,771` raw bytes.

Independent source-family credit: `1`.

## Explicit zero-capacity exclusions

The following do not count as admitted Click code capacity:

- `docs/**`, documentation renderings and generated documentation;
- `tests/**`, examples and benchmark/evaluation material;
- release archives, wheels, sdists, build/dist outputs and packaging metadata;
- generated, minified, binary, vendored or third-party copies;
- `LICENSE.txt` and `src/click/py.typed`;
- every Click path not explicitly present in the three-object allowlist.

The release wheel and sdist hashes are retained only as provenance metadata and contribute zero capacity.

## Mixed-lineage exclusions

Two files were specifically rejected from Click-native family capacity after source review:

- `src/click/parser.py`, blob `4fcbf7caa83a474cee2d3ea25da56b0399dd893a`: its file notice states that the module started as largely a copy-paste from Python stdlib `optparse` and incorporates Gregory P. Ward / Python Software Foundation material.
- `src/click/_textwrap.py`, blob `82840f2dff3ce627712c0ece2752382a0f7dab8b`: the implementation explicitly mirrors the CPython `textwrap.TextWrapper` algorithm.

They are conservatively excluded rather than counted as new Click-native capacity.

## Parse, secrets, privacy and quality gates

The dedicated verifier `tools/validate_next100_046_click_code_source.py` is stdlib-only and network-bounded. It processes immutable upstream bytes in memory and does not vendor them into this repository.

For every selected object it requires:

- exact byte size and Git blob reconstruction;
- strict UTF-8 identity-preserving normalization;
- Python `ast.parse` and `compile` success;
- no private-key, GitHub/OpenAI/AWS/Slack/npm token or JWT-like secret pattern;
- no non-example email-like address or private-network endpoint under the configured privacy heuristic;
- no generated-code header marker;
- substantive implementation thresholds over raw bytes, code lines, AST nodes and function/class definitions.

The exact upstream release commit also has successful upstream `pre-commit` and `Tests` workflow runs (`28079506908` and `28079506934` respectively), providing independent corroborating execution evidence for the pinned commit.

## Dedup and family identity

Exact dedup rejects repeated Git blobs or raw SHA-256 identities. Near-dedup uses lower-cased lexical 5-token shingles and rejects Jaccard `>= 0.85` or containment `>= 0.90`.

The verifier compares the three Click objects internally and against 23 pinned code objects covering the terminal DATA-227 HTTPX/Requests baseline plus current CPython, Django, Starlette, Jinja and Flask admission candidates available at qualification time.

Family identity is `github:pallets/click`. Shared Pallets organization membership with Flask/Jinja and Flask's dependency on Click do not by themselves collapse repository families. Collapse requires fork/mirror/vendor/shared-source lineage or duplicate/copy evidence. The mixed-stdlib files identified above are excluded rather than used to inflate Click family capacity.

## Evaluation firewall

Candidate role: `TRAINING_ONLY`.

Evaluation use: `NOT_AUTHORIZED`.

No Click object in this authority may be used as evaluation data without a separate evaluation reservation authority. If any selected Click blob becomes evaluation-reserved before sealing or ingestion, the affected object must be excluded and this authority reissued.

## Concurrency seal

Immediately before the final verdict, re-read the live code-source and evaluation registries and inspect newly published sibling authorities. Open branches are candidate evidence, not automatic terminal registry entries. If a new exact blob/path collision, evaluation reservation, fork/mirror lineage, or near-duplicate conflict is found, fail closed to `RETEST` or `REJECT` rather than preserving `ADMIT`.

No model training is executed by this worker. No paid compute is authorized.
