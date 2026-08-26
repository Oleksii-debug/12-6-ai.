# NEXT100-047 — bounded Jinja code source admission

Worker: `NEXT100-047-CODE-JINJA`

Execution: `LOCAL_FREE` only.

## Candidate authority

Upstream repository: `pallets/jinja`

Exact revision: `5ef70112a1ff19c05324ff889dd30405b1002044`

Source family: `github:pallets/jinja`

License: `BSD-3-Clause`, exact `LICENSE.txt` Git blob `c37cae49ec77ad6ebb25568c1605f1fee5313cfb`, 1,475 bytes.

Project purpose decision: training use is `ALLOWED` for this exact bounded snapshot under the reviewed BSD-3-Clause grant. Redistribution is `ALLOWED_WITH_LICENSE_CONDITIONS`: preserve the required notice/conditions/disclaimer and non-endorsement boundary. Public availability alone is not the rights basis.

## Bounded implementation inventory

| path | Git blob SHA-1 | bytes |
| --- | --- | ---: |
| `src/jinja2/lexer.py` | `e35cd471e98f516221759ef4345867d71d28230a` | 29,687 |
| `src/jinja2/parser.py` | `3ae857ebe2a937c1362c44f38e5c436bfd3c84b0` | 40,095 |
| `src/jinja2/compiler.py` | `84cd513028021b327715400bfa89d28f4e3120cd` | 73,918 |
| `src/jinja2/environment.py` | `acaaffb5946c7a4f8973db47a317a286d496ada1` | 60,847 |
| `src/jinja2/runtime.py` | `667c0416dd6a524dbee6853df7efec67a463d667` | 34,148 |

Bounded code capacity if all hard gates pass: **238,695 bytes**.

This is one independent source family, not five families.

## Excluded from code capacity

Documentation, tests, examples, scripts, packaging/build metadata, the license itself, generated `_identifier.py`, `py.typed`, lock files, and any vendored/generated/minified/build derivative are excluded. These bytes may not be used to inflate code capacity.

## Hard qualification gates

The dedicated exact-head validator fetches only the pinned public GitHub objects and requires:

- canonical upstream repository identity; fork and mirror status must be false/null;
- exact byte size and Git blob SHA-1 for the license and every source object;
- strict UTF-8 identity-preserving normalization (`raw == normalized`);
- Python AST parse plus `compile()` success for every selected file;
- private-key/token/credential-pattern scan with no hit;
- privacy-like email/private-network endpoint scan with no unsafe hit;
- generated/vendored/build/minified exclusions;
- substantive implementation thresholds for size, code lines, AST nodes, and function/class definitions;
- no exact duplicate among the five selected objects;
- no `>=0.85` lowercased five-token-shingle Jaccard near duplicate within the Jinja selection or against the incumbent terminal DATA-227 `encode/httpx` and `psf/requests` objects;
- no selected exact Jinja path/blob identity reserved for evaluation at the mandatory final live-registry recheck.

The runner emits raw SHA-256 and identity-normalization SHA-256 for every object, a self-hashed machine authority report, raw source snapshots, and the exact license evidence as a retained workflow artifact.

## Family identity

`pallets/jinja` is treated as one independent upstream code family. Relationship to other Python web libraries does not collapse family identity by dependency alone; duplicate/fork/mirror evidence would be required for collapse. The current incumbent code families are `github:encode/httpx` and `github:psf/requests`.

## Evaluation boundary

Initial live search found no Jinja evaluation reservation. This admission grants training-purpose authority only. It does not create evaluation-use authority. A later evaluation authority that reserves any exact selected path/blob before final sealing invalidates that object for training and requires capacity recomputation.

## Claim boundary

This authority covers only the five pinned implementation files above. It does not admit the whole Jinja repository, does not count documentation/tests/generated material as code, does not execute model training, and does not make a production-corpus or representativeness claim.

Terminal `ADMIT` is valid only after the dedicated exact-head workflow is terminal-success and the required final live-registry/evaluation-reservation concurrency check remains clean. Otherwise the terminal status is `RETEST` with the exact failing gate.
