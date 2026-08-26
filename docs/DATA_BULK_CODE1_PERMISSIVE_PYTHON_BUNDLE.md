# DATA-BULK-CODE-1 — permissive Python implementation bundle

This lane is the first executable child of DATA-BULK-ACQ-V1's `code-permissive-python-implementation-pool`. The parent plans 2.2M gross bytes across at least eight independent families, but prospective bytes receive zero capacity credit. DATA-BULK-CODE-1 therefore measures a concrete first six-family bundle without assuming how many bytes will survive.

## Exact source vector

| Family | Exact commit | Implementation root | License authority |
| --- | --- | --- | --- |
| `github:pallets/flask` | `d318b683471101618febed18996405ad26462110` | `src/flask` | BSD-3-Clause, `LICENSE.txt` blob `9d227a0cc43c3268d15722b763bd94ad298645a1` |
| `github:pallets/click` | `68e7ea7228ca144c52e4d1d282cc09da59f7771f` | `src/click` | BSD-3-Clause, `LICENSE.txt` blob `d12a849186982399c537c5b9a8fd77bf2edd5eab` |
| `github:pallets/jinja` | `5ef70112a1ff19c05324ff889dd30405b1002044` | `src/jinja2` | BSD-3-Clause, `LICENSE.txt` blob `c37cae49ec77ad6ebb25568c1605f1fee5313cfb` |
| `github:pallets/werkzeug` | `0005c79e09bae5f4cc2bd8ccd468d7dafe24a455` | `src/werkzeug` | BSD-3-Clause, `LICENSE.txt` blob `c37cae49ec77ad6ebb25568c1605f1fee5313cfb` |
| `github:agronholm/anyio` | `ae250440c90020b030ba4e83cccc37e9a84512c5` | `src/anyio` | MIT, `LICENSE` blob `104eebf5a3002fccdaceef3a4cb936173c1c2035` |
| `github:pytest-dev/pytest` | `28549a5f6b82bc916bb2ec5cb9fbfffe9b79fc66` | `src/_pytest` | MIT, `LICENSE` blob `c3f1657fce94589bd1ec7cead810639047f3d359` |

The contract identity is `7fd2228208f928859ebe68e947a72c977cda6952035a654d12923ce3a19a7dd6`.

## Admission surface

The materializer initializes a fresh Git repository per source, fetches only the exact bound commit, proves `HEAD`, and verifies the bound license file as an exact Git blob before source bytes are considered.

Only implementation `.py` files under the exact source root are examined. Test, documentation, example, build, generated and vendored directory components are excluded. Symlinks, empty files, files above 512 KiB, non-UTF-8 files, AST-invalid Python and files matching the bounded credential-pattern set receive no byte credit. Exact file hashes duplicated anywhere earlier in the bundle are also excluded from later credit rather than counted twice.

Every admitted object receives repository, path, SHA-256 and exact UTF-8 byte count. Family totals and the aggregate total are derived from that immutable per-file ledger. No planned or repository-size estimate is used as evidence.

## Two-clean-materialization requirement

The dedicated workflow performs the complete external acquisition twice from separate empty directories. The resulting canonical JSON evidence must be byte-identical. A changed upstream branch is irrelevant because every source is fetched by immutable commit SHA; a changed or unavailable bound object fails closed.

The terminal workflow artifact is still source-intake evidence, not Research Corpus V1. A successor must bind the artifact/report identity into the canonical source registry and then execute global exact/near dedup, reserved-evaluation decontamination, post-composition quality/privacy checks, cluster-safe split, deterministic packing, two clean corpus builds and the post-pack unique non-ignored causal-loss ledger.

## Scale boundary

This package starts with six independent repository families; DATA-BULK-ACQ-V1 asks for at least eight families in this pool. Six green families therefore do not complete the planned pool. At least two additional independent canonical families, or a separately justified parent-plan revision, remain necessary.

`eligible_utf8_bytes` are source bytes only. They are not tokenizer tokens and never become optimized causal-loss positions automatically. Authorized training exposure remains zero in this lane.

## Prohibited claims

No corpus identity, shard identity, tokenizer fit, model training, optimizer update, learned-20M/100M claim, final-test access or paid compute is authorized here. Evaluation remains separately firewalled.
