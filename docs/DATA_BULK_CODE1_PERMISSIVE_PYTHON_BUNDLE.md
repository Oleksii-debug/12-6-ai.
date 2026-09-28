# DATA-BULK-CODE-1 — permissive Python implementation bundle

This is the current-main successor of issue #635. It restores the existing six-family
bulk-code acquisition contract without merging the obsolete 2026-08-26 branch ancestry.
The immutable source vector and contract identity are preserved; only execution and
fail-closed security semantics are hardened.

## Why this exists

The learned-20M critical path remains data-limited. Current source-registry planning
still has a large code shortfall before downstream global dedup, decontamination,
split/packing and unique-loss accounting. This lane measures real source bytes from a
bounded permissive multi-repository bundle instead of creating one small source PR per
file or treating repository-size estimates as capacity.

No measured byte receives automatic Research Corpus V1 credit from this package.

## Exact source vector

| Family | Exact commit | Implementation root | License authority |
| --- | --- | --- | --- |
| `github:pallets/flask` | `d318b683471101618febed18996405ad26462110` | `src/flask` | BSD-3-Clause, `LICENSE.txt` blob `9d227a0cc43c3268d15722b763bd94ad298645a1` |
| `github:pallets/click` | `68e7ea7228ca144c52e4d1d282cc09da59f7771f` | `src/click` | BSD-3-Clause, `LICENSE.txt` blob `d12a849186982399c537c5b9a8fd77bf2edd5eab` |
| `github:pallets/jinja` | `5ef70112a1ff19c05324ff889dd30405b1002044` | `src/jinja2` | BSD-3-Clause, `LICENSE.txt` blob `c37cae49ec77ad6ebb25568c1605f1fee5313cfb` |
| `github:pallets/werkzeug` | `0005c79e09bae5f4cc2bd8ccd468d7dafe24a455` | `src/werkzeug` | BSD-3-Clause, `LICENSE.txt` blob `c37cae49ec77ad6ebb25568c1605f1fee5313cfb` |
| `github:agronholm/anyio` | `ae250440c90020b030ba4e83cccc37e9a84512c5` | `src/anyio` | MIT, `LICENSE` blob `104eebf5a3002fccdaceef3a4cb936173c1c2035` |
| `github:pytest-dev/pytest` | `28549a5f6b82bc916bb2ec5cb9fbfffe9b79fc66` | `src/_pytest` | MIT, `LICENSE` blob `c3f1657fce94589bd1ec7cead810639047f3d359` |

Frozen contract identity:
`7fd2228208f928859ebe68e947a72c977cda6952035a654d12923ce3a19a7dd6`.

## Admission surface

The materializer creates a fresh Git repository for each source, fetches the exact bound
commit, proves `HEAD`, and verifies the bound license object by Git blob identity before
any implementation file is eligible.

Only `.py` implementation files under the exact source root are examined. Test,
documentation, example, build, generated and vendored directory components are excluded.
Symlinks, empty files, files above 512 KiB, non-UTF-8 files and AST-invalid Python receive
zero byte credit. A bounded generated-file header detector additionally excludes common
`auto-generated` / `do not edit` files that live directly inside an otherwise eligible
source root.

Every admitted object receives exact repository, path, commit, SHA-256 and UTF-8 byte
count. Family and aggregate byte totals are derived only from that ledger.

## Security correction

Issue #635 requires the credential/secret scan to fail closed. The historical
materializer represented a high-confidence credential match as an ordinary exclusion,
which could still yield a successful report. The current-main successor fixes this in
the core materializer: any configured private-key/token/access-key hit terminates the
whole materialization immediately.

The report validator independently rejects any legacy report containing a
`credential_pattern` exclusion and requires explicit
`PASS_NO_HIGH_CONFIDENCE_HITS` security evidence. A separate firewall remains as
defense in depth.

This is a bounded pattern scan, not a universal proof that arbitrary source code can
contain no sensitive value.

## Two-clean-materialization requirement

The restored acquisition workflow performs the complete external acquisition twice from
two empty roots. Both reports must be byte-identical, then each is checked by the
security firewall and the canonical report validator. The retained artifact is the
exact report from that execution head.

This dedicated network workflow is retained because external immutable source
materialization is itself the scientific evidence required by #635; generic repository
CI cannot substitute for fetching and hashing the six upstream source trees.

## Downstream boundary

A successful artifact is source-intake evidence only. Before any bytes can become
training authority, the incumbent D03/D04 path must consume them through:

1. global cross-source exact/near/lineage dedup;
2. reserved-evaluation decontamination;
3. post-composition quality/privacy checks;
4. balance and family caps;
5. cluster-safe split and deterministic packing;
6. two clean corpus builds;
7. exact post-pack unique nonignored causal-loss ledger.

Source bytes are not tokenizer tokens and are not optimized causal-loss positions.
Authorized training exposure remains zero in this lane.

## Scale boundary

The bundle measures six independent canonical repository families. Historical planning
asked for at least eight families in this pool, so this package by itself does not claim
that code acquisition is finished even if its byte yield is large. Any later family
addition must be separately pinned and must survive the same materialization and global
dedup rules.

## Prohibited claims

No corpus or shard identity, tokenizer fit, optimizer update, learned model checkpoint,
final-test access, paid compute, learned-20M promotion, or 200M promotion is authorized
by DATA-BULK-CODE-1.
