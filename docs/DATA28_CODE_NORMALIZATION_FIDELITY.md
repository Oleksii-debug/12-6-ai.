# DATA-28 Code Normalization Fidelity

## Scope

DATA-28 prevents code corpus records from entering the natural-text normalization path before tokenization. The implementation is based on DATA-10 head `077205ef2b1662a5029bc77b8fc762078cabeb17` and composes with the existing D03 dataset builder rather than creating a parallel corpus registry.

## Audit result

The incumbent D03 `normalize_text` function applies Unicode NFKC, converts CRLF/CR to LF, collapses intra-line whitespace, removes empty lines, and strips outer whitespace. Those transformations are suitable only for the incumbent natural-text policy.

DATA-10 introduced `_normalize_code_layout` to avoid whitespace collapse, but it still applied NFKC, converted CRLF/CR to LF, and stripped outer newlines. That means source code could retain indentation while still changing string literals, comments, Unicode identifiers, line endings, compatibility characters, and exact source identity.

The regression probe uses compatibility characters `①`, `ﬁ`, and `K` plus CRLF and tabs. The old DATA-10 path changes that source; DATA-28 must return it byte-for-byte unchanged.

## Code normalization contract

Schema: `12-6.code-normalization-evidence.v1`

Policy: `STRICT_UTF8_IDENTITY_PRESERVE_V1`

For accepted code, the only permitted normalization is the identity transform. DATA-28 does not:

- apply NFC, NFKC, NFD, or NFKD;
- change CRLF, LF, or CR line endings;
- convert tabs to spaces or spaces to tabs;
- trim leading or trailing whitespace/newlines;
- collapse blank lines or intra-line whitespace;
- rewrite string literals, comments, identifiers, escapes, or line continuations;
- reformat, transpile, lint-fix, or otherwise perform semantic code rewriting.

Accepted records require `source_sha256 == normalized_sha256`, zero recorded transformations, and the reason `identity_utf8_source_preserved`.

## Fail-closed artifact rejection

The code gate rejects rather than repairs:

- invalid UTF-8 byte sequences;
- U+FFFD replacement characters and surrogate code points;
- NUL and disallowed raw control characters;
- files with explicit generated-material path/header signals;
- files with explicit minified suffixes and an additional conservative extreme-minification heuristic for JavaScript/TypeScript.

Generated/minified detection is intentionally bounded. It is not a universal generated-code classifier and must not be described as one.

D03 retains its incumbent email/phone privacy hook for code, but natural-text length/alpha-ratio/whitespace quality heuristics are not applied to code.

## D03 integration

`build_dataset` now resolves each record as `natural` or `code`.

Natural records retain the incumbent D03 behavior. This is deliberately conditional so the committed natural-only S0 package remains byte-deterministic.

Code records are admitted through the identity-preserving code gate and receive:

- `modality=code`;
- `code_language` and optional source `path`;
- raw source SHA-256;
- normalized SHA-256;
- normalization schema/policy/reasons;
- UTF-8 byte counts;
- newline, tab, line-continuation, and longest-line structural counters.

The D03 manifest document assignment binds raw and normalized hashes and the normalization policy/reasons. Code datasets additionally contain normalization/rejection reason counters and the identity policy in dataset identity. No parallel source registry is introduced.

## DATA-10 integration

`strict_normalize_utf8(..., preserve_layout=True)` now delegates to the code identity gate instead of NFKC/newline normalization. `admit_for_pretraining` stores raw and normalized hashes plus normalization policy/reasons and fails if a code record's raw and normalized SHA-256 differ.

The multilingual manifest includes per-record normalization provenance.

## Executable fidelity matrix

`tests/test_code_normalization.py` covers:

- Python;
- JavaScript;
- TypeScript;
- C;
- C++;
- JSON;
- YAML;
- shell;
- Markdown fenced code blocks.

The fixtures exercise indentation, blank/newline structure, CRLF and LF, tabs, compatibility-character string literals, Unicode identifiers, comments, and backslash line continuations. Exact text and UTF-8 byte equality are primary invariants.

Parser evidence is used where cheaply available:

- Python: `ast.parse`, required;
- JSON: `json.loads`, required;
- JavaScript: `node --check`, conditional on installed Node;
- shell: `bash -n`, conditional on installed bash;
- C: `cc`/`gcc`/`clang -fsyntax-only`, conditional on an installed compiler;
- C++: `c++`/`g++`/`clang++ -fsyntax-only`, conditional on an installed compiler;
- YAML: PyYAML, conditional on installation;
- TypeScript: `tsc --noEmit`, conditional on installation.

Unavailable parsers are reported/skipped; success is never invented.

## Real licensed sample evidence

`configs/data/code_fidelity_real_samples.v1.json` pins real public files by repository, exact commit, path, and Git blob SHA-1. The mechanical runner verifies each Git blob identity before normalization and records SHA-256 before/after.

Current evidence inputs include:

- `pallets/click` Python and YAML samples, with BSD-3-Clause repository license metadata;
- `microsoft/TypeScript` TypeScript and JSON samples, with Apache-2.0 repository/file evidence;
- `libexpat/libexpat` C sample with an MIT SPDX file header and MIT repository metadata.

These files are fidelity evidence only. DATA-28 does not infer project model-training eligibility from repository license metadata, does not add these files to the pretraining corpus, and does not change the D03 rights resolver boundary.

## Machine report

`tools/run_code_fidelity_suite.py` downloads only the pinned fidelity samples, verifies the expected Git blob hash, applies strict UTF-8 identity normalization, runs available parsers before/after, and writes `data28-code-fidelity-evidence.json`.

The report binds:

- exact 12-6 source commit;
- sample registry SHA-256;
- per-sample repository/commit/path/blob identity;
- raw and normalized SHA-256;
- raw and normalized byte counts;
- structural counters;
- parser availability/success;
- reason counters;
- the old-NFKC regression probe;
- a self-hash of the canonical report payload.

## Truth boundary

DATA-28 proves mechanical source-fidelity properties for the covered code path and samples. It does not prove semantic equivalence for arbitrary programs, universal generated/minified detection, universal parser coverage, or external-source model-training eligibility.

The required merge condition is exact-head LOCAL_FREE CI: compile/lint, DATA-28 fidelity tests, incumbent D03 S0 byte-determinism tests, and the pinned real-sample evidence runner.
