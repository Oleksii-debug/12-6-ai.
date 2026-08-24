# D05 portable S0 inference artifact handoff

## Why this exists

The live physical repository is `Oleksii-debug/12-6-ai.` with a trailing period.
The D08 Windows probe already established that `actions/checkout` cannot materialize
that repository name in the normal Windows workspace path. That is a repository
identity/platform-path blocker, not evidence that the Python package or first-party
inference runtime itself is incompatible with Windows.

The current retained-inference worker is separately building a Linux-produced bundle
containing a real trained S0 checkpoint, first-party inference evidence, CLI/server
diagnostics, and a project wheel. This package does not duplicate that producer.
Instead, it defines the fail-closed consumer-side integrity contract required before
that bundle can be moved into an artifact-only Windows job.

## Live parent and collision boundary

This branch starts from exact-green PR #89 source
`c631c024e641dac102036fafee6d78ba31c067cd`.

It deliberately does not edit:

- D01 model architecture or ModelSpec/InitSpec;
- D02 Trainer or repeatability evidence;
- D03 data;
- D04 tokenizer/packing/evaluation;
- D05 checkpoint format, transactional loader, or retained-artifact producer;
- D07 generation, sampling, CLI, parity, or HTTP server;
- D08 dependency locks;
- D10 promotion/release authority.

The portable contract consumes the retained artifact schema
`12-6.s0-retained-inference-artifact.v1` produced by the parallel D05 worker.

## Fail-closed validation

`python tools/validate_s0_portable_artifact.py ARTIFACT_ROOT` validates the bundle
before any downstream platform run.

The validator requires and recomputes:

1. exact source Git SHA and artifact-manifest self-hash;
2. unique normalized POSIX-relative paths only;
3. no absolute path, `..`, backslash, symlink, non-regular, unmanifested, or missing
   file;
4. exact byte count and SHA-256 for every retained file;
5. exactly one project wheel;
6. a strict D05 checkpoint that still verifies under checkpoint-v1 and binds the
   same source SHA;
7. the retained first-party inference report self-hash and checkpoint ID;
8. random-init / pretraining-only / no-foreign-weight / no-behavioral-weight truth
   boundaries;
9. zero-tolerance direct-vs-reloaded parity already recorded by the producer;
10. prompt and stdin CLI diagnostics bound to the same checkpoint/source;
11. the loopback `/v1/completions` response as a raw text-completion surface;
12. exact D08 Linux locked-environment evidence and its self-hash.

The output is a second self-hashed
`12-6.s0-portable-inference-validation.v1` report. This is a consumer contract, not a
release attestation.

## Windows execution boundary

A safe future Windows job should not invoke `actions/checkout` at all. It should:

1. download the already verified portable bundle into `RUNNER_TEMP`;
2. verify the bundle before installing or loading anything;
3. create a runtime from an exact D08-owned `windows-x86_64` hash lock;
4. install the retained project wheel without dependency resolution drift;
5. load the retained checkpoint through the canonical first-party backend;
6. execute plain prompt and stdin/JSON CLI paths;
7. record console output with no ANSI/TUI dependency;
8. only then record a Windows runtime result.

That job is intentionally **not enabled here** because the live D08 lock inventory
contains only `linux-x86_64` and `linux-aarch64`. Installing floating Windows
dependencies merely to make a green job would weaken the reproducibility boundary.
The machine report therefore emits:

- `repository_checkout_required=false`;
- `artifact_only=true`;
- required profile `windows-x86_64`;
- `runtime_status=BLOCKED_BY_MISSING_HASH_LOCKED_WINDOWS_RUNTIME`;
- `nvda_status=NOT_TESTED`.

D08 owns the missing Windows lock/profile. Once that exact profile exists, D05/D07 can
add the artifact-only Windows execution job without changing Base semantics or
touching the trailing-dot checkout path.

## Broader live blockers

The S0 Product path itself is currently strong: PR #88 is exact-head green across CI,
real 40-step training and strict D04 evaluation, and PR #89 adds exact same-seed
repeatability/seed-causality evidence with all workflows green.

Promotion remains blocked outside this package:

- AUDIT-A #13 and AUDIT-B #14 still have `CHANGES_REQUIRED` as the latest actual
  verdict; #88 received retest handoffs, not new verdicts;
- `main` remains bootstrap-only and unprotected;
- D10 governance/release work is separate;
- vulnerability/license adjudication in the supply-chain lane is still unresolved;
- no materially paid compute is authorized.

This package cannot issue an audit verdict, candidate promotion, or STABLE status.
