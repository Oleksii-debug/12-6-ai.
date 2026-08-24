# D05 Windows inference accessibility smoke

This package closes a specific S0 local-test evidence gap without changing model, tokenizer,
checkpoint, training, evaluation, sampling, or raw Base semantics.

The physical GitHub repository is named `Oleksii-debug/12-6-ai.`. A normal GitHub-hosted
Windows checkout creates a workspace whose repository path ends in a period, which Windows
cannot represent reliably. Earlier Windows CI therefore failed inside checkout before Python.
That is a repository-identity/platform interaction, not evidence that the inference CLI itself
fails on Windows.

## What this workflow does

`.github/workflows/d05-windows-inference-accessibility.yml` deliberately avoids checkout in the
Windows job.

1. An Ubuntu job checks out the exact PR source head, verifies `git rev-parse HEAD`, and creates a
   deterministic Windows-safe ZIP containing `src/**/*.py`, the accessibility smoke tool, and
   `pyproject.toml`.
2. The source ZIP SHA-256 and exact 40-hex source SHA are recorded in a small manifest.
3. The ZIP and manifest are retained as a workflow artifact.
4. A `windows-2025` job uses only the GitHub Actions REST API to download that artifact into
   `RUNNER_TEMP`, never into the invalid repository-named workspace.
5. The Windows job verifies the manifest source SHA and source-ZIP SHA-256 before extraction.
6. It runs `tools/run_windows_inference_accessibility_smoke.py --require-windows` from the safe
   extracted path and retains the resulting JSON report.

The smoke executes the real `python -m twelve_six.inference.cli` process repeatedly with a tiny
synthetic backend that implements the public `InferenceBackend` protocol. This isolates the
Windows/stdin/stdout/stderr/accessibility transport surface from the heavyweight canonical Torch
checkpoint runtime.

The checks cover:

- plain prompt text on stdout with diagnostics isolated to stderr;
- UTF-8 Ukrainian text supplied through stdin;
- JSON diagnostics and token IDs;
- greedy generation;
- same-seed sampled generation repeatability;
- text-stop and token-stop behavior;
- context-limit termination;
- over-context fail-closed errors with no traceback leakage;
- ANSI/control-character-free stdout and stderr suitable for simple screen-reader consumption;
- paths containing spaces and non-ASCII characters.

The machine report schema is `12-6.windows-inference-accessibility-smoke.v1` and binds the exact
source SHA plus the verified source-bundle SHA-256.

## Truth boundary

A green Windows smoke means the real D07 CLI transport/process path works on a GitHub-hosted
Windows runner when source is delivered through a Windows-safe exact-source artifact. It does not
mean all project code has become Windows-supported.

Specifically, this package does **not** claim:

- a canonical D01/D04/D05 first-party Torch checkpoint was loaded on Windows;
- the Linux hash-locked runtime is equivalent to a Windows dependency lock;
- NVDA itself was installed, attached, or exercised in an interactive session;
- the physical trailing-period repository name is fixed;
- public serving, TLS, auth, streaming, batching, KV-cache, throughput, or GPU support;
- AUDIT-A/AUDIT-B PASS, CANDIDATE, STABLE, or any promotion authority.

The canonical Base remains random-initialized and pretraining-only. No instruction, alignment,
refusal, ethics, personality, domain-specialization behavior, foreign pretrained weights, or
materially paid compute are introduced here.

## Next Windows step

If this transport smoke is green, the next separate platform task is a D08/D05-owned exact Windows
runtime lock plus canonical first-party checkpoint load/generation on Windows. That work must not
reuse Linux lock identities or infer NVDA live compatibility from this non-interactive CI smoke.
