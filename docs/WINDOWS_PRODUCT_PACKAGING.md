# Windows terminal product packaging

## Scope

This is the D08 Windows product-install boundary for raw 12-6 Base. It intentionally does not add a GUI, chat template, custom installer, bundled checkpoint, or alternate inference implementation.

The physical GitHub repository name still ends in a period. That is a real Windows checkout hazard and is not hidden or renamed here. The product path avoids requiring a repository checkout by producing and consuming exact GitHub Actions artifacts under Windows-safe paths.

## Three replaceable layers

The Windows product is split into three independent layers:

1. **Runtime** — exact CPython 3.11.9 plus the `windows-x86_64` D08 wheelhouse/lock artifact. The artifact contains dependency wheels and lock evidence, not application or model bytes.
2. **Application** — the pure Python `twelve-six-ai` wheel plus `launcher.py` and `12-6.cmd`. It contains no dependency wheelhouse and no checkpoint bytes.
3. **Model/checkpoint** — any independently obtained compatible D05 checkpoint directory selected at invocation time with `--checkpoint`. Replacing a model does not rebuild the runtime or application.

This uses maintained mechanisms: official CPython, `venv`, pip wheels, `--require-hashes`, and the existing first-party CLI/server. No MSI, EXE installer framework, or repository clone is required.

## Exact Windows runtime

The committed Windows profile is `windows-x86_64` and requires exact CPython `3.11.9`. Its runtime lock contains 12 exact distributions, including:

- `torch==2.13.0`;
- `numpy==2.4.6`;
- `safetensors==0.8.0`.

The build toolchain includes exact `pip==26.2.1`, `setuptools==84.0.0`, `wheel==0.48.0`, and `packaging==26.3`.

Every non-comment lock line is exact `name==version` plus SHA-256. The lock is platform-specific: Windows artifact hashes are not copied from Linux.

## Artifact-only install layout

A product directory has this shape after installation:

```text
12-6 AI/
  12-6.cmd
  launcher.py
  runtime/
    Scripts/python.exe
    12-6-lock/
      profile.json
      runtime.lock.txt
      toolchain.lock.txt
      dev.lock.txt
  model checkpoints live elsewhere and are passed by path
```

The GitHub Actions acceptance job creates this layout under a path containing both spaces and Ukrainian Unicode, installs runtime dependencies from the downloaded runtime wheelhouse with `PIP_NO_INDEX=1`, installs the application wheel with `--no-deps`, and never checks out the repository.

For a manual installation, first install official CPython 3.11.9 and verify the patch version before creating the runtime environment. Then reproduce the same standard `venv` plus offline pip steps using the two product artifacts. A different Python 3.11 patch is rejected by `12-6 status`.

## Text interface

The entry point is `12-6.cmd` and has three commands:

```text
12-6 status [--checkpoint PATH] [--json]
12-6 generate --checkpoint PATH [generation options]
12-6 serve --checkpoint PATH [server options]
```

Examples:

```powershell
& '.\12-6.cmd' status --json
'Привіт з українського stdin' | & '.\12-6.cmd' generate --checkpoint 'D:\Моделі 12-6\checkpoint-001'
& '.\12-6.cmd' serve --checkpoint 'D:\Моделі 12-6\checkpoint-001' --host 127.0.0.1 --port 8000 --json-diagnostics
```

`generate` and `serve` delegate to the existing first-party modules with inherited stdin/stdout/stderr. The packaging layer does not duplicate sampling, checkpoint loading, model semantics, or HTTP completion semantics.

## Status and exit codes

`status` verifies the installed D08 profile self-hash, runtime-lock SHA-256, exact interpreter, exact installed runtime distribution versions, application distribution presence, and imports of NumPy, SafeTensors, and Torch. With `--checkpoint`, it additionally calls the canonical checkpoint verifier and reports checkpoint identity fields.

Launcher-owned exit codes are:

- `0` — ready/success;
- `2` — launcher usage error;
- `10` — runtime/application environment invalid or missing;
- `20` — selected checkpoint missing or fails verification.

`generate` and `serve` preserve the exit code of the delegated first-party command. Errors remain plain text on stderr; JSON status is available for diagnostics and automation.

## CI proof boundary

The Windows product workflow has four distinct stages:

- exact source + application wheel production on the known Linux authority path;
- real `windows-2025` execution of the committed `windows-x86_64` D08 lock on CPython 3.11.9 under a safe Unicode path;
- construction of a Windows runtime wheelhouse artifact from exact hashes;
- a second `windows-2025` job that downloads only runtime and application artifacts, performs an offline install under a Unicode/space path, and exercises status, checkpoint selection, stdin passthrough, generation command routing, and local API command routing.

The packaging workflow deliberately records canonical checkpoint execution as `NOT_TESTED_IN_THIS_PACKAGING_PR`. Actual trained/random-init compatible checkpoint loading and raw Torch generation belongs to the Windows runtime milestone, not to a synthetic packaging proof.

Manual NVDA accessibility is also `NOT_TESTED`. The workflow proves a keyboard/text-only interface shape and Unicode/stdin/stdout behavior; it does not claim a human screen-reader PASS.
