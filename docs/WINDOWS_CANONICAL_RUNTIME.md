# Canonical Windows runtime evidence

This lane is the execution successor to the Windows-safe CLI transport proof. It is not a second inference backend and it does not reinterpret generation semantics.

## Physical repository blocker

The physical GitHub repository name is `12-6-ai.` with a trailing period. That name is not a valid normal Windows checkout directory component. The Windows authority job therefore performs no repository checkout and does not rename or hide the repository identity. An Ubuntu producer checks out the exact source SHA, builds sealed artifacts, and the Windows job operates only under `RUNNER_TEMP`.

The Windows execution root intentionally contains both spaces and Ukrainian Unicode. The evidence records `repository_checkout_used=false`, `artifact_only_safe_path_used=true`, and the trailing-dot blocker explicitly.

## Exact runtime

The Windows runtime profile is `requirements/locks/windows-x86_64/profile.json`.

- CPython: `3.11.9`
- Runtime lock: 12 exact distributions, SHA-256 `378f29100558c527e6ffca2ed5a679b93eafcd5b45a1cc926b2acaf2890f160e`
- Toolchain lock: 4 exact distributions, SHA-256 `fb96f952476295bdc43284345cd5102ad17de64713e5529a66c09e8cd935f1fd`
- Dev lock: 7 exact distributions, SHA-256 `7356d8ee1bd1b58afd27b3346f56f40bc17b0820dc991cbe11e5bdd84006567b`
- Windows profile semantic hash: `0c535270e1cae548a0fdd75892c2a4496734ee8a02d0f618d1416375889c0008`

The runtime lock includes `torch==2.13.0`, `numpy==2.4.6`, `safetensors==0.8.0`, and the exact transitive Windows runtime closure required by the PyPI Windows Torch wheel. Linux CUDA/NVIDIA/Triton wheels are not part of the Windows lock.

The Windows job installs toolchain and runtime from a sealed wheelhouse with `--no-index`, `--require-hashes`, and `--no-deps`. The project wheel is built from the exact source SHA by the locked Linux producer, SHA-256 sealed into the artifact manifest, and installed locally with no dependency resolution.

## Checkpoint and generation authority

The producer reuses `twelve_six.inference.s0_artifact` to build the retained real S0 checkpoint. That path already binds the random-init Base semantics, D05 checkpoint identity, D07 first-party loader, exact direct/reloaded parity, and first-party generation. Windows consumes that checkpoint; it does not retrain, substitute foreign weights, or duplicate generation logic.

On Windows the lane must complete all of the following before it can claim canonical execution evidence:

1. Verify every sealed artifact byte against `artifact-manifest.json`.
2. Verify the self-hashed Windows D08 profile and every committed lock file.
3. Verify the Windows toolchain/runtime wheelhouse exactly matches the committed package/version/hash sets.
4. Create a venv under a path containing spaces and Ukrainian Unicode.
5. Install the exact runtime offline and require `pip check` to pass.
6. Run `python -m twelve_six.inference.s0_artifact validate` against the retained checkpoint bytes.
7. Invoke the installed `twelve-six-generate` console script with Ukrainian UTF-8 text through stdin.
8. Execute both plain-output mode and `--json` diagnostics mode and require backend diagnostics to report `first_party_torch`, the exact source SHA, and the retained checkpoint ID.

The resulting machine evidence uses schema `12-6.windows-canonical-checkpoint-execution.v1` and authority `FREE_HOSTED_CPU_WINDOWS_RUNTIME_EVIDENCE_NOT_PROMOTION`.

## Accessibility truth boundary

The automated lane may establish keyboard/text-interface suitability: stdin piping, text stdout, JSON diagnostics, no interactive TTY requirement, Unicode transport, and safe paths.

It does **not** establish manual NVDA accessibility. Until a human actually executes the interface with NVDA, the machine evidence must remain `manual_nvda_accessibility=NOT_TESTED_REQUIRES_HUMAN`.

This lane is also not promotion authority and does not imply candidate/stable release approval.
