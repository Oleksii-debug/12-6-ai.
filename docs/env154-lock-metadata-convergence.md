# ENV-154 lock metadata convergence

ENV-154 preserves the exact-hash dependency contract while separating purpose-scoped experiment verification from repository-wide release verification.

## Historical failure reproduced

TRAIN-41 GitHub Actions run `32862102098`, exact source `55c452e651ce1254d2bd21c3ec7746ee26ac6ee7`, stopped before environment creation. `tools/verify_locked_environment.py --profile linux-x86_64` entered aggregate `validate_lock_index()` and failed with `profile is stale for current pyproject.toml`. The experiment never reached its locked install or training step.

ENV-154 retains this failure mode as an adversarial regression: a changed `pyproject.toml` invalidates the referenced canonical profile until metadata is deterministically re-derived.

## Authority model

The component lock text remains authoritative dependency content. ENV-154 never resolves, upgrades, removes, or adds packages.

`tools/converge_environment_metadata.py` is the single derivation path for metadata identities. It derives canonical platform profile manifests from current `pyproject.toml` plus exact toolchain/runtime/dev lock bytes, derives `requirements/locks/index.json` from those manifests, refreshes purpose-profile canonical/base references and purpose lock identities, and derives `requirements/profiles/index.json` from those purpose profiles.

There are no manually maintained aggregate SHA tables in this path. A changed component byte necessarily changes its derived profile identity and every aggregate/reference identity that depends on it.

## Verification scopes

Ordinary `validate_lock_index()` verification is purpose/platform scoped and validates the current canonical platform profile unless an explicit profile set is supplied. An unrelated optional or other-platform profile therefore cannot prevent a narrow experiment from validating its own exact environment.

Release/repository verification must call `validate_global_lock_index()`. That path validates every canonical profile and remains fail-closed on any stale toolchain/runtime/dev or project-metadata binding.

Purpose-specific D08 profiles continue to use `tools/verify_purpose_environment.py`; tokenizer/Transformers overlays and the CUDA base-role remain bound to exact canonical identities.

## Migration commands

After an intentional exact lock-byte or `pyproject.toml` change:

```bash
python tools/converge_environment_metadata.py --write
python tools/converge_environment_metadata.py --check
```

Review and commit only the resulting metadata JSON changes together with the intentional source lock/project change. `--write` rejects floating, unhashed, or duplicate lock lines and is required to be byte-idempotent.

For a narrow Linux x86_64 experiment:

```bash
python tools/verify_locked_environment.py --profile linux-x86_64 --validate-only
```

For release/global verification:

```bash
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from twelve_six.integration.dependency_lock import validate_global_lock_index
validate_global_lock_index(root=Path('.'), index_path='requirements/locks/index.json')
print('PASS')
PY
```

The ENV-154 workflow installs exact toolchain/runtime/dev locks before invoking pytest or Ruff, performs import preflight, runs adversarial drift tests, then executes `--write` and requires a clean metadata diff to prove deterministic idempotence.
