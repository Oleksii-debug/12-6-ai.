# Plan 1 Section 6 — Migration Contract Baseline v1 package

The additive import surface `twelve_six.contracts` exports the **same Python classes
and functions** owned by `artifact_identity` and `system_architecture`. It deliberately
does not implement training, evaluation, inference, agent, provider, checkpoint,
registry, scheduler, or qualification engines.

- `BASELINE_SCHEMA_VERSION=1` records the already accepted Plan-1 S1/S2 baseline.
- `ArtifactRef`, `ArtifactManifest`, `GenerationIdentityManifest` and their existing
  canonical parser retain byte-identical serialization and fail-closed identities.
- `ContractEvidenceRef` binds an opaque SHA-256 evidence identity to an exact Git commit.
  It never asserts a PASS, signs an artifact or grants an authority.
- `ContractSignal` is inert versioned error/lifecycle interchange. A VERIFIED event is
  merely a claim to be independently checked by the existing acceptance authority.
- `ContractEvolutionReceipt` rejects semantic drift under the same schema version,
  downgrade, unversioned migrations, missing fixture digest or absent impact analysis.
  A future version with a fixture is a **requalification request**, never silent approval.

Plans 2–8 may keep importing the already accepted baseline; there is no Plan-1
completion barrier. Future breaking semantics require a new version, migration
fixtures, compatibility tests and demonstrated impact before any other plan
could be reopened. Executable capability/readiness remains solely Plan 8.
Real product packaging, signing and acceptance remain Plan 10.

Scoped checks:
`pytest -q tests/test_plan1_section6_contracts.py tests/test_artifact_identity_section1.py
tests/test_system_architecture_section0.py`; `ruff check src/twelve_six/contracts.py
tests/test_plan1_section6_contracts.py`.
