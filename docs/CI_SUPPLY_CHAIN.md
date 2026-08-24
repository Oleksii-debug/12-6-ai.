# CI Supply-Chain Boundary

This document records D10 integration/CI controls. It is not a claim that the repository supply chain is fully audited.

## Pinned workflow inputs

The primary CI workflow pins third-party actions by immutable full commit SHA rather than mutable major-version tags:

- `actions/checkout` v7.0.1: `3d3c42e5aac5ba805825da76410c181273ba90b1`;
- `gitleaks/gitleaks-action` v3.0.0: `e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e`;
- `actions/setup-python` v7.0.0: `5fda3b95a4ea91299a34e894583c3862153e4b97`;
- `actions/upload-artifact` v4.6.2: `ea165f8d65b6e75b540449e92b4886f43607fa02`.

The checkout is full-history (`fetch-depth: 0`) so candidate ancestry validation and the maintained Gitleaks secret scanner can inspect Git history. Persisted checkout credentials are disabled. Gitleaks PR comments and SARIF artifact upload are disabled so the scan does not require broader write permissions or create extra artifacts.

Python is pinned to 3.11.16 for this CI lane. CI no longer performs an unbounded `pip install --upgrade pip`; it records the provisioned pip version and installs the project with pip's version check disabled.

## Repository-wide workflow policy

`tools/check_workflow_policy.py` scans every `.github/workflows/*.yml` and `.yaml` file. This converts the primary-workflow pinning convention into an integration gate that also applies when specialist workflows are selectively composed.

The current fail-closed rules are:

- external GitHub Actions and reusable workflows must use a full lowercase 40-hex commit SHA;
- Docker actions must use a `sha256:` image digest rather than a mutable tag;
- local `./...` actions remain allowed because their content is already bound by the repository Git tree;
- every `actions/checkout` step must set `persist-credentials: false`;
- every `actions/setup-python` step must select one exact `X.Y.Z` runtime;
- workflows may not self-update pip using `pip install --upgrade pip` or `pip install -U pip`.

The implementation is deliberately a narrow line-oriented scanner rather than a partial YAML interpreter. It validates only supply-chain-sensitive workflow constructs whose GitHub syntax is line-oriented. Negative tests reproduce mutable action refs, mutable reusable-workflow refs, Docker tags, persisted checkout credentials, minor-only Python pins and floating pip upgrades.

## Tracked repository artifact policy

`tools/check_repo_policy.py` validates exact `git ls-files` content before tests. Current fail-closed policy rejects:

- tracked files larger than 5 MiB;
- model/checkpoint payload formats: SafeTensors, CKPT, PTH, PT and GGUF;
- tracked archives: ZIP, 7z, RAR, TAR/TGZ/TAR.GZ;
- tracked top-level `artifacts/` or `checkpoints/` trees;
- tracked symlinks and unsafe relative paths.

Large checkpoints/run outputs remain external artifacts referenced by manifests and hashes rather than Git content. The size threshold and format allow/deny policy are integration controls and can only be changed deliberately with review.

## Exact environment inventory

`tools/capture_environment_inventory.py` produces schema `12-6.environment-inventory.v1` after the CI environment is installed. The inventory binds:

- canonical repository identity;
- exact source Git object ID, using the pull-request head SHA rather than silently substituting the merge ref;
- exact Python implementation/version;
- all declared `build-system`, runtime and optional `pyproject.toml` requirements;
- installed normalized distribution names and exact versions;
- available license expression/license classifier metadata, with unresolved metadata counted explicitly;
- sanitized editable/VCS provenance without persisting local filesystem URLs;
- SHA-256 of each installed distribution `RECORD` when available.

The complete inventory has its own canonical SHA-256. `validate_environment_inventory()` rejects a tampered self-hash, truncated source SHA, inconsistent package count, invalid package RECORD hash or conflicting same-name installation evidence. Byte-identical duplicate metadata emitted by editable-install discovery is deterministically deduplicated.

CI retains the resulting JSON as a 30-day `ci-environment-inventory` workflow artifact using the immutable upload-action pin above. That artifact is intended to be consumed later by checkpoint/release composition. Retention does not itself make it a canonical checkpoint identity.

## Evidence boundary

The scanner, repository policy, workflow policy and inventory are only evidence for the exact head on which their workflow completed. A green Gitleaks result means the maintained scanner found no configured leak on that tested checkout/history; it is not proof that no secret can exist outside the scanner's rules. Any real leaked credential must be rotated even if later removed from Git history.

The environment inventory records observed installed state; it does **not** prove that a future install will resolve to the same state. It is also not vulnerability adjudication or legal/license approval.

## Why AUDIT-B B-003 remains partially open

The following still remain before B-003 can be considered closed:

- a reviewed, hash-locked full transitive dependency/environment lock for the final integrated candidate rather than only lower-bound package constraints plus observed installed inventory;
- dependency vulnerability adjudication and owner/auditor handling of unresolved license metadata;
- release-time verification that the locked dependency graph and exact environment-inventory hash match the candidate/checkpoint manifests;
- independent AUDIT-B retest of secret scan, repository/workflow policy, environment artifact, dependency lock and the exact integrated candidate/release surface.

These controls must be evaluated on the exact integrated candidate. No supply-chain PASS, release, STABLE promotion or audit authority is implied by this package.

## Update policy

When changing an action pin, record the upstream release/tag and exact commit SHA in PR evidence. Newer versions are not automatically authoritative; the resulting exact-head CI and audit evidence remain required.
