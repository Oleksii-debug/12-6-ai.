# CI Supply-Chain Boundary

This document records D10 integration/CI controls. It is not a claim that the repository supply chain is fully audited.

## Pinned workflow inputs

The primary CI workflow pins third-party actions by immutable full commit SHA rather than mutable major-version tags:

- `actions/checkout` v7.0.1: `3d3c42e5aac5ba805825da76410c181273ba90b1`;
- `gitleaks/gitleaks-action` v3.0.0: `e0c47f4f8be36e29cdc102c57e68cb5cbf0e8d1e`;
- `actions/setup-python` v7.0.0: `5fda3b95a4ea91299a34e894583c3862153e4b97`.

The checkout is full-history (`fetch-depth: 0`) so candidate ancestry validation and the maintained Gitleaks secret scanner can inspect Git history. Persisted checkout credentials are disabled. Gitleaks PR comments and SARIF artifact upload are disabled so the scan does not require broader write permissions or create extra artifacts.

Python is pinned to 3.11.16 for this CI lane. CI no longer performs an unbounded `pip install --upgrade pip`; it records the provisioned pip version and installs the project with pip's version check disabled.

## Tracked repository artifact policy

`tools/check_repo_policy.py` validates exact `git ls-files` content before tests. Current fail-closed policy rejects:

- tracked files larger than 5 MiB;
- model/checkpoint payload formats: SafeTensors, CKPT, PTH, PT and GGUF;
- tracked archives: ZIP, 7z, RAR, TAR/TGZ/TAR.GZ;
- tracked top-level `artifacts/` or `checkpoints/` trees;
- tracked symlinks and unsafe relative paths.

Large checkpoints/run outputs remain external artifacts referenced by manifests and hashes rather than Git content. The size threshold and format allow/deny policy are integration controls and can only be changed deliberately with review.

## Evidence boundary

The scanner and repository policy are only configured gates until their exact-head workflow completes successfully. A green Gitleaks result means the maintained scanner found no configured leak on that tested checkout/history; it is not proof that no secret can exist outside the scanner's rules. Any real leaked credential must be rotated even if later removed from Git history.

## Why this is only a partial B-003 repair

The following remain required before AUDIT-B B-003 can be considered closed:

- a reviewed, hash-locked dependency/environment lock for the final integrated candidate rather than only lower-bound package constraints;
- dependency vulnerability/license review and an SBOM or equivalent release inventory;
- review of every additional GitHub Action introduced by specialist workflows, each pinned by immutable SHA;
- release-time verification that the locked dependency graph and environment identity match the candidate/checkpoint manifests;
- independent AUDIT-B retest of the secret-scan/artifact-policy results and the exact integrated candidate/release surface.

These controls must be evaluated on the exact integrated candidate. Action pinning, a maintained history scan, repository artifact policy, and removal of a floating pip upgrade are risk reduction, not a supply-chain PASS.

## Update policy

When changing an action pin, record the upstream release/tag and exact commit SHA in the PR evidence. Newer versions are not automatically authoritative; the resulting exact-head CI and audit evidence remain required.
