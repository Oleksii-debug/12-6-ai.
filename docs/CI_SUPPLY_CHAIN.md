# CI Supply-Chain Boundary

This document records D10 integration/CI controls. It is not a claim that the repository supply chain is fully audited.

## Pinned workflow inputs

The primary CI workflow pins third-party actions by immutable full commit SHA rather than mutable major-version tags:

- `actions/checkout` v7.0.1: `3d3c42e5aac5ba805825da76410c181273ba90b1`;
- `actions/setup-python` v7.0.0: `5fda3b95a4ea91299a34e894583c3862153e4b97`.

The checkout is full-history (`fetch-depth: 0`) so future candidate ancestry validation can inspect source commits, and persisted checkout credentials are disabled. Python is pinned to 3.11.16 for this CI lane. CI no longer performs an unbounded `pip install --upgrade pip`; it records the provisioned pip version and installs the project with pip's version check disabled.

## Why this is only a partial B-003 repair

The following remain required before AUDIT-B B-003 can be considered closed:

- a reviewed, hash-locked dependency/environment lock for the final integrated candidate rather than only lower-bound package constraints;
- secret scanning of the current tree and relevant Git history using a maintained scanner;
- explicit repository file-size/binary/artifact policy enforcement;
- dependency vulnerability/license review and an SBOM or equivalent release inventory;
- review of every additional GitHub Action introduced by specialist workflows, each pinned by immutable SHA;
- release-time verification that the locked dependency graph and environment identity match the candidate/checkpoint manifests.

These controls must be evaluated on the exact integrated candidate. Pinning two actions and removing a floating pip upgrade is risk reduction, not a supply-chain PASS.

## Update policy

When changing an action pin, record the upstream release/tag and exact commit SHA in the PR evidence. Newer versions are not automatically authoritative; the resulting exact-head CI and audit evidence remain required.
