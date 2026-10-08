# Dependency security evidence

This package extends the exact dependency locks from PR #58 without changing `pyproject.toml`, the committed lock files, or the primary CI workflow. It addresses the remaining evidence portion of AUDIT-B B-003: a deterministic lock-derived software inventory plus current vulnerability and license metadata observations bound to the exact source and lock identities.

## Authority chain

The dependency-security report is valid only when all of the following match:

- physical repository identity `Oleksii-debug/12-6-ai.`;
- full exact source Git SHA;
- current `requirements/locks/index.json` semantic identity;
- current physical SHA-256 of `requirements/locks/index.json`;
- current profile-manifest identities for every supported lock profile;
- the exact component set and versions reconstructed from the committed hash locks;
- deterministic SBOM SHA-256;
- complete OSV and PyPI evidence-source status;
- report self-hash;
- optional freshness limit enforced by the offline verifier.

Any current-lock drift, source-SHA mismatch, missing/extra component, malformed evidence record, self-hash change, incomplete scan source, future timestamp, or stale report is rejected.

## Deterministic lock-derived SBOM

`build_lock_sbom()` validates the complete dependency-lock index first and then reconstructs every exact distribution from the toolchain/runtime/dev lock groups for both supported Linux profiles. It records:

- normalized PyPI distribution name and exact version;
- package URL (`pkg:pypi/...`);
- lock-group membership;
- locked artifact SHA-256 values;
- profile membership and profile-manifest identities;
- exact source SHA and lock-index identities.

The resulting `12-6.dependency-sbom.v1` object is self-hashed. It is a project evidence schema, not a claim of external CycloneDX/SPDX certification.

## Current vulnerability and license observations

`tools/collect_dependency_security_evidence.py` uses only the Python standard library and does not resolve or install any package. It queries:

- OSV `querybatch` with exact PyPI name/version pairs from the committed locks;
- PyPI JSON metadata for the same exact package versions.

The collector retains compact vulnerability IDs/aliases/modified timestamps, per-result digests, PyPI metadata digests, declared license-expression/classifier evidence, and hashes rather than copying long license text. Every locked component must have both an OSV query result and a PyPI metadata result or collection fails.

Evidence status is deliberately separate from audit or release authority:

- `EVIDENCE_COMPLETE_NO_REVIEW_FINDINGS` means the collector observed no OSV vulnerabilities and no unresolved license declaration in the queried evidence at that time;
- `EVIDENCE_COMPLETE_REVIEW_REQUIRED` means collection completed but at least one vulnerability or unresolved license declaration requires human/domain adjudication.

A successful evidence workflow can therefore carry `REVIEW_REQUIRED`. This is intentional: a finding is not converted into an infrastructure failure or silently represented as an audit PASS.

## Offline verification and freshness

`tools/verify_dependency_security_evidence.py` performs no network access. It rebuilds the SBOM from the current repository locks, verifies the exact source SHA, report self-hash, component identities, scan completeness, review status, truth boundary, and a caller-selected maximum evidence age.

For release/candidate use, D10 should supply the exact candidate head and a bounded freshness window. A report from a different SHA, an earlier lock, or an expired evidence window must not be reused.

The optional `--require-no-review-findings` flag is a strict policy gate. It returns non-zero when evidence is complete but requires review. The normal collection workflow does not enable that flag because vulnerability/license risk adjudication is not delegated to the collector.

## CI artifact

`.github/workflows/dependency-security-evidence.yml` is intentionally separate from the primary locked CI workflow. It uses immutable action SHAs, exact CPython 3.11.16, no `pip install`, and uploads both the source-bound SBOM and security-evidence JSON for 30 days. Collection failure is CI-red; vulnerability/license findings remain explicit in the retained report.

The inherited PR #58 primary CI still provides the authoritative locked clean-install, editable/wheel/import/CLI smoke, repository policy, Ruff, focused S0 integration, repo-wide pytest, and stage-candidate validation path.

## D10 / AUDIT-B handoff

This package supplies evidence, not a verdict. D10 may bind the retained SBOM/evidence artifact hashes to a future exact candidate/release attestation. AUDIT-B remains the independent authority for B-003 closure and any license/vulnerability-risk disposition.

No CANDIDATE/STABLE state, audit verdict, paid compute authorization, Base behavior, foreign pretrained weight, instruction/alignment/refusal behavior, or model-quality claim is created by this package.
