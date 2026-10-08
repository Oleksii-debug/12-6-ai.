# Plan 1 / Section 5 — supply-chain and deterministic fixture boundaries

The accepted Plan 1 S4 CPython 3.11.16 dependency-lock index is the sole installed-environment authority. The historical D08 SBOM implementation is reused, adapted to Linux x86-64, Linux arm64 and Windows x86-64, and kept separate from Plan 10 final release promotion.

## Third-party inventory

The SBOM enumerates the exact hashed runtime, developer and toolchain lock components for all three supported platforms. Optional research candidates are NOT admitted merely because their names appear in a roadmap. The current mandatory project runtime is declared by the pyproject dependency set; there is no automatic promotion of foreign model weights or third-party corpus data.

The evidence collector observes PyPI license metadata and OSV vulnerability responses, preserving an explicit review-required state. The notice inventory binds every component/version/profile and publisher metadata SHA256 to the current SBOM. An absent/unknown license, unverified license-text body, missing source proof, or missing advisory observation must NEVER be treated as an approval. No license text or vulnerability risk approval is manufactured. License compatibility and final legal review are separate release gates.

## Reusable engineering fixtures only

The new release_fixture module provides deterministic ZIP bytes, SHA256 identity inspection, a test-key HMAC receipt, monotonic update admission and rollback only to the digest embedded by the immediately preceding receipt. Negative tests cover symlinks, traversal, duplicate entries, wrong key, wrong digest, forged receipt, downgrade, rollback mismatch, corrupt archives and deterministic clean rebuild.

HMAC fixture receipts are NOT production code-signing. No signing secrets are stored. Plan 10 owns final production signing, champion assembly, operator update and physical release; these fixtures cannot declare release-ready.

## Qualification

Run: python -m pytest -q tests/test_dependency_security.py tests/test_plan1_section5_release_fixture.py tests/test_plan1_section5_notices.py

Run: python -m ruff check src/twelve_six/integration/dependency_security.py src/twelve_six/integration/release_fixture.py tests/test_dependency_security.py tests/test_plan1_section5_release_fixture.py tests/test_plan1_section5_notices.py

Exact-head Actions and accepted-main readback are mandatory before terminal DONE. A queued or failing check is not PASS.
