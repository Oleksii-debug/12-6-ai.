# Section 2 — repository executable/control surface coverage

## Scope

This evidence closes a completeness gap in the Section-2 capability graph. The existing
package-source inventory covers `src/twelve_six/**/*.py`, but accepted main also contains
executable or control-bearing repository surfaces outside that package root.

The machine-readable authority is:

`configs/control/product_repository_executable_surface_rules_v1.json`

The validator is:

`tools/validate_section2_repository_surface_coverage.py`

Permanent adversarial coverage is:

`tests/test_repository_surface_inventory_section2.py`

## Qualified baseline and current-main equivalence

The rule authority preserves the terminally qualified product baseline:

- qualified `main@019944d5fe12334791f05f1232d13de4a12e37d3`;
- exact baseline tree `c727add7897dd94bdb02493e0cd7a565be7e8d9f`;
- terminal baseline CI remains the separately recorded run `37248299503` and is **not**
  reassigned to a newer commit.

It also pins the repository main observed for this candidate:

- current `main@330fb46aa3199e26d8b7e49968ee93fb12447560`;
- exact current tree `20559f951b6e2c744832ce0cd1cb324e2e1c2eea`;
- the validator requires that SHA to equal the live `origin/main`/local `main` ref;
- all **233 capability-bearing blobs** across `src/twelve_six/**/*.py`, `tools/`,
  workflows and `pyproject.toml` must have an identical path→blob-SHA map between the
  qualified baseline and current main.

The only baseline→current-main changes are coordination-only `AGENTS.md` and
`SEQUENTIAL_CLOSURE_STATE.md`; neither is a capability-bearing product/executable surface.
Therefore the older terminal CI remains evidence for the unchanged product baseline without
falsely claiming that CI ran on `330fb46aa3199e26d8b7e49968ee93fb12447560`.

The external executable/control classification covers:

- all 117 accepted-main `tools/` executable surfaces;
- the shared `.github/workflows/ci.yml` control surface;
- `pyproject.toml` packaging/entry-point surface.

That is 119/119 accepted-main repository executable/control surfaces outside the
`src/twelve_six/**/*.py` package-source inventory.

Rules are deliberately semantic and closed: every main surface must match exactly one
capability rule. The exact per-capability distribution is pinned. A new file that matches no
rule, an ambiguous rule, a changed count/distribution, an unknown capability, or a missing
user/operator journey fails validation.

## Candidate overlay rule

A later candidate may not hide a newly added executable/control file behind accepted-main
coverage. Every path present in the checkout but absent from the pinned main tree must be
listed explicitly in `candidate_overrides`.

This validator classifies itself as the single current candidate overlay and binds it to
`executable-capability-map`. It therefore does not exempt its own implementation from the
coverage rule it enforces.

## Truth boundary

This inventory classifies repository surfaces into capability families. It does not upgrade
an UNAVAILABLE capability to AVAILABLE, does not grant learned-weight/training/final-test
authority, and does not substitute static coverage for exact-head shared CI or predecessor
closure. Section 2 remains IN_PROGRESS until those gates close.
