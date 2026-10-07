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

The rule authority binds the qualified repaired Section-1 baseline:

- accepted `main@49218c0c581b73bcd0985646f48bf35300b1948c`;
- exact baseline tree `2f8c32273994595b0bd466a293ee727d6f56d2ef`;
- that tree is identical to exact-head qualified candidate
  `49218c0c581b73bcd0985646f48bf35300b1948c`, whose CI run `37608406911`
  is terminal SUCCESS.

It also pins the live repository main observed for this candidate:

- current `main@5c041ca56edda55a5c3334f722361754051e121c`;
- exact current tree `95ad101c8965fd49c1711027146253a093fce2f8`;
- the validator requires that SHA to equal the live `origin/main`/local `main` ref;
- all **235 capability-bearing blobs** across `src/twelve_six/**/*.py`, `tools/`,
  workflows and `pyproject.toml` must have an identical path→blob-SHA map between the
  qualified accepted tree and current main.

The accepted merge→current-main change is the Section-1 terminal ledger update in
`SEQUENTIAL_CLOSURE_STATE.md`, which is not a capability-bearing product/executable surface.
The exact candidate CI is used only through proven tree equivalence; this does not claim that CI
ran on the later ledger-only SHA.

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
