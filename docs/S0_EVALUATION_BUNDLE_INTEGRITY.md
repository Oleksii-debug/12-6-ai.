# S0 exact-candidate evaluation bundle integrity

The D04 strict evaluator produces quality evidence only. A green quality run must not be
confused with audit or promotion authority.

The dedicated D04 workflow now proves two separate facts on the same full candidate SHA:

1. D08 locked-environment verification succeeds for Linux x86_64, exact CPython 3.11.16,
   exact lock/profile identities, clean editable/wheel installs, and repository checks.
2. The real D04 evaluation executes from a venv installed only from those hash-locked
   toolchain/runtime/dev lock files plus the offline local project checkout.

After evaluation, `tools/validate_s0_evaluation_bundle.py` validates the runtime evidence
and cross-binds these exact files:

- `candidate_evidence.json`
- `stage_gate_report.json`
- `promotion_eligibility.json`
- `locked-environment-linux-x86_64.json`

It rejects stale runtime source SHAs, lock/profile drift, missing repository-check PASS,
candidate/report SHA mixing, diverging promotion summaries/blockers, missing report hashes,
and bundle tampering. The output `evaluation_bundle.json` contains per-report SHA-256 values,
the exact D08 runtime evidence SHA-256, core candidate identities, quality summary, and a
self-hash.

`quality_overall_status=PASS` and 15/15 D06 quality gates are still not promotion. Exact
candidate CI, D10 integration/release authority, protected governance, and independent
candidate-bound AUDIT-A/AUDIT-B verdicts remain external requirements.
