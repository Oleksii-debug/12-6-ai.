# Plan 1 Section 8 — foundation qualification (candidate, NOT DONE)

This is the only unfinished Section in the eight-Section `1. Перший план`.
Sections 1–7 are already terminal DONE and must not be reopened merely to run this
qualification. No Section 9 exists in this plan.

## Existing implementation reused

- Architecture and canonical artifact identity: Plan-1 S1–S2 incumbent modules.
- Fail-closed third-party reuse/Base ancestry: existing S3 tests.
- Exact CPython 3.11.16 lock/index and trust boundaries: incumbent S4.
- Source-bound SBOM, unresolved notices and fixture-only signing/update/rollback:
  incumbent S5. No production release approval is asserted.
- Versioned Migration Contract Baseline v1: incumbent S6 reexports, not a fork.
- Static environment/backend matrix: incumbent S7, always UNQUALIFIED; Plan 8
  exclusively owns executable readiness.

## Exact candidate check

Run `python tools/plan1_section8_qualify.py` in a clean checkout with dev
dependencies already present. The script rejects modified tracked files, binds
checks to the full Git commit SHA, checks scoped Ruff, composes all Plan-1
pytest suites including adversarial and recovery cases, compiles package
sources, and builds the project wheel twice *offline* with a fixed epoch.
Only byte-identical wheels qualify. On success it prints a machine-readable
receipt including the exact candidate SHA and wheel SHA-256. It never sets
DONE by itself. CI invokes the same runner on the canonical Section-8 branch
via the existing shared workflow; no extra workflow is introduced.

Any command failure, reproducibility drift or source mutation is a hard FAIL.
`git`, `ruff`, `pytest`, `pip` and local wheel build must actually execute;
their absence is not a PASS. This script is not a physical Windows/Linux
clean-room install, GPU qualification, Plan-8 readiness proof, Plan-9 training
approval or Plan-10 final product release. No paid compute is authorized.

## Final closure requirements

1. Inspect exact frozen PR head and the evidence of the completed qualification
   on that exact SHA. Fix real applicable failures before claiming PASS.
2. Merge a qualified candidate into `main`, read back accepted code, tests and
   evidence with exact Git SHAs, and ensure the accepted source wasn't changed
   after qualification.
3. Verify the migration contract remains compatible without reopening Plans
   2–8 just because Plan-1 packaging is complete.
4. Only then mark Plan 1 / Section 8 terminal DONE in
   `MULTI_PLAN_CLOSURE_STATE.md` and the assigned Drive `1. Перший план`,
   with actual SHA and tests. This closes the entire Plan 1.
