# Sequential Closure State

This file is the durable GitHub mirror for ordered Section/Subsection closure.

## Binding closure lifecycle v2

This registry obeys the root `AGENTS.md` **Terminal Section Closure Protocol v2**. The following invariants are mandatory when selecting or updating a front:

- **Closure is the optimization target.** Commit/PR count, execution-unit floors, depth targets and elapsed worker time do not justify additional mutation.
- **One mutation front.** Mutate only the earliest actionable unfinished Section, except for a minimal named direct dependency required to close it.
- **Audit existing first.** If acceptance-critical implementation already exists, qualify/close it instead of rebuilding or expanding it.
- **One canonical finisher.** Record/reuse one Section finisher lineage; intermediate feature/integration/prequal merges are not closure.
- **Candidate freeze.** Once internally controllable acceptance requirements are satisfied, designate and freeze an exact candidate SHA. No unrelated hardening or speculative edge-case work after freeze.
- **Exact-SHA qualification.** Pending CI freezes the candidate; it does not authorize a new SHA. A failed gate permits only the smallest proven gating repair before refreeze.
- **Integration then readback.** DONE requires required canonical integration plus post-merge/readback evidence, not merely an intermediate merge.
- **External-only remainder.** Use `INTERNAL_DONE_BLOCKED_EXTERNAL` when internal scope is exhausted and a genuinely external fact remains. This is not DONE, but the frozen Section becomes immutable for autonomous sequencing until the unblock condition changes.
- **No reconvergence carousel.** Do not repeatedly propagate a moving predecessor into later Sections. Later work waits for a frozen/accepted predecessor or the explicit external-block escape.
- **Acceptance boundary is fixed.** New non-gating improvements discovered after freeze go to later/backlog scope. They do not silently enlarge the current Section.
- **Reopen narrowly.** A DONE Section may reopen only for a demonstrated regression, invalid evidence, changed acceptance contract or breaking later integration; record the exact reason first.

Recommended lifecycle states are:
`OPEN -> IMPLEMENTING -> CANDIDATE_FROZEN -> QUALIFYING -> DONE`,
or `... -> INTERNAL_DONE_BLOCKED_EXTERNAL` when only an external unblock remains.
`REOPENED` is exceptional and must name the invalidated surface.

For the current front, durable state should identify: **canonical finisher**, **candidate SHA if frozen**, **remaining acceptance-critical gap**, and **exact unblock condition if externally blocked**.

## Rules

- Read the current canonical Section plan and live default branch before updating this file.
- Record only evidence-backed DONE state; never infer closure from a chat summary, a PR existing, queued CI, or a single green test.
- Once recorded DONE, a Section/Subsection is skipped by normal workers and is not re-entered unless it is explicitly marked REOPENED for a demonstrated regression, invalidated evidence, changed acceptance contract, or broken later integration.
- If parallel workers produce duplicate closure lineages, preserve one canonical lineage, converge unique required changes, then close/supersede the duplicate and delete the duplicate branch when safe.
- Update this ledger in the same run that closes or reopens scope.

## Closure registry

| Section / Subsection | State | Canonical evidence / lineage | Accepted source / build | Notes |
| --- | --- | --- | --- | --- |
| Section 0 — Цільова архітектура 12-6 як повної AI-системи | DONE | PR #3074 merged; candidate `4a1486798d8da3ac59bf1a1e6a7673dcb490afaa`; exact-head CI `37606263374` terminal SUCCESS | `main@d74c5c4f6dee7027ca713b3e6579bb66a5fefaf0` | Current 96-Section plan acceptance 0.1/0.2 is integrated: typed seven-plane architecture boundaries and replaceable cognitive-core/runtime-shell contract. Ruff + full pytest passed on the exact candidate head; merge preserved the qualified candidate tree. No training/data/paid-compute/release authority was widened. |
| Section 1 — Єдина система ідентичностей і маніфестів | DONE | PR #3090 merged; candidate `0e1f301c5123b4e52c111cb94264cfd61b60bf4b`; exact-head CI `37615156740` terminal SUCCESS | `main@9ef945ff977dd4674a9b7cf4d6fd2efff6302eeb` | Current-plan 1.1/1.2 identity and cryptographic cross-binding authority is repaired and integrated. Ruff + full pytest passed on the exact candidate head; merge commit preserves the qualified candidate tree byte-for-byte. Reopened capability/source truth remained fail-closed during repair; no training/data/final-test/paid-compute/physical/release authority was widened. |
| Section 2 — Executable capability map і acceptance graph | CANDIDATE_FROZEN / QUALIFYING | canonical finisher PR #3093; exact candidate `b7d0af0f4d0cc3d05d7593142eb882f5ab5640dc`; promotion CI `37627655061`; phase-1 PR #3092 already merged/qualified | phase-1 integrated at `main@ee7ade7e80e9e4e6fe7ffd5ebd5012a61bbc5fb6`; phase-2 candidate not accepted until CI/integration/readback | **PRIMARY FROZEN FRONT.** Candidate promotes executable-capability-map to AVAILABLE, source inventory to 117 accepted-main / 0 overlays, and removes candidate override. Pending CI does not authorize mutation. Remaining path: exact-head CI SUCCESS -> guarded integration of this SHA -> post-merge readback -> durable DONE. Sections 3-6 remain parked drafts. |
