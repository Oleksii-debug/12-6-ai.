# Multi-Plan Parallel Closure Protocol v4 — owner directive

**This v4 directive overrides every older conflicting global-sequential / one-mutation-front / earliest-monolithic-Section rule in this repository.**

## Canonical work-selection authority

- Read `PROJECT_PLAN_INDEX.md`, `MULTI_PLAN_CLOSURE_STATE.md`, the numbered Drive plan assigned by the owner, and live GitHub.
- The former monolithic 96-Section plan is historical/audit material only and MUST NOT be used to choose the next work front.
- New plan Sections start at 1; there is no Section 0 in the new execution plans.

## Independent plan law

- Plans 1–8 are independent engineering plans.
- There is **no priority order between Plans 1–8**. Plan 7 may finish before Plan 2; Plan 5 may be started before Plan 1 is terminal.
- When assigned a plan, skip all Sections in that plan already marked DONE and work the numerically first unfinished Section **inside that plan**.
- Parallel workers may work on different numbered plans simultaneously.
- A worker assigned to one plan must not silently mutate another plan's scope. Use versioned contracts/fixtures/mocks for missing cross-plan implementations.
- A fixture/mock may prove component behavior but never substitutes for final cross-plan integration evidence.

## Convergence plans

- Plan 9 is a real training/champion convergence plan. It consumes required terminal artifacts from the engineering plans and may wait only on the specific inputs its current Section actually requires.
- Plan 10 is final whole-product integration/release and owns Nika integration, whole-product scenarios, physical Windows/NVDA/HIL, server/cloud parity, packaging/release and final qualification.

## Migrated terminal work

- Plan 1 / Section 1 is DONE from former Section 0.
- Plan 1 / Section 2 is DONE from former Section 1.
- Plan 8 / Section 1 is DONE from former Section 2.
- Plan 8 / Section 2 is DONE from former Section 3.
- Plan 8 / Section 3 inherits former Section 4 and is QUALIFYING, not DONE unless live evidence later closes it.

## Closure semantics

The Simplified Section Closure Protocol v3 DONE rule remains binding **within each new plan**: finish all repository-controllable work, run available relevant checks, record terminal DONE durably, then advance to the next unfinished Section of that same plan. Human/NVDA acceptance remains final-product work unless the assigned final plan explicitly reaches that gate.

# AGENTS.md

## Simplified Section Closure Protocol v3 — owner directive 2026-10-07

**This v3 directive overrides Terminal Section Closure Protocol v2 and every older conflicting Section-closure rule in this repository.**

### DONE rule
A Section is `DONE` when all work that is controllable inside the repository has been completed and integrated, and all tests/checks that are actually available to autonomous workers have passed. Do not keep a Section open merely to wait for evidence that cannot presently be produced by the repository or its workers.

### Human/NVDA acceptance is final-product work, not an intermediate blocker
- Manual owner/NVDA testing MUST NOT block any intermediate Section.
- Do not ask the owner to test unfinished or partially assembled product scope.
- Do not use missing manual NVDA evidence as `INTERNAL_DONE_BLOCKED_EXTERNAL` for an intermediate Section.
- Run automated accessibility checks when they exist and are relevant, but reserve real owner/NVDA acceptance for the final whole-product handoff/release stage, when there is a genuinely usable build to test.
- Failure to have final owner/NVDA acceptance before that final stage is not a defect, blocker, or reason to slow sequential Section closure.

### External infrastructure
- A queued check that is expected to run normally may be awaited without mutating the frozen candidate.
- If hosted CI/runner/infrastructure is unavailable and workers cannot restore it, record that fact, use all repository-local/static/test evidence that is actually available, and do not keep an otherwise complete Section permanently open solely because the external runner did not execute.
- A known failing test/check remains a real blocker until repaired. "Unavailable" is not the same as "failed."

### Sequencing and terminal lock
- After the simplified DONE rule is met, record `DONE` durably and immediately advance to the next Section.
- A DONE Section is terminally skipped by ordinary workers.
- Reopen only for a concrete demonstrated regression, invalid closure evidence, a materially changed acceptance contract, or a later integration change that demonstrably broke the closed scope.
- Do not invent extra hardening, polishing, repeat audits, duplicate PRs, or owner-side testing merely to delay closure.

### Final acceptance
Final whole-product release may still require real user/NVDA/device acceptance where applicable. That requirement belongs at the final product acceptance/handoff gate only, after a usable build exists.


## Terminal Section Closure Protocol v2 — owner directive 2026-10-07

**This section overrides every older coordination rule in this repository, including any instruction to use the full execution window, keep creating residual work, prepare FRONT-2, avoid idling, satisfy a depth/work-unit floor, or keep mutating while CI is pending. Product correctness/safety requirements remain binding.**

### Objective

Optimize for **honestly closed Sections**, not commits, PR count, execution units, branch activity, or the amount of hardening performed.

A worker must prefer the shortest evidence-correct path from the current repository state to terminal closure of the earliest actionable Section.

### Mandatory lifecycle

For the earliest actionable Section, execute this lifecycle in order:

1. **AUDIT EXISTING** — map the fixed acceptance contract to live code, tests, evidence, and already integrated capabilities. Existing correct implementation is an asset, not a reason to reimplement.
2. **IMPLEMENT ONLY MISSING** — change only acceptance-critical gaps. Reuse/repair/converge existing canonical work before creating anything new.
3. **CANDIDATE** — as soon as all internally controllable acceptance requirements appear satisfied, designate one canonical finisher lineage and one candidate SHA.
4. **FREEZE** — freeze that candidate. After freeze, unrelated hardening, polishing, speculative edge-case hunting, refactors, extra features, and “while we are here” changes are forbidden.
5. **QUALIFY EXACT SHA** — run the required tests/gates against that exact candidate. Pending/queued CI does not authorize changing the SHA.
6. **REPAIR ONLY PROVEN GATING FAILURE** — if qualification fails, make the smallest acceptance-relevant repair on the same canonical finisher, refreeze a new SHA, and rerun affected gates. A discovered non-gating improvement goes to backlog/later QA scope.
7. **INTEGRATE** — merge/converge the qualified candidate into the canonical integration authority required by the plan.
8. **POST-MERGE READBACK** — verify accepted tree/SHA and any required downstream evidence after integration.
9. **DONE** — update `SEQUENTIAL_CLOSURE_STATE.md` and applicable canonical control record in the same closure run, then immediately select the next Section.

### One-front law

- There is exactly **one mutation front**: the earliest actionable not-closed Section.
- Creating or mutating FRONT-2/Section N+1 is **forbidden** while Section N still has internally controllable acceptance work.
- Work on a later Section is allowed only when it is a **named direct dependency required to close the primary Section**, and only the minimum dependency slice may be changed.
- Do not create speculative prequalification, next-section integration, or reconvergence branches merely because the primary candidate is waiting for CI.
- Repeatedly propagating a moving Section-N parent into Section N+1 is forbidden. Section N+1 waits for a frozen/accepted predecessor, except for the explicit external-block rule below.

### External-block escape without false DONE

If all internally controllable acceptance work is complete and the only remaining requirements are genuinely external (for example human NVDA/physical-device evidence, vendor/certificate/account approval, externally unavailable infrastructure, or an external fact that cannot be manufactured):

- freeze the exact internal candidate;
- record **INTERNAL_DONE_BLOCKED_EXTERNAL** with exact SHA, completed evidence, missing external evidence, and the precise unblock condition;
- treat that Section as **immutable for autonomous sequencing** while the external condition is unchanged;
- move to the next earliest Section whose implementation does not depend on the missing external fact;
- do **not** call the blocked Section DONE;
- do **not** reopen or mutate it just to consume worker time;
- when the external fact arrives, requalify only the dependency surface it can invalidate, then complete terminal closure.

Queued CI is not automatically an external blocker. If the frozen exact-SHA CI is merely pending, keep the candidate frozen and check its result; do not move the SHA or invent new work. If CI is demonstrably unavailable for an extended period and no autonomous action can restore it, record the exact infrastructure blocker before using this escape.

### One canonical finisher

- Each active Section has one canonical finisher branch/PR/lineage, recorded in durable state.
- Before creating a branch or PR, inspect active lineages. Reuse the canonical finisher whenever possible.
- Parallel workers may contribute non-overlapping acceptance-critical fixes, tests, or evidence, but those contributions must converge into the same finisher.
- Alternate integration trees, competing “whole Section” PRs, and reconvergence carousels are forbidden.
- A merged PR into an intermediate feature/integration/prequal branch is **not Section closure**.
- If duplicate lineages already exist, preserve unique required changes, converge once, supersede the rest, and stop propagating duplicates.

### Acceptance-contract boundary

- The canonical Section plan defines the acceptance boundary. Workers may not silently enlarge it.
- After CANDIDATE/FREEZE, a newly imagined edge case does not block closure unless it demonstrates violation of an existing acceptance requirement, regression, security/correctness invariant, or required negative/recovery case.
- Non-gating improvements must be recorded for later scope instead of extending the current Section indefinitely.
- A work-unit/depth floor, token budget, run duration, “do not stop after one PR”, or “use the full execution window” rule can **never** force extra mutation after the closure candidate is ready.

### Already-implemented Section rule

If the Section’s required capability already exists in current canonical code:

- do not rebuild it;
- perform a closure audit against the acceptance contract;
- reuse current implementation/evidence;
- add only missing tests/evidence/integration;
- create/freeze the closure candidate;
- qualify and close it.

The correct action for an already-implemented Section is **qualification and closure**, not invention of more implementation.

### CI and SHA discipline

- One qualification cycle = one frozen SHA.
- Never invalidate green/pending exact-head evidence with unrelated commits.
- Never claim PASS from queued, cancelled, skipped, stale-base, different-SHA, or intermediate-branch CI.
- On failure, repair the proven failure only, then refreeze.
- On success, integrate promptly; do not continue polishing the candidate.

### Reopen discipline

A DONE Section may be REOPENED only for a concrete demonstrated reason: regression, invalid closure evidence, materially changed acceptance contract, or later integration that breaks the closed scope.

Before mutation, record the exact reopen reason and affected evidence. Reopen only the invalidated surface; do not restart the entire Section by default.

### Required worker decision at every run

Before writing code, answer from live state:

1. What is the earliest actionable Section?
2. Does its required implementation already exist?
3. What exact acceptance-critical gap remains?
4. What is the one canonical finisher?
5. Is there already a frozen candidate SHA?
6. If frozen, am I permitted to mutate it? Only a proven gating failure permits that.
7. Can this run close the Section now? If yes, closure takes priority over every depth/work-unit target.

**Closure beats activity. Frozen candidates beat moving targets. Existing implementation beats reimplementation. Exact evidence beats PR count.**


## Canonical autonomous worker policy

This root file is the highest-priority repository instruction for worker coordination and supersedes older conflicting worker-count, WIP, ownership/claim, serialization, exclusive-integration, PR-order, and CI-wait restrictions.

- No repository-defined worker, coordinator, WIP, work-package, branch, or PR cap.
- Claims, leases, ownership, assignments, queues, and coordinator labels are advisory only and never block useful safe work.
- Workers may create branches, commits, PRs, fixes, tests, integration commits, and merges when GitHub permissions allow and evidence is adequate.
- Parallel implementation/testing must obey the Sequential closure authority below: stay on the earliest unfinished Section except for dependency-safe work that directly supports it or the next front.
- Queued, pending, slow, or unavailable CI is never by itself a reason to terminate; record it and continue work permitted by the current closure front.
- A blocked primary closure front may be left only after all safe internally controllable residual work is exhausted and the blocker is durably recorded.
- Do not idle because another PR, branch, worker, check, review, claim, or queue is active. If overlap occurs, switch to non-conflicting work or reconcile/rebase instead of abandoning the run.
- No repository-defined exclusive integration owner is required.
- Use the full execution window while useful safe work remains.

This removes orchestration throttles only. Product correctness, security, privacy, data integrity, accessibility, licensing, truthful tests/release evidence, model/research integrity, and other domain-specific safety requirements remain mandatory. Do not bypass externally enforced GitHub permissions or branch protections; continue useful work elsewhere while they are pending.

Chat history is temporary. Preserve meaningful progress in repository state.

## Sequential closure authority — owner directive 2026-10-07

This section is the controlling coordination rule if any older repository text, worker prompt, issue, roadmap note, swarm rule, claim/ownership rule, or “parallel lane” instruction conflicts with it.

- Read the current canonical ordered Section plan and the live repository state before choosing work.
- The numerically earliest Section that is not honestly closed is the primary closure front. The next unfinished Section may be prepared only when this is dependency-safe, directly removes a dependency, or the primary front is genuinely non-actionable after all safe internal residual work is exhausted.
- Parallel workers are allowed, but parallelism does not authorize skipping the ordered closure front. Workers should take non-overlapping residuals of the same current front or dependency-safe preparation for the next front.
- A Section or Subsection may be marked DONE only after its required implementation/integration and applicable tests, negative/failure/recovery evidence, accessibility/security/performance/packaging evidence, and exact durable source state are satisfied.
- Every DONE Section/Subsection must be recorded durably in GitHub before the worker moves on. Use `SEQUENTIAL_CLOSURE_STATE.md` plus the repository’s existing canonical issue/status/control records when applicable.
- Once a Section/Subsection is durably recorded DONE, workers MUST NOT routinely re-enter it for reimplementation, polishing, re-auditing, or repeat verification. Skip it and work on the earliest unfinished Section.
- A closed Section/Subsection may be reopened only for a demonstrated regression, invalidated closure evidence, changed acceptance contract, or a later integration change that demonstrably broke it. Record `REOPENED` and the exact reason before new work begins.
- If two workers close the same scope concurrently, keep one canonical closure lineage/evidence set. Converge any unique necessary changes, then close/supersede the duplicate PR/branch/issue; delete the duplicate branch when safe. Never count one Section twice.
- Before creating a new PR/branch for the current Section, inspect existing active lineages and reuse/repair/converge them when possible.
- If another worker closes the current Section/Subsection while you are working, refresh the live registry immediately. Do not keep mutating already-DONE scope merely because your branch or PR is still open. Preserve and converge only genuinely unique required changes; otherwise close/supersede/retarget the duplicate lineage and move to the earliest unfinished Section.
- External blockers do not justify false DONE. Finish all internally controllable work, record the blocker precisely, and proceed only to dependency-safe work.

Chat history is not closure authority. Durable GitHub state is.
