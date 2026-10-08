# 12-6 AI — Multi-Plan Closure State

This file is the durable GitHub coordination mirror for the new independent-plan architecture.

## Binding rules

- `PROJECT_PLAN_INDEX.md`, `MULTI_PLAN_PARALLELISM_CONTRACT.md`, this file, and the numbered Drive plan assigned by the owner are the work-selection authority.
- Plans 1–8 are independent. There is no "earliest global Section" and no requirement to finish Plan 1 before Plan 5, etc.
- This GitHub file is the live status authority. Drive `Статус` lines are migration snapshots only.
- Inside one assigned plan, skip terminal DONE Sections and take that plan's first unfinished Section.
- A worker may not silently expand into another plan. Cross-plan interfaces use frozen contracts/fixtures/mocks unless the owner assigns that plan too.
- Plan 9 is a convergence/training-campaign plan and consumes real upstream terminal artifacts.
- Plan 10 is final whole-product integration/release.
- DONE remains terminal under Simplified Section Closure Protocol semantics: reopen only for demonstrated regression, invalid evidence, materially changed acceptance contract, or breaking integration.
- Legacy monolithic 96-Section numbering is audit-only and must not determine work order.

## Migration status

| New plan | New Section | State | Migrated source/evidence |
| --- | ---: | --- | --- |
| Plan 1 | 1 | DONE | Former Section 0; PR #3074 and accepted closure evidence |
| Plan 1 | 2 | DONE | Former Section 1; PR #3090 and accepted closure evidence |
| Plan 8 | 1 | DONE | Former Section 2; PR #3093 and accepted closure evidence |
| Plan 8 | 2 | DONE | Former Section 3; PR #3079 / #3094 and accepted SIL evidence |
| Plan 8 | 3 | QUALIFYING | Former Section 4; canonical finisher PR #3080; terminal qualification/integration/readback still required |

## Initial plan fronts

| Plan | First unfinished Section at migration | Coordination state |
| ---: | ---: | --- |
| 1 | 3 | OPEN |
| 2 | 1 | OPEN |
| 3 | 1 | OPEN |
| 4 | 1 | OPEN |
| 5 | 1 | OPEN |
| 6 | 1 | OPEN |
| 7 | 1 | OPEN |
| 8 | 3 | QUALIFYING |
| 9 | 1 | WAITING_UPSTREAM |
| 10 | 1 | WAITING_UPSTREAM |

Update this file when a new-plan Section reaches terminal DONE/REOPENED or when the first unfinished front for a plan changes.
