# Sequential Closure State

This file is the durable GitHub mirror for ordered Section/Subsection closure.

## Rules

- Read the current canonical Section plan and live default branch before updating this file.
- Record only evidence-backed DONE state; never infer closure from a chat summary, a PR existing, queued CI, or a single green test.
- Once recorded DONE, a Section/Subsection is skipped by normal workers and is not re-entered unless it is explicitly marked REOPENED for a demonstrated regression, invalidated evidence, changed acceptance contract, or broken later integration.
- If parallel workers produce duplicate closure lineages, preserve one canonical lineage, converge unique required changes, then close/supersede the duplicate and delete the duplicate branch when safe.
- Update this ledger in the same run that closes or reopens scope.

## Closure registry

| Section / Subsection | State | Canonical evidence / lineage | Accepted source / build | Notes |
| --- | --- | --- | --- | --- |
| Section 0 — Цільова архітектура 12-6 як повної AI-системи | IN_PROGRESS | PR #3074; branch `section/0-system-architecture-contract-v1`; converged from `main@03296d2d33bdeec9167649067af94ef392bdebe3` | — | Candidate architecture implementation exists; Section 0 is not DONE until accepted integration and terminal exact-head CI on the integrated candidate. |
