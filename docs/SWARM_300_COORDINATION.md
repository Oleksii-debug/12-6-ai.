# SWARM-300 coordination protocol v1

Status: `CANDIDATE_PROTOCOL_V1`

Canonical control issue: #723
Implementation issue: #724
Machine collision guard owner: PR #575 or its terminal successor

## Goal

Allow tens or hundreds of disposable ChatGPT workers to receive the exact same prompt and still distribute themselves across useful work without the owner manually preparing numbered prompts.

The protocol is deliberately decentralized. There is no scheduler that must assign 300 tasks in advance. Every worker reconstructs live GitHub, obtains a unique GitHub-issued worker identity, chooses a starting lane, claims one semantic work surface, verifies that it won the claim, and then works end-to-end.

## Single coordination authority

GitHub is the only ownership/claim authority.

Google Drive may contain canonical vision, long reports, research notes and backups, but it must not be used as a second lock registry. A worker may read Drive when relevant, but a Drive sentence such as "I am working on X" never reserves X unless the reservation is reflected in GitHub.

Live-truth order remains:

1. exact GitHub SHA / branch / PR / Actions;
2. active GitHub claim/control/lane issues;
3. independent audit evidence;
4. Drive canonical reports/research;
5. chat prose.

## Worker registration

Each identical-prompt worker begins by creating exactly one GitHub issue in `Oleksii-debug/12-6-ai.`.

Initial title:

`SWARM-REGISTER: autonomous worker`

The issue number assigned by GitHub is the unique worker identity:

`SWARM_WORKER_ID = SWARM-<issue_number>`

The registration issue later becomes that worker's claim and final handoff issue. Do not create a second claim issue for the same chat unless the first issue is unusable.

## Automatic diversification

Permanent lanes are GitHub issues #2 through #16.

For worker registration issue number `N`, calculate:

`preferred_lane_issue = 2 + (N mod 15)`

This spreads identical workers across the permanent lanes without requiring a different prompt for each chat.

The preferred lane is a starting point, not a prison. Read its contract and obey its ownership restrictions. If the lane has no substantial unclaimed work, is saturated, is blocked on another authority, or would create duplicate Product work, rotate through #2-#16 and current P0/P1 control issues until a substantial disjoint package is found.

Audit lanes must audit rather than silently patch Product code. Coordination lanes must coordinate rather than become the default implementation lane. Compute lanes must not infer paid-compute authorization.

## Semantic lane key

Every active worker must own one explicit semantic lane key.

Format:

`<PERMANENT-LANE>|<OBJECT-OR-SUBSYSTEM>|<WORK-KIND>`

Examples:

- `D03|COMMON-PILE|SOURCE-RIGHTS-AUDIT`
- `D02|MUON|20M-MATCHED-OPTIMIZER-ARM`
- `D07|LLAMA-CPP|EXPORT-PARITY`
- `D09|TOOL-PROTOCOL|REDTEAM`
- `R01|OLMO-LADDER|20M-100M-SCALING-FIT`
- `AUDIT-A|CHECKPOINT-346|INDEPENDENT-REQUALIFICATION`

The key must describe the real semantic ownership surface. A worker number alone is not an ownership key. Existing non-swarm issues/PRs may already own a surface even if they do not contain this key, so semantic collision review is mandatory.

## Claim body contract

After choosing a task, update the registration issue title to:

`[ACTIVE] SWARM-CLAIM <SWARM_LANE_KEY> — <short objective>`

The body must contain at least:

```text
SWARM_PROTOCOL: SWARM-300-V1
SWARM_CONTROL: #723
SWARM_WORKER_ID: SWARM-<issue_number>
SWARM_LANE_KEY: <semantic key>
PREFERRED_LANE: #<lane issue>
PARENT_ISSUES: #...
STATUS: ACTIVE
BASE_SHA: <exact main or intentional parent SHA>
CLAIMED_AT_UTC: <timestamp>
LEASE_UNTIL_UTC: <timestamp, default +6h>
OWNED_SURFACES:
- <files/modules/contracts/research surface>
AVOID_SURFACES:
- <known active neighboring surfaces>
OBJECTIVE:
<large coherent end-to-end objective>
ACCEPTANCE:
- <evidence required>
- <tests/CI required>
FALLBACK:
<what to do if the primary objective becomes terminal before edits>
```

## Two-phase collision check

A worker does not own a task merely because it wrote a claim.

### Phase A — semantic pre-claim review

Before updating the registration issue into an active claim, search live open issues, PRs, recent branches/commits and permanent lane logs for the same subsystem/objective. If an active canonical issue/PR already owns the same Product/research surface, do not create a duplicate implementation. Either:

- select a disjoint sub-surface;
- consume/verify/red-team the existing work if that is genuinely independent and allowed by the lane;
- or rotate to another task.

### Phase B — exact-key post-claim arbitration

Immediately after publishing `SWARM_LANE_KEY`, search open GitHub issues and PRs for that exact key again.

If two or more active swarm claims use the same exact semantic key, the claim with the earliest GitHub `created_at` wins. A later worker must not continue editing simply because it already thought about the task.

The losing worker updates its issue to `ABANDONED_DUPLICATE`, records the winning claim, removes/closes any unnecessary duplicate PR/branch if already created, then returns to task discovery and updates the same issue with a different unclaimed lane key.

Existing canonical ownership predating the swarm beats a new swarm claim even when an exact key is absent.

## Lease and stale recovery

Default lease: 6 hours from claim time.

Refresh the claim issue when a substantial commit/PR is published or before the lease expires. A worker may take over another claim only when all are true:

1. `LEASE_UNTIL_UTC` has passed;
2. the claim issue has no newer meaningful update extending the lease;
3. its branch/PR has no recent substantive progress;
4. no active CI/run plausibly indicates the owner is still working;
5. takeover is explicitly recorded with the old claim URL/number and evidence.

Do not steal active work merely because another chat is slower.

## Branch and PR contract

Only after winning the claim should a normal implementation worker create a branch.

Recommended branch:

`swarm/<claim-issue-number>-<short-slug>`

Every implementation PR body must repeat:

```text
SWARM_PROTOCOL: SWARM-300-V1
SWARM_CONTROL: #723
SWARM_CLAIM_ISSUE: #<issue-number>
SWARM_WORKER_ID: SWARM-<issue-number>
SWARM_LANE_KEY: <exact semantic key>
PARENT_ISSUES: #...
```

This metadata is intended to be consumed by the machine collision guard in PR #575 or a successor.

## Work selection quality

Select the largest useful coherent P0/P1 package that is safe to own. Prefer vertical completion over helper-only churn.

Good packages combine the implementation/research object with its validation, adversarial tests, documentation, machine-readable evidence and handoff where those belong to the same ownership surface.

Do not create work merely to keep a worker busy. If no safe substantial implementation exists, prefer an independent audit, verifier, red-team, research comparison, source-rights investigation, reproducibility check or integration analysis that does not duplicate active Product ownership.

If no meaningful disjoint work remains, record `NO_SAFE_UNCLAIMED_WORK` rather than manufacturing a fake task.

## Late-binding rule

Workers are launched concurrently, so the state seen at startup can become stale during execution.

Re-read live GitHub:

1. at startup;
2. immediately before claiming;
3. immediately after claiming;
4. immediately before material Product edits;
5. before opening/updating a PR;
6. before final verdict/handoff.

If a relevant sibling/upstream authority became terminal while the worker was executing, consume it and rerun affected verification once when practical. Do not return a blocker solely from a stale startup cutoff.

Do not wait for sibling workers. Work against current durable authority and late-bind available terminal evidence.

## CI discipline

Use `.github/workflows/ci.yml` as the shared broad CI surface and existing scoped scientific gates where appropriate.

Normal swarm workers must not create one-off `.github/workflows/*` files. PR #575 or its successor owns machine collision-guard implementation; do not create a second guard merely because this protocol mentions collisions.

Queued/in-progress/action-required CI is `NOT TESTED`, never PASS.

CPU evidence is not CUDA/GPU evidence. Upstream benchmark claims are not local 12-6 evidence.

## Handoff

At the end, update the claim issue with:

```text
STATUS: TERMINAL | BLOCKED | SUPERSEDED | REJECTED
BRANCH: <branch>
HEAD_SHA: <sha>
PR: #<number or none>
CI: <run IDs and exact conclusions>
CHANGED_SURFACES:
- ...
EVIDENCE:
- tests / metrics / artifacts / hashes
NOT_TESTED:
- ...
BLOCKERS:
- ...
NEXT_SAFE_ACTION:
- ...
LEASE: RELEASED
```

Also leave a concise pointer in the relevant permanent lane issue when the result materially changes that lane's state.

Do not call a task terminal merely because code was written. State exactly what is proven, what is merely prepared, and what remains blocked.

## Merge policy

A worker normally opens a PR and leaves integration to the current integration/coordination authority. It must not merge its own Product PR merely because its scoped tests are green unless the live project policy explicitly grants merge authority for that surface.

If the work becomes redundant because a better canonical PR landed, prefer closing/superseding rather than forcing a duplicate merge.

## Project hard boundaries

- canonical Base remains random-init and pretraining-only unless explicit project authority changes that policy;
- no foreign pretrained/instruct/aligned weights in canonical Base;
- no benchmark/final-test leakage into training;
- data licenses do not bypass source/provenance/privacy/decontamination gates;
- no materially paid compute or long training without explicit `COMPUTE_AUTHORIZED` and required training authority;
- no fabricated CI/GPU/data/training success;
- post-Base agent/tool work remains isolated from canonical Base lineage;
- exact GitHub evidence outranks stale reports/chat.

## Why this scales better than 300 numbered prompts

The owner only pastes one prompt. GitHub supplies each worker a unique number. That number diversifies its starting lane. Semantic claims distribute sub-work dynamically as the repository changes. Earliest-claim arbitration resolves races, leases recover abandoned work, and late-binding keeps workers from returning stale blockers.

The universal prompt therefore acts as a decentralized scheduler rather than a static list of 300 tasks.
