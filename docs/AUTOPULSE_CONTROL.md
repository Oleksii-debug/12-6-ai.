# 12-6 AI AUTOPULSE CONTROL v0.4

## Live-truth order
1. GitHub exact commits / branches / PR / Actions.
2. Direct GitHub issue/PR collections, central control issue, active swarm claims and permanent lane issues.
3. Auditor evidence.
4. Drive canonical docs/research/reports.
5. Chat text only when not yet durably recorded.

## Universal pulse
Any owner message equivalent to START / CONTINUE / NEXT / AUTOPULSE triggers live recovery and work. A worker never asks what to continue if state is recoverable.

## Universal identical-prompt swarm
For large disposable-chat swarms, use `docs/SWARM_300_COORDINATION.md`, `docs/UNIVERSAL_SWARM_PROMPT.md`, `configs/swarm/swarm300_protocol_v2.json` and control issue #723.

GitHub is the only ownership/claim authority. Drive is not a lock registry.

Every identical-prompt worker performs a read-only scout before its first mutation, self-registers through one GitHub issue, derives a two-dimensional lane/work-kind routing slot from the issue number, compares multiple candidate packages, publishes one canonical semantic `SWARM_LANE_KEY`, wins direct-collection exact-key arbitration before Product edits, and leaves durable branch/SHA/PR/CI/handoff evidence in the same claim issue.

An older canonical active issue/PR that semantically owns a surface beats a later swarm claim even if it lacks swarm metadata. GitHub Search is discovery-only for exact races; direct paginated open-issue state is used for exact-key arbitration. Exact-key races resolve by earliest GitHub `created_at`, then lowest issue number.

Normal swarm work must satisfy the large-package gate: primary work + validation + evidence/closure and at least four package dimensions. Micro-lint/config/docs/status tasks are not standalone swarm packages when useful adjacent same-surface work exists.

GitHub API/Actions capacity is a real scale boundary. Workers obey rate-limit signals, never hammer failed writes, and use CI pressure backpressure so development concurrency does not imply one remote CI run per worker. PR #575 or its terminal successor remains the machine collision-guard owner and is not assumed deployed merely because the protocol exists.

## Fail-open development, fail-closed promotion
Ordinary disjoint development continues even if Coordinator/Auditor is temporarily stale. Stage promotion, canonical release, dataset freeze and paid-compute launch do not proceed without their required evidence/gates.

## Training authorization
LOCAL_FREE: tiny/local/smoke work may proceed.
PREPARED_NOT_LAUNCHED: paid run is fully configured but not launched.
COMPUTE_AUTHORIZED: owner/budget authorization exists.
RUNNING: exact SHA/config/manifest recorded.
COMPLETED: artifacts and metrics recorded.
AUDITED: independent evidence accepted.

## Durable checkpoint
Every worker records branch, SHA, PR if any, tests, status, blockers, ownership and next action in its permanent GitHub lane issue or, for universal swarm workers, in its dedicated claim issue with a concise pointer into the permanent lane. Large reports may additionally go to Drive.

## Collision rule
One Product surface has one active owner unless explicitly coordinated. Safe overlap is encouraged only in clearly disjoint modules/tests/research or intentionally independent audits. Never overwrite another lane's canonical implementation just to keep busy.

Before material edits, a swarm worker performs semantic collision review and direct-collection exact-key arbitration. Losing a claim means rotate to another task rather than proceeding in parallel.

## CI discipline
Use shared repository CI for normal swarm work. Do not create one-off Actions workflows per disposable worker. Queued, running or action-required checks are NOT TESTED, never PASS.

CI pressure for universal swarm work is measured from queued + in-progress Actions: GREEN <=25, AMBER 26..100, RED >=101. RED routine work prefers strong local validation plus branch/SHA/issue handoff rather than adding remote fanout; critical terminal unblock/integrity/integration evidence may still justify a remote run.

## Scale rule
S0-S3 prioritize proving the reusable factory. S4+ increasingly prioritize data quality, throughput, distributed training and evaluation. Post-training/reasoning infrastructure may be built early, but must not alter the clean Base lineage until the project explicitly enables it.
