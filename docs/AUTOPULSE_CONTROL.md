# 12-6 AI AUTOPULSE CONTROL v0.3

## Live-truth order
1. GitHub exact commits / branches / PR / Actions.
2. Central control issue, active swarm claims and permanent lane issues.
3. Auditor evidence.
4. Drive canonical docs/research/reports.
5. Chat text only when not yet durably recorded.

## Universal pulse
Any owner message equivalent to START / CONTINUE / NEXT / AUTOPULSE triggers live recovery and work. A worker never asks what to continue if state is recoverable.

## Universal identical-prompt swarm
For large disposable-chat swarms, use `docs/SWARM_300_COORDINATION.md`, `docs/UNIVERSAL_SWARM_PROMPT.md` and control issue #723.

GitHub is the only ownership/claim authority. Drive is not a lock registry.

Every identical-prompt worker self-registers through one GitHub issue, derives its worker ID from the GitHub issue number, uses that number to diversify its preferred permanent lane, publishes one semantic `SWARM_LANE_KEY`, wins the post-claim collision check before Product edits, and leaves its branch/SHA/PR/CI/handoff in the same claim issue.

An older canonical active issue/PR that semantically owns a surface beats a later swarm claim even if it lacks swarm metadata. Exact-key swarm races are resolved by earliest GitHub claim `created_at`. Stale takeover requires an expired lease plus proof of no recent issue/branch/PR/CI progress.

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
Every worker records branch, SHA, PR, tests, status, blockers, ownership and next action in its permanent GitHub lane issue or, for SWARM-300 workers, in its dedicated claim issue with a concise pointer into the permanent lane. Large reports may additionally go to Drive.

## Collision rule
One Product surface has one active owner unless explicitly coordinated. Safe-overlap is encouraged in different files/modules/tests/research. Never overwrite another lane's canonical implementation just to keep busy.

Before material edits, a swarm worker performs both semantic collision review and exact-key post-claim arbitration. Losing a claim means rotate to another task rather than proceeding in parallel.

## CI discipline
Use shared repository CI for normal swarm work. Do not create one-off Actions workflows per disposable worker. Queued, running or action-required checks are NOT TESTED, never PASS.

## Scale rule
S0-S3 prioritize proving the reusable factory. S4+ increasingly prioritize data quality, throughput, distributed training and evaluation. Post-training/reasoning infrastructure may be built early, but must not alter the clean Base lineage until the project explicitly enables it.
