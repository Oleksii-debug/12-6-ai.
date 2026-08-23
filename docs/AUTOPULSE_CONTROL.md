# 12-6 AI AUTOPULSE CONTROL v0.2

## Live-truth order
1. GitHub exact commits / branches / PR / Actions.
2. Central control issue and lane issues.
3. Auditor evidence.
4. Drive canonical docs/research/reports.
5. Chat text only when not yet durably recorded.

## Universal pulse
Any owner message equivalent to START / CONTINUE / NEXT / AUTOPULSE triggers live recovery and work. A worker never asks what to continue if state is recoverable.

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
Every worker records branch, SHA, PR, tests, status, blockers, ownership and next action in its permanent GitHub lane issue. Large reports may additionally go to Drive.

## Collision rule
One Product surface has one active owner unless explicitly coordinated. Safe-overlap is encouraged in different files/modules/tests/research. Never overwrite another lane's canonical implementation just to keep busy.

## Scale rule
S0-S3 prioritize proving the reusable factory. S4+ increasingly prioritize data quality, throughput, distributed training and evaluation. Post-training/reasoning infrastructure may be built early, but must not alter the clean Base lineage until the project explicitly enables it.
