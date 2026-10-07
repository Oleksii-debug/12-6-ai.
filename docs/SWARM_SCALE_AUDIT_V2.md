# SWARM scale audit V2

Issue: #732
Parent control: #723

## Verdict before V2

SWARM-300-V1 was directionally sound but was not strong enough to call safe for an immediate 500–1000-worker launch.

The audit found four material scale hazards:

1. exact-key arbitration relied on GitHub Search, which is a discovery surface rather than an atomic lock and can lag newly-created content;
2. the referenced machine collision guard (#575) was still an open PR and therefore could not be treated as deployed protection;
3. 15 lane modulo routing was insufficient diversification for hundreds of workers;
4. one-worker/one-PR/one-CI scaling could recreate the repository's previous runner saturation even if Product ownership collisions were reduced.

A fifth external constraint is GitHub API secondary limiting. GitHub's current public REST documentation states that secondary limits can include no more than 100 concurrent REST/GraphQL requests and, in general, no more than 80 content-generating requests per minute or 500 per hour. GitHub's best-practices documentation recommends avoiding concurrent API requests, pausing between mutative requests, and honoring Retry-After / rate-reset signals. These limits are external and may change.

Official references:
- https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api
- https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api

## V2 repairs

### Read-first startup

Workers reconstruct the repository before their registration write. This reduces stale claims and naturally disperses the write burst.

### 120 coarse routing slots

V2 combines 15 permanent lanes with eight work kinds. Any 120 consecutive integer worker IDs map one-to-one across all 120 coarse slots. The checked-in simulation tests also exercise 200 and 1000 sequential IDs and require the bucket counts to differ by at most one.

This is diversification, not a proof of unique semantic tasks. Workers still perform live collision review.

### Large vertical gate

Normal packages need at least four dimensions and must contain primary work, validation and evidence/closure. Micro-lint/config/docs/status tasks are explicitly rejected as standalone swarm packages when adjacent same-surface work is available.

### Direct collection exact arbitration

GitHub Search remains useful for semantic discovery. Exact claim arbitration instead reads the direct paginated OPEN issue collection and compares `SWARM_LANE_KEY` values from current claim bodies.

Earliest `created_at` wins; lower issue number breaks a timestamp tie.

This is stronger than search-index arbitration but is still an application-level distributed claim protocol, not a transactional database lock. The system therefore retains semantic pre-claim review, post-claim arbitration, leases and duplicate-abandonment behavior.

### Rate-limit fail-safe

403/429 is not interpreted as claim success. Workers do not create substitute duplicate issues/PRs and do not hammer retries. If durable mutation cannot resume safely, the result is `RATE_LIMIT_DEFERRED_READ_ONLY` and owns no Product surface.

### Development versus validation concurrency

V2 explicitly separates worker count from Actions count.

Before PR creation workers read lightweight queued and in-progress run counts:
- GREEN <= 25
- AMBER 26..100
- RED >= 101

GREEN permits normal one-PR-per-claim behavior. AMBER limits PRs to substantial P0/P1/evidence needs. RED prefers branch/SHA/issue handoff and reserves new CI fanout for critical terminal unblock, integrity/security or integration authority.

At the V2 implementation cutoff, live Actions API returned 5 queued and 9 in-progress runs, or 14 total, which is GREEN. This is a point-in-time observation, not a future guarantee.

### Machine guard truthfulness

PR #575 or its terminal successor remains owner of machine collision-guard code. V2 does not claim this guard is merged or active and does not duplicate it.

## What V2 can and cannot guarantee

V2 can deterministically validate routing distribution, lane-key normalization, exact-key winner selection, large-package admission rules and CI-pressure classification in local tests.

V2 cannot guarantee that 200 independent chats will produce 200 useful mergeable packages. Useful concurrency is bounded by the number of genuinely disjoint valuable surfaces, GitHub write capacity, CI capacity, integration bandwidth and the quality of each worker's live semantic collision analysis.

It also cannot make GitHub's API rate limits disappear. A 500–1000-chat launch must therefore not be modeled as 500–1000 simultaneous mutative API clients.

## Recommended empirical rollout

The first 200-chat wave should be treated as a scale experiment and measured on:
- registration/claim success rate;
- exact and semantic collision rate;
- percentage of workers producing a large-package-qualified result;
- RATE_LIMIT_DEFERRED_READ_ONLY count;
- NO_SAFE_UNCLAIMED_WORK count;
- PR count versus branch-only handoffs;
- queued/in-progress Actions peak;
- terminal-green package count;
- superseded/duplicate PR count;
- integration throughput and useful merged output.

Do not infer a 500–1000-worker optimum until these measurements exist.

The expected speedup comes from eliminating manual task authoring and allowing many disjoint verticals to advance concurrently. It is not expected to remain linear as worker count rises.
