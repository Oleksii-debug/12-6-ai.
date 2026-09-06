# SWARM-300 coordination protocol v2

Status: `READY_V2_FOR_200_WORKER_TRIAL`

Canonical control issue: #723
Scale-audit issue: #732
Machine collision guard owner: PR #575 or its terminal successor; it is not assumed deployed until merged/terminal on the active integration line.

## Owner launch entrypoint

For repetitive manual launching, the preferred copy-paste entrypoint is `docs/SWARM_BOOTSTRAP_PROMPT.md` rather than copying this entire protocol or the full worker prompt into every chat.

The bootstrap is intentionally short. Each new chat must read the current canonical `docs/UNIVERSAL_SWARM_PROMPT.md`, this coordination protocol, the machine-readable swarm protocol and issue #723 from live `main` before doing work. If those authorities conflict or cannot be read, the bootstrap forbids GitHub mutations and fails closed.

This indirection means future READY swarm protocol upgrades can be picked up by newly launched chats without the owner manually replacing a long prompt in every launch.

## Purpose

Support identical-prompt autonomous development at 200-worker scale and remain structurally safe for later 500–1000-worker experiments without pretending GitHub itself has unlimited concurrent write or CI capacity.

The system separates four concerns:

1. **read-only discovery concurrency** — many workers can reconstruct live state in parallel;
2. **ownership claims** — GitHub issues remain the single durable claim authority;
3. **development concurrency** — workers may advance disjoint branches/packages in parallel;
4. **validation concurrency** — remote PR/Actions fanout is pressure-controlled instead of scaling one-for-one with worker count.

This distinction is mandatory. `1000 chats` must never be interpreted as `1000 simultaneous GitHub mutations plus 1000 CI runs`.

## Truth and ownership

Truth order:
1. exact Git SHA / branch / PR / Actions;
2. direct GitHub issue/PR collections and claim/control/lane state;
3. independent audit evidence;
4. Drive reports/research/backups;
5. chat prose.

GitHub is the sole ownership registry. Drive is not a lock service.

## Startup read-first rule

Every worker performs a minimum read-only scout before its first GitHub mutation. It reads project rules, #723, current main, open P0/P1 surfaces and relevant lane state. Only then does it create one registration issue.

This reduces stale claims and naturally spreads mutative traffic across time.

## Worker identity and routing

The registration issue number `N` becomes `SWARM-N`.

Coarse routing uses two dimensions:

`preferred_lane = 2 + (N mod 15)`

and

`preferred_work_kind = WORK_KINDS[(N div 15) mod 8]`

with eight work kinds:
- PRODUCT_VERTICAL
- INDEPENDENT_VERIFY
- REDTEAM_AUDIT
- INTEGRATION_CONVERGENCE
- PERFORMANCE_RUNTIME
- DATA_SOURCE_OR_PIPELINE
- OPEN_SOURCE_REUSE_RESEARCH
- REPRODUCIBILITY_RELEASE

There are therefore 120 deterministic coarse routing slots before repetition. The slot is a starting bias, not ownership permission.

## Candidate selection

Before claim, a worker should compare at least three materially different candidates when possible and prefer the package with the best combination of:
- P0/P1 blocker impact;
- end-to-end closability;
- LOCAL_FREE feasibility;
- low collision risk;
- independent validation value;
- manageable CI cost.

Do not simply choose the globally most obvious issue; identical workers must be diversified by lane/work-kind and live ownership.

## Large-package gate

A normal claim must contain at least four of:
- implementation or primary research;
- focused tests;
- adversarial/negative tests;
- machine-readable evidence/validator;
- docs/operator handoff;
- live authority binding;
- end-to-end/integration proof;
- measured benchmark/reproducibility proof.

It must include primary work, real validation, and an evidence/closure dimension.

A one-line lint fix, one config field, docs-only restatement, tiny helper, duplicate validator/readiness gate, status-only PR or cosmetic refactor is not a standalone swarm task when useful adjacent same-surface work exists.

If a valid package shrinks to a micro-fix, the worker continues into safe adjacent work inside the same ownership surface rather than terminating early.

## Semantic ownership key

Canonical format:

`<LANE>|<CANONICAL-OBJECT>|<WORK-KIND>|<QUALIFIER>`

All fields use uppercase ASCII hyphenated terms. Known project identifiers are reused exactly rather than renamed with synonyms.

Examples:
- `D03|COMMON-PILE|SOURCE-RIGHTS-AUDIT|V1`
- `D02|MODEL-341|OPTIMIZER-EXPERIMENT|MUON-VS-ADAMW`
- `D05|MODEL-341|CHECKPOINT-REQUALIFICATION|CORRUPTION-MATRIX`
- `R01|OLMO-LADDER|SCALING-FIT|20M-100M`

Semantic review against existing non-swarm ownership remains mandatory.

## Exact race arbitration

GitHub Search is allowed for discovery but is not the exact lock mechanism.

After publishing a claim, a worker reads the direct paginated OPEN issue collection and extracts all active claim bodies carrying the same exact `SWARM_LANE_KEY`.

Winner:
1. earliest GitHub `created_at`;
2. lowest issue number if timestamps tie.

A losing worker changes the same registration issue to `ABANDONED_DUPLICATE`, records the winner, selects another package and reclaims. It does not create a new worker issue.

Existing canonical semantic ownership beats a later swarm claim even without an exact key.

## Rate-limit behavior

GitHub API capacity is an external constraint. Workers must not hammer writes or create duplicate fallbacks after 403/429.

On rate limiting, honor `Retry-After` or rate-reset information when available. If durable mutation cannot safely resume, the worker owns no new Product surface and exits read-only as `RATE_LIMIT_DEFERRED_READ_ONLY`.

This status is intentionally non-owning and does not block another worker.

## Claim lease

Default lease is six hours. Takeover requires all of:
- expiry;
- no newer meaningful issue update;
- no recent substantive branch/PR progress;
- no active CI plausibly owned by the claimant;
- explicit takeover evidence.

## Branch discipline

One active branch per claim:

`swarm/<claim-issue-number>-<slug>`

Branch only after exact claim arbitration is won. Prefer substantial bundled commits over repeated trivial GitHub mutations.

## CI pressure control

Development concurrency and validation concurrency are separate.

Before opening a PR, workers inspect lightweight queued + in-progress Actions counts and classify:

- GREEN <= 25
- AMBER 26..100
- RED >= 101

GREEN: normal one-PR-per-claim behavior.

AMBER: PR only for substantial P0/P1 work or when exact-head remote evidence is materially needed.

RED: routine work should stop adding PR/CI fanout. Prefer strong local validation followed by durable branch/SHA + claim-issue handoff. A new PR/remote run is reserved for critical terminal unblock, security/integrity, integration or similarly high-value authority.

No swarm worker creates a temporary Actions workflow. Shared `ci.yml` and existing scoped gates remain the validation surfaces.

Queued/running/action-required never equals PASS.

## PR metadata

Every swarm PR contains:

```text
SWARM_PROTOCOL: SWARM-300-V2
SWARM_CONTROL: #723
SWARM_CLAIM_ISSUE: #<issue-number>
SWARM_WORKER_ID: SWARM-<issue-number>
SWARM_LANE_KEY: <canonical key>
PARENT_ISSUES: #...
PACKAGE_DIMENSIONS: <dimensions>
```

One PR maximum per claim.

## Late binding

Workers refresh live authority at startup, before registration/claim, after claim, before material edits, before PR decisions and before final handoff.

A blocker from startup must be reconsidered if a sibling/upstream authority terminalizes during execution.

If a better canonical implementation lands, converge/adapt/verify/supersede rather than force duplicate output.

## Final statuses

Allowed terminal handoff states include:
- TERMINAL
- BLOCKED
- SUPERSEDED
- REJECTED
- NO_SAFE_UNCLAIMED_WORK
- RATE_LIMIT_DEFERRED_READ_ONLY

Every handoff records lane key, package dimensions, branch, exact SHA, PR if any, CI pressure, CI results, changed surfaces, evidence, untested items, blockers, next safe action and released lease.

## Project hard boundaries

Canonical Base remains random-init and pretraining-only until explicit authority changes it. No foreign pretrained weights enter canonical Base. No benchmark/final-test leakage. Data rights remain source/provenance-specific. No materially paid compute without explicit authorization. CPU evidence is not GPU evidence. Post-Base work does not mutate canonical Base lineage. No fabricated terminality.

## Scale interpretation

The protocol is designed so the owner can paste one prompt repeatedly. It is expected to improve throughput by reducing manual assignment and duplicate work, but it is not a promise of linear speedup. GitHub writes, active Product surfaces, integration bandwidth and CI runners become bottlenecks as worker count rises.

A 200-worker trial is therefore an empirical scale test. Any move to 500–1000 workers should use measured claim collision rate, rate-limit events, useful-package yield, PR/CI pressure and integration throughput from the 200-worker run rather than assuming 5x more chats produces 5x more useful development.
