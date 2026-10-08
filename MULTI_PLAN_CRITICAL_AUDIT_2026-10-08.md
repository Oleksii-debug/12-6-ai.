# 12-6 AI — Critical Multi-Plan Audit — 2026-10-08

## Verdict

Plan count remains exactly 10.

- Plans 1–8: independent engineering plans, no priority order.
- Plan 9: per-Section campaign/champion convergence.
- Plan 10: final whole-product integration/release only.

The old 96-Section monolith remains audit-only.

## Completeness

- Legacy Sections reviewed: 96/96.
- Unmapped legacy Sections: 0.
- Detailed mapping: `LEGACY_96_TO_MULTIPLAN_COVERAGE.md`.

## Corrections made during this audit

1. Removed hidden Plan-1 contract dependency.
   - Migration Contract Baseline v1 is already provided by migrated DONE architecture/identity work.
   - Plans 2–8 do not wait for Plan 1 Sections 3–8.

2. Removed production-data dependency from Plan 3.
   - Training runtime qualifies on exposure/tokenizer/shard contract fixtures.
   - Production data binds in Plan 9.

3. Made Plan 4 independent of real tokenizer/model/champion.
   - Evaluation/inference/serving qualify on fixture packages.

4. Clarified Plan 5 boundary.
   - It owns 12-6 agent-facing contracts/reference runtime, not Nika Core product implementation.

5. Preserved clean Base lineage.
   - Canonical Base remains random-init and does not descend from foreign pretrained/instruct/aligned weights.
   - Post-training/self-learning paths produce descendants.

6. Updated scaling critical path.
   - learned-20M remains mandatory terminal proof.
   - ~200M is the first serious product-scale target under the newer accelerated strategy.
   - 35M/50M/100M are optional risk-reduction probes, not mandatory serial campaigns.
   - 300–500M is an optional bridge before ~1B.

7. Made Plan 9 dependencies per-Section rather than blanket upstream waiting.
   - Base campaign needs Plans 2–4 plus applicable evidence.
   - Post-Base stages add Plan 6.
   - Scaling stages add Plan 7 only when activated.

8. Reduced Plan-10 critical-path engineering.
   - Reusable packaging/signing/update belongs to Plan 1.
   - Agent/reference runtime belongs to Plan 5.
   - Qualification/operator/physical harness belongs to Plan 8.
   - Plan 10 performs final binding, scenarios and release evidence.
   - A large missing component reopens its owning engineering plan.

9. Resolved capability-map ownership collision.
   - Plan 8 owns executable acceptance/readiness capability map.
   - Plan 1 owns only static backend/environment compatibility matrix.

10. Added repository mutation ownership/conflict keys.
   - `MULTI_PLAN_PARALLELISM_CONTRACT.md` defines default mutation surfaces and shared-file rules.

11. Made live GitHub status authoritative.
   - `MULTI_PLAN_CLOSURE_STATE.md` overrides Drive status snapshots.

12. Retained compute authority.
   - LOCAL_FREE/tiny/proxy work may run under policy.
   - Materially paid compute requires explicit authorization or an approved budget policy.

## Parallelism assessment

The architecture now exposes eight independent engineering fronts.
That makes a 5–7x wall-clock improvement plausible only when:
- enough workers are available;
- the workload is reasonably balanced across fronts;
- workers obey mutation ownership/conflict keys;
- CI and merge throughput do not become the bottleneck;
- shared contract changes remain rare;
- Plan 9/10 integration is not started prematurely.

It is not a guaranteed 5–7x speedup: final wall-clock gain remains bounded by the slowest independent plan, shared CI/GitHub throughput, real data acquisition, training compute, and unavoidable convergence gates.

The plan architecture itself no longer serializes Plans 1–8.
