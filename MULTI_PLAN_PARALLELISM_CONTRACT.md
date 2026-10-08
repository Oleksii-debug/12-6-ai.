# 12-6 AI — Multi-Plan Parallelism Contract

## Purpose

This file makes the ten-plan architecture safely parallel in the repository.
It is coordination authority, not an additional project plan.

## Migration Contract Baseline v1

Plans 1–8 do not wait for Plan 1 to finish.

The cross-plan baseline already exists from migrated accepted work:
- former Section 0 -> Plan 1 / Section 1 DONE: typed architecture / execution-plane boundaries;
- former Section 1 -> Plan 1 / Section 2 DONE: canonical artifact/run/model/dataset/tokenizer/checkpoint/eval identities, manifests and fail-closed cross-binding.

Other plans may use frozen fixtures/mocks that conform to this baseline and may reach terminal DONE before the concrete producer plan is complete.
When the real peer artifact arrives, compatibility verification is required, but a compatible real artifact does not reopen a terminal plan.

A breaking contract change requires:
1. a new contract/schema version;
2. compatibility/migration evidence;
3. explicit impact analysis naming affected terminal plans;
4. reopening only when actual incompatibility/regression is demonstrated.

## Parallel ownership surfaces

These are default mutation ownership boundaries. They are not claims on every future filename; semantic ownership wins.

### Plan 1 — architecture / contracts / supply chain
Primary surfaces:
- core architecture and artifact identity contracts;
- dependency/environment locks and reusable release infrastructure;
- SBOM/license/supply-chain policy;
- shared contract packaging and capability/support schemas.

Existing examples:
- `src/twelve_six/system_architecture.py`
- `src/twelve_six/artifact_identity.py`
- dependency/build metadata when the change is Plan-1-owned.

Conflict key: `core-contracts`.

### Plan 2 — data / tokenizer / packing
Primary surfaces:
- `src/twelve_six/data/**`
- `src/twelve_six/tokenization/**`
- `src/twelve_six/packing/**`
- data-source/provenance/dedup/privacy/split/tokenizer/packing tooling;
- `configs/data/**` and data-owned manifests.

Conflict keys: `data-lineage`, `tokenizer`, `packing`.

### Plan 3 — model kernel / training / checkpoint runtime
Primary surfaces:
- `src/twelve_six/model.py`
- `src/twelve_six/training/**`
- `src/twelve_six/checkpoint/**`
- portable run packet/binding implementation;
- generic optimizer/scheduler/run-state/checkpoint/resume code.

Conflict keys: `model-kernel`, `training-runtime`, `checkpoint-runtime`.

### Plan 4 — evaluation / inference / serving
Primary surfaces:
- evaluation framework and `configs/evaluation/**`;
- inference/export/quantization/service/ModelGateway/multi-model-serving modules;
- backend-parity evidence specific to inference/serving.

Conflict keys: `evaluation`, `inference-serving`, `model-gateway`.

### Plan 5 — agent-facing contracts / reference runtime
Primary surfaces:
- agent/task/context/memory/tool/world/self-model reference runtime;
- browser/computer/multimodal tool contracts and reference adapters;
- agent-runtime tests/fixtures.

This plan must not implement Nika Core product-specific identity, voice UI or application state.

Conflict keys: `agent-runtime`, `tool-contracts`.

### Plan 6 — post-Base learning / research / evolution
Primary surfaces:
- instruction/preference/reasoning/agentic post-training engines;
- teacher/data-generation/self-play/continual-learning research;
- candidate/descendant/evolution/research-loop modules and tests.

It must preserve clean canonical Base lineage.

Conflict keys: `post-base-learning`, `evolution-engine`.

### Plan 7 — scaling / distributed infrastructure
Primary surfaces:
- `src/twelve_six/accelerated_scaling.py`;
- scaling/growth/distributed/topology/resource-admission modules;
- distributed checkpoint transport integration and scale-proxy tests.

Conflict keys: `scaling`, `distributed-runtime`.

### Plan 8 — qualification / evidence / autonomous repair
Primary surfaces:
- `src/twelve_six/ai_qa_control.py`
- `src/twelve_six/sil_qualification.py`
- Windows qualification/operator tooling such as `windows_operator_*.py`;
- evidence, scenario, fault-injection, physical-agent and qualification infrastructure.

Conflict keys: `qualification`, `evidence-plane`, `physical-test-fabric`.

### Plan 9 — real campaigns / champion lifecycle
Primary surfaces:
- campaign/run manifests and exact launch/readiness bindings;
- run/eval/promotion receipts and champion lineage;
- campaign-specific configs/results/evidence.

Plan 9 should consume generic engines from Plans 2–7. If a generic engine defect is found, repair it in its owning plan (or a narrowly coordinated owner-approved integration PR) rather than forking a second implementation inside Plan 9.

Conflict keys: `campaign-authority`, `champion-lineage`.

### Plan 10 — whole-product integration / release
Primary surfaces:
- final integration bindings, packaging assembly, release candidate manifests;
- whole-product scenario results, HIL/NVDA/cloud/release evidence;
- final cards/docs/support/release records.

Plan 10 is glue + final qualification. A substantial missing component reopens the owning engineering plan instead of becoming a new Plan-10 subsystem.

Conflict keys: `whole-product-integration`, `release-candidate`.

## Shared integration files

Files such as root `pyproject.toml`, `src/twelve_six/__init__.py`, broad CI workflows, root registries, shared schema files and global docs may be touched by multiple plans.

Rules:
- minimize shared-file edits;
- name the affected conflict key in the PR/commit handoff;
- refresh main before integration;
- do not run competing broad rewrites of the same shared file;
- prefer additive versioned adapters over breaking edits;
- if two active plans require the same shared mutation, converge one minimal integration change and let both consume it.

## Independence test for Plans 1–8

A plan is genuinely independent only if all of the following are true:
1. it can build/test with Migration Contract Baseline v1 and local fixtures for peer outputs;
2. it has no requirement for a real peer implementation to earn component-level DONE;
3. it does not claim cross-plan or whole-product evidence from fixtures;
4. it owns its mutable implementation surfaces or uses a coordinated shared-contract change;
5. compatible arrival of a peer implementation does not require reopening it.

If any Section violates these rules, repair the plan/contract instead of serializing all workers.

## Campaign/release dependency rule

Plan 9 and Plan 10 use per-Section dependency gates, not a blanket "all previous plans first" rule.

- Plan 9 learned-20M base campaign needs the real data/tokenizer, training runtime, evaluation/inference path and applicable qualification evidence.
- Plan 9 post-Base stages additionally need the relevant Plan-6 engine.
- Plan 9 scaling stages need Plan-7 capability only when that scale stage is activated.
- Plan 10 consumes terminal/accepted artifacts required by each final journey.

## Compute authority

LOCAL_FREE/tiny/smoke/proxy qualification may run under repository policy.
Materially paid cloud/GPU training requires explicit `TRAINING_AUTHORIZED` / `COMPUTE_AUTHORIZED` or an already approved budget policy.
A generic owner message such as "continue" does not grant paid-compute authority.
