# Global Coordinator Bridge

12-6 AI participates in the owner's cross-project autonomous development control plane.

## Purpose

Workers, Codex Cloud and Work must not rely on stale prompt text as the current project plan. Before substantive work, reconstruct the newest project state and read the current coordination surfaces. The cross-project coordinator updates priorities; workers consume those priorities rather than requiring manual prompt rewrites from Oleksii.

## Coordination hierarchy

1. Global coordinator master: cross-project priorities for Nika Core, 12-6 AI, Accessible Chess and future projects.
2. Project-local orchestration plan: current 12-6 priorities, completed work, do-not-repeat items, active ownership and next unowned packages.
3. Worker execution: each run chooses only current unowned work in its lane and leaves durable evidence.

## Audit cadence

Do not depend on exact wall-clock slots. If the configured project audit interval has expired, the first available coordinator-capable run performs the audit and refreshes the plan. A missed run due to credits/usage does not invalidate the system; the next successful run reconstructs live state and continues.

Suggested active-project audit interval: 4–6 hours, with immediate refresh after major convergence/integration changes.

## Codex Cloud

On start, Codex reads the current project plan and avoids packages already assigned to scheduled workers. It takes a large unowned package, persists progress after coherent milestones, and refreshes recommendations when a major change invalidates worker direction. Do not wait until the last turn of a long run to leave coordination state.

## Work

Work acts as a periodic high-capability architect/auditor. It may review the full project and update the current plan after major architectural, scientific or integration changes. Oleksii should not need to copy new prompts into five scheduled workers after each Work run.

## 12-6 current product priority

The global coordinator must preserve the learned-20M critical path. Preparation, agent/living-agent research and open-source discovery must not indefinitely defer real learned-20M training, inference and the first usable Nika/12-6 dialogue surface.

## Owner-facing reporting

Oleksii is not expected to manage GitHub mechanics. Reports to him should say in plain Ukrainian: what is ready, what works, what does not work yet, what changed, current readiness, next product milestone, and whether owner approval/budget is required.
