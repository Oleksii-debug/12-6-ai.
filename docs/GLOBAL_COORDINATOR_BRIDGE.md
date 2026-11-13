# Global Coordinator Bridge

12-6 AI uses the shared cross-project coordination process.

## Purpose

Before substantive work, verify the current project state and read the coordination records. The cross-project coordinator updates the shared priorities, which guide scheduled work and integration.

## Coordination hierarchy

1. Global coordinator master: cross-project priorities for Nika Core, 12-6 AI, Accessible Chess and future projects.
2. Project-local orchestration plan: current 12-6 priorities, completed work, do-not-repeat items, active ownership and next unowned packages.
3. Worker execution: each run chooses only current unowned work in its lane and leaves durable evidence.

## Audit cadence

Do not depend on exact wall-clock slots. If the configured project audit interval has expired, the first available coordinator-capable run performs the audit and refreshes the plan. After a missed run, the next available coordinator reconstructs the current state and resumes the audit.

Suggested active-project audit interval: 4–6 hours, with immediate refresh after major convergence/integration changes.

## Extended integration

Review the current plan, select a large unowned package and avoid overlapping scheduled assignments. Record progress after each coherent milestone. Update the shared priorities when a major change affects scheduled work.

## Periodic technical audit

Review the full project after major architectural, scientific or integration changes and record updated priorities in the shared plan.

## 12-6 current product priority

The global coordinator must preserve the learned-20M critical path. Preparation, agent/living-agent research and open-source discovery must not indefinitely defer real learned-20M training, inference and the first usable Nika/12-6 dialogue surface.

## Owner-facing reporting

Oleksii is not expected to manage GitHub mechanics. Reports to him should say in plain Ukrainian: what is ready, what works, what does not work yet, what changed, current readiness, next product milestone, and whether owner approval/budget is required.
