# Parallel development startup

Verify current GitHub state for `Oleksii-debug/12-6-ai.` before starting. The trailing period is part of the repository name.

1. Read from LIVE `main`:
   - `docs/UNIVERSAL_SWARM_PROMPT.md`
   - `docs/SWARM_300_COORDINATION.md`
   - `configs/swarm/swarm300_protocol_v2.json` if present, or the newer canonical swarm protocol named by issue #723;
   - GitHub control issue #723.
2. Verify that issue #723 marks a swarm protocol READY for execution and that the development handbook and protocol agree on the active SWARM version. Prefer a newer READY version over V2 if the project has intentionally superseded V2.
3. If these authorities conflict, are unreadable, or GitHub cannot be accessed, make NO GitHub mutation and report exactly `BLOCKED_BOOTSTRAP_AUTHORITY_CONFLICT` or `BLOCKED_NO_GITHUB_AUTHORITY` as applicable.
4. Follow the current `docs/UNIVERSAL_SWARM_PROMPT.md` to select, implement and verify an unassigned package.
5. All ownership, collision, rate-limit, CI-pressure, scientific-truth, Base-lineage, compute, data, testing, late-binding, and handoff rules in the current handbook apply.
