# NEXT100-094 Objective End-to-End Research Fixture

Worker: `NEXT100-094-ENDTOEND-RESEARCH-FIXTURE`

This is a LOCAL_FREE integration fixture, not a capability or quality claim. It changes no Base weights and uses no external LLM.

## Objective task

A sealed synthetic inventory ledger contains 7 crates with 6 usable units per crate before damage. The current verified damage record says 5 units are damaged. An older `damaged_units=3` record is explicitly superseded. The objectively checkable answer is therefore `7 * 6 - 5 = 37`.

The initial high-scoring hypothesis deliberately claims 39. Retrieval must exclude the superseded damage record, a deterministic local tool must calculate 37, the stale hypothesis must receive hard contradictory evidence and be rejected, and an immutable revision must claim 37.

## Terminal components pinned into the integration tree

- POSTBASE-351 model adapter head `e805bf715617999209aef88946cea01f3668f583`; adapter blob `e5fa340a1ff1d57c46b9304e3b2dfbc0c1d517d6`. The learned 10M adapter call is plumbing-only and is never answer authority.
- POSTBASE-255 deliberation head `486bd91ca03bed41750c638d702f557f320b780a`; source blob `b3d787f14f70ab42890024297ca6a0bd2738d0e2`.
- POSTBASE-256 hypothesis search head `ea1d8fff0d3235660dffe7ba411e192df83f5e1d`; source blob `3a26aa1385dc52c7f7dfeba3f78ff327c3fddf24`.
- POSTBASE-357 verifier convergence head `7eac24e250c0853745208bab8ba9b2d3d104fbf5`; production verifier blob `e3c0504c1a6b3768c8aaea1aaaa3b3eb637eaab7`.
- POSTBASE-358 memory/RAG convergence head `976adda1cfe981d7b6363d267854759bee802006`; exact memory/RAG blobs are pinned by the workflow.

POSTBASE-254 and POSTBASE-355 tool-protocol branches were not terminal at fixture construction time because their exact heads had no successful terminal component gate. They are not silently promoted. Deterministic tool execution here uses the terminal POSTBASE-255 synchronous tool-executor seam with a project-owned arithmetic tool.

Training/data-production components such as SFT runner and teacher factory are not runtime dependencies of this research inference fixture and are not invoked.

## Verifier policy

The final claim is checked by the terminal deterministic verifier ensemble with all six deterministic dimensions present: exact answer, unit-test evidence, numeric calculation, provenance, internal consistency, and cross-candidate contradiction checking. The categorical POSTBASE-357 result is adapted into POSTBASE-255 ranking only through an explicit fail-closed adapter: only `PASS` maps to score 1.0; every other status maps to score 0.0.

## Public trace

The emitted JSON includes only high-level auditable state: component policy, retrieved memory IDs and provenance hashes, supersession exclusion, hashed deliberation/tool trace, hypothesis IDs/statuses/evidence IDs, verifier statuses, and the final verified answer. Deliberation wall-clock fields are removed before canonical serialization because they are nondeterministic. Private scratch values are asserted absent.

The workflow runs the full fixture twice and requires byte-identical canonical JSON. It also hashes the learned Base checkpoint tree before and after the two runs and requires byte identity.
