# Section 4 — AI QA defect → repair → retest control plane

## Sequential position

This is later-section fallback work. The canonical numerical frontier remains Section 0.
Section 4 cannot become DONE or merge ahead of Sections 0–3.

## Implemented control loop

`src/twelve_six/ai_qa_control.py` adds a fail-closed control protocol around the existing
Section-3 SIL plane instead of creating a second routing/swarm framework.

The control layer can:

1. ingest native Section-3 SIL FAIL evidence or a strict normalized CI/physical observation;
2. classify the failure deterministically as TEST, TIMEOUT, ENVIRONMENT, INTEGRITY, PHYSICAL,
   or UNKNOWN;
3. derive a minimal checked-in pytest reproducer through the Section-3 no-shell command parser;
4. bind the defect to an exact failing SHA and source-evidence identity;
5. materialize an isolated local Git repair branch from the exact failing SHA, apply the bounded
   patch without a shell, commit it deterministically, and bind base SHA, candidate SHA, patch
   SHA-256, proposer identity and the exact failure-packet identity;
6. execute component and adversarial regressions on the exact clean candidate checkout;
7. ingest a native SIL receipt bound to the exact candidate SHA;
8. require an explicit physical PASS when physical scope is REQUIRED, or an explicit
   NOT_APPLICABLE scope receipt when physical scope is NONE; and
9. emit only `READY_FOR_INDEPENDENT_PROMOTION` when all four gates are represented. The local
   runtime never emits authoritative `PROMOTE`: its actor IDs are provenance labels, not an
   authenticated proof that a different worker produced the decision. Final promotion therefore
   remains an external independent evidence/review gate.

The durable command surface supports failure-packet creation, external candidate binding,
`materialize-candidate` for an actual isolated local Git repair lineage, automated regression
execution, native SIL receipt creation, software-only physical-scope receipt creation, and
independent promotion assessment. Materialization never pushes a remote or reads credentials. Repair candidates also cannot rewrite the CI workflow-policy module, tests/workflows/control/tooling trust roots, or `.gitignore`; this prevents a candidate from weakening its own judge or hiding generated state from clean-worktree qualification. It uses a private temporary
Git index plus plumbing (`read-tree`, `apply --cached`, `write-tree`, `commit-tree`) rather
than checkout/worktree mutation, disables replacement-object resolution for exact-SHA semantics,
and suppresses repository hooks for the atomic ref creation. It creates only a deterministic local
`aiqa/repair/*` ref, so publication remains an explicit authorized operation.

## Gate chain

Canonical gate order:

1. component reproducer;
2. adversarial regression;
3. Section-3 SIL journey evidence;
4. corresponding physical gate or explicit software-only NOT_APPLICABLE scope.

Component/adversarial execution reuses `sil_qualification.parse_vector_command` and the
Section-3 exact-input runner rather than creating a second subprocess framework. Each gate gets a
canonical AI-QA regression input envelope bound to defect/candidate/gate/argv; PASS requires the
subprocess to report consumption of that exact envelope identity. Exact candidate SHA and
tracked-clean state are mandatory before execution.

A gate receipt is bound to candidate SHA, evidence identity and actor identity. Imported receipt
JSON is never itself promotion authority. The `assess` path re-runs component/adversarial tests on
the exact clean candidate, independently re-verifies the raw SIL evidence+log against package,
capability-registry and scenario authority, and accepts durable receipt bundles only when every
field exactly matches those live-verified receipts. Missing, duplicate, stale, wrong-SHA, forged-hash
or FAIL receipts block promotion. SIL/physical evidence produced by the same repair proposer also
blocks promotion. Software-only physical `NOT_APPLICABLE` is recomputed from the exact failure and
candidate; a required physical PASS remains blocked until a trusted physical verifier is integrated.

## Current closure boundary

The deterministic control protocol and executable CLI are implemented, but Section 4 is not
closure-ready yet. Before READY/DONE, current authority still needs:

- exact-head shared CI for this candidate;
- convergence on accepted predecessor Sections;
- at least one live failure round-trip showing failure evidence → minimal reproducer → isolated
  repair commit → component/adversarial retest → SIL → applicable physical decision;
- at least one live round-trip must exercise the new exact-failing-SHA local mutation adapter and
  preserve its resulting candidate evidence on an accepted lineage; and
- external independent evidence/review that converts a locally READY candidate into an actual promotion; self-asserted local actor labels are not sufficient authority.

## Durable surfaces

- `configs/control/ai_qa_policy_v1.json`
- `src/twelve_six/ai_qa_control.py`
- `tests/test_ai_qa_control_section4.py`
- `docs/AI_QA_CONTROL_SECTION4.md`
- `SEQUENTIAL_CLOSURE_STATE.md`

## Truth boundary

This control plane does not grant corpus admission, tokenizer-fit, optimizer/training,
learned-weight, final-test, paid-compute, physical-device, release or scale-promotion authority.
A repair candidate is not promoted merely because its proposer reports confidence or local green
tests.

Every automated regression chain now re-resolves the candidate commit with replacement objects disabled and requires exactly one parent equal to the failure packet's failing Git SHA. A hand-authored candidate manifest cannot substitute an unrelated passing commit merely by claiming the correct base SHA.
