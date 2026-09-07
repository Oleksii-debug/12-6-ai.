# Fresh reserved-evaluation source-object screen V1

## Purpose

This package closes an early execution seam between the terminal post-global-dedup
survivor graph and the incumbent DATA-232 matcher. It does not define a new matching
algorithm and does not grant training authority.

The #824 -> #832 lineage is a **source-object** authority: it retains one physical
source object per terminal dedup component and carries comparison-payload identities.
It is not yet the exact final training-record inventory required by #548. Therefore
this package deliberately reports only a source-object reserved-evaluation screen.
It must never claim that final record-level reserved-evaluation decontamination is
complete.

The runner consumes the exact survivor-bound inventory produced by #832, reconstructs
comparison payloads only in an ephemeral workspace, and delegates all matching to the
DATA-232 implementation converged by #857.

## Fail-closed bindings

A real screen requires externally supplied exact identities for the post-dedup
inventory, survivor authority, selection-validation reservation, final-test reservation,
#832 handoff Git head and #857 matcher Git head. Self-consistent replacement of any
durable report is insufficient.

The CLI accepts only selection-validation payload rows. There is intentionally no
argument for a final-test payload. Evaluation authority metadata is passed through the
incumbent DATA-232 validator, which rejects outcome-bearing fields.

## Output boundary

A successful run produces `SOURCE_OBJECT_PASS_CLEAN` or
`SOURCE_OBJECT_PASS_WITH_EXCLUSIONS`. The durable wrapper contains only hashes, counts,
matcher evidence and exclusion metadata already made text-free by DATA-232.

Even after a successful source-object screen, all of the following remain false or
required:

- `record_level_training_inventory_materialized = false`;
- `record_level_reserved_evaluation_decontamination_required = true`;
- `reserved_evaluation_decontamination_complete = false`;
- `authorized_training_exposure = 0`.

The final training-record graph must be materialized after source normalization and
record/chunk extraction, then decontaminated again against the immutable evaluation
reservations before split/training authority. Quality/privacy, balance/family caps,
cluster-safe split, deterministic packing, two-clean-build proof and the post-pack
unique-loss ledger also remain mandatory successors.

## Execution sequence

1. Wait for #832 and #857 exact heads to reach terminal shared-CI success.
2. Bind their exact terminal Git SHAs and the terminal #832 inventory identity.
3. Reconstruct retained comparison payloads in an ephemeral directory.
4. Run the source-object screen twice independently with immutable inputs.
5. Require byte-identical durable reports and verify them against external bindings.
6. Persist only the text-free source-object screen evidence.
7. Materialize the final record-level training inventory and rerun reserved-evaluation
   decontamination before any split/training promotion.

No paid compute, optimizer step, tokenizer fit, model training or final-test payload
access is part of this package.
