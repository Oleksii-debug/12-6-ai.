# Fresh reserved-evaluation decontamination V1

## Purpose

This package closes only the execution seam between the terminal post-global-dedup
survivor graph and the incumbent DATA-232 matcher. It does not define a new matching
algorithm and does not grant training authority.

The runner consumes the exact survivor-bound inventory produced by the #832 lineage,
reconstructs comparison payloads only in an ephemeral workspace, and delegates all
matching to the DATA-232 implementation converged by #857.

## Fail-closed bindings

A real scan requires externally supplied exact identities for the post-dedup inventory,
survivor authority, selection-validation reservation, final-test reservation, #832
handoff Git head and #857 matcher Git head. Self-consistent replacement of any durable
report is insufficient.

The CLI accepts only selection-validation payload rows. There is intentionally no
argument for a final-test payload. Evaluation authority metadata is passed through the
incumbent DATA-232 validator, which rejects outcome-bearing fields.

## Output boundary

The durable wrapper contains only hashes, counts, matcher evidence and exclusion
metadata already made text-free by DATA-232. It records whether decontamination is
`PASS_CLEAN` or `PASS_WITH_EXCLUSIONS`, but always keeps authorized training exposure at
zero. Quality/privacy, balance/family caps, cluster-safe split, deterministic packing,
two-clean-build proof and the post-pack unique-loss ledger remain mandatory successors.

## Execution sequence

1. Wait for #832 and #857 exact heads to reach terminal shared-CI success.
2. Bind their exact terminal Git SHAs and the terminal #832 inventory identity.
3. Reconstruct retained comparison payloads in an ephemeral directory.
4. Run the CLI twice independently with the same immutable inputs.
5. Require byte-identical durable reports and verify both against the external bindings.
6. Persist only the text-free report and environment/lineage evidence.

No paid compute, optimizer step, tokenizer fit, model training or final-test payload
access is part of this package.
