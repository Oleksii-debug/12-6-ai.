# NEXT100-065F — current-main global cross-source dedup V8

## Purpose

This package continues the incumbent NEXT100-065 / issue #569 global-dedup lineage on current `main` after merged PR #818 added a terminal six-family permissive Python source bundle. It is a selective successor, not a second dedup architecture.

The historical PR #632 branch is heavily diverged from current `main`. V8 therefore does **not** merge that obsolete ancestry. Instead, execution checks out the exact terminal V7 head `d3333ec1b4a508df232a5aefccd6686adda745fb`, reproduces its exact 35-object report identities, captures the exact authority-approved payload graph at the incumbent V3 matcher boundary, and then appends the exact 229 implementation files terminalized by PR #818 before invoking that same lineage-aware matcher once over the composed graph.

## Exact inputs

Terminal V7 baseline:

- head `d3333ec1b4a508df232a5aefccd6686adda745fb`;
- immutable artifact `9635595510`;
- V7 report identity `80997ac88b9d604afaf652807cda2a2d9fd0f6cb75754460ae6f4aa7af6e0267`;
- nested V3 dedup identity `c33e0d06a469473aac191e9b5bf7baec23322cd3e9200f0caab2633c921afd84`;
- 35 objects / 2,215,615 conservative unique source bytes / UK4-EN5-code6 families.

Merged DATA-BULK-CODE-1:

- PR #818 merged to `main@4fa4839f2882984e7bf148dc38ce315d51b60957`;
- terminal execution head `a045c602bfcead862f7852924fb78a5d78c992d6` / run `34155446113`;
- artifact `10030825509`, digest `sha256:9febb6e4e900df63c58b1cb0ed2a7003df56d6e9bd593e4b4adbaa54496010ac`;
- terminal report identity `80f3a8f20dfb82825e6c89ac1f76f2f41233296b8a426a1cf466b0fcc0892985`;
- 229 exact eligible Python files / 3,880,009 UTF-8 bytes / six source families.

The composed pre-global-dedup graph is exactly 264 source objects and 6,095,624 declared source-capacity bytes, with UK4 / EN5 / code12 source families. This number is an input total, not an assumed post-dedup result.

## Execution semantics

The runner reproduces terminal V7 rather than trusting copied metadata. During the V7 execution it temporarily wraps the incumbent `audit_payloads` call only to capture the exact already-validated inventory and payload bytes; the original matcher still produces and verifies the V7 report. The wrapper is then removed.

The current-main PR #818 materializer is executed fresh. It rechecks exact Git commits, licenses, selection policy, AST/UTF-8 constraints and the bounded credential firewall. Every selected file is reread from the exact checkout and must match its terminal SHA-256 and byte count before it enters the composed graph.

Finally, the original V3 lineage-aware global matcher is invoked over all 264 payloads. The resulting exact/near/fragment/lineage discount is measured, not predeclared. Two independent executions must produce byte-identical reports before terminal evidence can be retained.

## Truth boundary

This V8 result is still pre-decontamination source capacity. The previous PASS_CLEAN evaluation-decontamination result covered only the earlier 48-record graph and cannot be reused for the 229 new code objects. New bytes therefore require a fresh reserved-evaluation decontamination run, then post-composition quality/privacy, balance/family caps, cluster-safe split, deterministic packing/two-clean-build proof and a positive post-pack unique causal-loss ledger.

Always false/zero here: corpus release, tokenizer fit authority, optimized training exposure, model training, final-test payload access and paid compute. `LOCAL_FREE` only.

## Temporary execution workflow

A temporary PR-scoped network workflow may be used to obtain the required two-clean real-source execution because generic shared CI cannot substitute for external immutable source materialization. After terminal evidence is retained on the branch, that temporary workflow must be removed before final merge; shared CI on the workflow-free exact head remains the repository integration gate.
