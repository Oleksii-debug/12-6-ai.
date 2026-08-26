# NEXT100-070 Two Clean Build

Worker: `NEXT100-070-TWO-CLEAN-BUILD`

Execution profile: `LOCAL_FREE`

## Purpose

This worker tests deterministic corpus construction at the current live corpus boundary. It must never convert missing hard-gate evidence into an implied corpus product.

The worker is stacked on terminal DATA-301 / frozen DATA-300 v2 and late-binds the current canonical source registry and purpose-specific rights vector before execution.

## Bound candidate

The live canonical DATA-287 registry and DATA-293 rights vector still describe the same exact five training-admitted objects frozen by DATA-300:

- 5 source objects;
- 4 independent source families;
- 183,061 unique normalized bytes;
- family counts: Ukrainian 1, English 1, code 2;
- all five objects have explicit model-training authority;
- evaluation purpose is not inferred from training rights.

DATA-300 requires at least two independent families per stratum and forbids replay/duplication as quota repair. Therefore the current family-constrained balanced no-replay capacity is zero.

New source-specific ADMIT authorities are not silently composed into this candidate. They require a terminal successor source-registry/corpus authority before they can change the canonical build inventory.

## Refreshed live gate evidence

The historical DATA-300 blocker list is not copied blindly. NEXT100-070 binds later terminal evidence where available:

- `G05_QUALITY`: DATA-296 successfully scanned the exact five-source / 183,061-byte candidate, but its whole-source sanity check exposed a filter-granularity hazard. Until a successor freezes the granularity policy, release remains blocked.
- `G06_PRIVACY`: VERIFY-307 reran the incumbent privacy policy over all five candidate objects; all five passed with zero candidate findings/redactions. Its adversarial suite still recorded false negatives, so this is a current-candidate scan PASS rather than a universal detector-strength claim.
- `G07_DEDUP`: DATA-298 terminal source-level evidence measures 183,061 -> 183,061 bytes with zero duplicate discount. Cluster-safe split/decontamination work is still not reached; the newer NEXT100-065 V3 exact-head workflow is queued and is not promoted to authority.
- `G08_RESERVED_DECONTAMINATION`: NEXT100-066 terminally blocks because no exact candidate corpus identity exists. No contamination PASS is claimed.
- `G09_BALANCE_DIVERSITY`: hard FAIL at family counts `1/1/2`; no replay budget is authorized.
- `G10_SELECTION_VALIDATION`: EVAL-303 is now terminal and nonempty: 10 selection records, UA 8 / EN 2 / code 0, with an exact DATA-300 object/hash exclusion proof. Near-copy/cluster decontamination remains a separate mandatory gate.
- `G12_UNIQUE_LOSS`: DATA-294 does not cover all five current source objects, so no full current-candidate causal loss ledger exists.

Any one hard prebuild failure is sufficient to prevent split/shard materialization. `G09` alone is mathematically decisive for the current candidate.

## Deterministic execution boundary

The harness validates the exact Git blob and declared identity of the late-bound registry, validates the exact rights blob, and requires source-by-source equality with the frozen DATA-300 inventory. It then emits canonical JSON under a clean output root.

The emitted surfaces are:

- `normalized_records/manifest.json`;
- `split_manifests/manifest.json`;
- `shards/manifest.json`;
- `rights_manifest.json`;
- `quality_evidence.json`;
- `privacy_evidence.json`;
- `dedup_evidence.json`;
- `decontamination_evidence.json`;
- `loss_ledger.json`;
- `gate_report.json`;
- `tree_manifest.json`.

When a hard prebuild gate fails, split and shard payloads are not created. Their manifests explicitly say `NOT_REACHED_PREBUILD_HARD_GATES`, carry no corpus identity, and have zero payload files. This is blocker evidence, not an empty corpus.

Canonical JSON uses UTF-8, sorted keys, compact separators and one trailing newline. Wall-clock time, host name, absolute workspace path, UUIDs, filesystem iteration order and network response order do not enter output identity.

## Two independent clean executions

The dedicated workflow uses two separate `ubuntu-24.04` jobs, `clean-a` and `clean-b`. Neither job uses `actions/cache` or any project cache. Each job:

1. checks out the candidate;
2. fetches the exact configured DATA-287 and DATA-293 branch heads;
3. fails if either live branch head differs from its late-bound expected SHA;
4. extracts the exact registry and rights files from those commits;
5. creates a fresh empty output root;
6. executes the deterministic harness to the current hard blocker;
7. hashes every emitted file in canonical relative-path order.

A third job requires both the composite tree digest and the complete base64-encoded path/SHA listing to be identical. Equality therefore covers every required surface, including `tree_manifest.json` itself.

The workflow contains no package installation, model training, tokenizer fitting, GPU work, paid compute, mutable shared cache, or corpus replay.

## Claim boundary

A successful NEXT100-070 workflow means only that two isolated executions reproduced the same exact authority-bound blocker tree. It does **not** mean:

- corpus built;
- corpus frozen;
- shards built;
- decontamination passed;
- full five-source loss ledger exists;
- family diversity is sufficient;
- model training is authorized or executed.

If a later terminal canonical registry or rights authority changes the source vector, this worker must receive a successor binding and rerun before its evidence may be cited.
