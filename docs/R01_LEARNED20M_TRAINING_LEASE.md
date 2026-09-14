# Learned-20M launch manifest + TRAINING_RUN lease V1

This contract closes the duplicate-launch safety gap identified by #653 without granting training authority or executing an optimizer update. It binds one immutable learned-20M launch manifest to one machine-readable `TRAINING_RUN` lease and provides deterministic idempotency keys for side effects such as checkpoint publication and log posting.

## Truth boundary

Validation, hashing, lease construction, local exclusive acquisition, renewal, termination, or idempotency-key generation do **not** change scientific truth. In particular this module always reports `external_authority_verified=false`, `optimizer_start_permitted_by_this_module=false`, `global_exclusivity_proven=false`, and `scientific_truth_changed=false`.

The manifest contains exact references and SHA-256 identities for training/compute authority evidence, but those references are not self-authenticating. The canonical launcher must verify them through the existing trusted authority chain independently of candidate manifest content before any optimizer effect.

## Manifest identity

`launch_manifest_sha256()` hashes canonical UTF-8 JSON (`sort_keys`, compact separators, integer-only numeric values) covering exact code, ModelSpec, InitSpec, tokenizer, corpus, split, packing, unique-loss ledger, training configuration, portable packet/binding, recipe/exposure budget, checkpoint contract, evaluation firewall, free-only resource envelope, authority evidence references, and execution backend.

The checked contract is deliberately free-only: resource class is `LOCAL_FREE`, `GITHUB_HOSTED_FREE`, or `FREE_GPU`; maximum material cost is exactly zero; `materially_paid=false`; and final-test payload access must remain false.

## Lease lifecycle

`build_training_run_lease()` binds a unique `run_id` and holder to the exact manifest hash plus the exact training/compute authority references and hashes. A RUNNING lease has a maximum six-hour TTL and an explicit renewal sequence. Expired leases cannot be renewed. Terminal states are `COMPLETED`, `FAILED`, and `ABORTED`; terminal records cannot transition again.

`acquire_local_training_run_lease()` derives exactly one path from the manifest hash and uses an exclusive filesystem create. A second acquisition for the same manifest on that filesystem fails. Existing records are never silently overwritten, including expired or terminal records; they remain evidence requiring an explicit higher-level replacement/relaunch decision.

### Critical scope limit

The exclusive-create primitive proves only `SINGLE_SHARED_FILESYSTEM_ONLY` atomicity. Separate GitHub-hosted runners do not share that filesystem. Therefore this primitive **must not** be treated as a global or distributed training lock. The GitHub-hosted execution carrier (for example the disjoint #1790 lineage or its lawful successor) must add one canonical cross-runner/global exclusivity mechanism and bind it to the same manifest identity before optimizer step 1.

## Idempotency

`training_side_effect_idempotency_key()` deterministically binds the launch-manifest SHA-256, run ID, effect kind, and logical step. The same logical effect gets the same key; changing manifest/run/effect/step changes the key. Consumers should use it for retryable external effects and checkpoint/log publication while still verifying the underlying artifact identity.

## Assessor

```text
python tools/assess_learned20m_training_lease.py MANIFEST [LEASE] [--now YYYY-MM-DDTHH:MM:SSZ]
```

A zero exit means the **local duplicate-guard contract** is active for the supplied manifest/lease at that time. It does not mean training is authorized. Missing, malformed, terminal, or expired leases fail closed.

## Required launcher composition before real training

Before the first real optimizer step, the canonical launcher must require all of these independently: terminal scientific/corpus/tokenizer/exposure authority; externally verified training and compute authority; exact immutable launch-manifest identity; one global/cross-runner single-training lease; this local duplicate guard where applicable; canonical runtime/recovery/evaluation gates; and stable idempotency for external effects. Any missing gate keeps optimizer execution blocked.
