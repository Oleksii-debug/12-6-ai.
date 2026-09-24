# Learned-20M launch manifest + TRAINING_RUN lease V1

This contract closes the duplicate-launch safety gap identified by #653 without granting training authority or executing an optimizer update. It binds one immutable learned-20M launch manifest to one machine-readable `TRAINING_RUN` lease and provides deterministic idempotency keys for side effects such as checkpoint publication and log posting.

## Truth boundary

Validation, hashing, lease construction, local exclusive acquisition, renewal, termination, or idempotency-key generation do **not** change scientific truth. In particular this module always reports `optimizer_start_permitted_by_this_module=false`, `global_exclusivity_proven=false`, and `scientific_truth_changed=false`.

A legacy shape-only manifest contains exact references and SHA-256 identities for training/compute authority evidence, but those references are not self-authenticating. It is structurally valid only; it is never sufficient for the finalized launch-authority gate.

A finalized manifest embeds one closed-world `terminal_authority` envelope in the same V1 manifest. The envelope is self-hashed for corruption detection, but self-hashing is **not** trust: `assess_terminal_launch_authority()` additionally requires an independently supplied expected terminal-authority SHA-256. Rehashing a substituted candidate therefore cannot authorize itself.

## Manifest identity

`launch_manifest_sha256()` hashes canonical UTF-8 JSON (`sort_keys`, compact separators, integer-only numeric values) covering exact code, ModelSpec, InitSpec, tokenizer, corpus, split, packing, unique-loss ledger, training configuration, portable packet/binding, recipe/exposure budget, checkpoint contract, evaluation firewall, free-only resource envelope, authority evidence references, execution backend, and—when finalized—the entire terminal-authority envelope.

The terminal envelope additionally binds the incumbent launch-input/carrier authority, tokenizer-decision root, loss-bearing manifest, exposure plan, recipe/seed identities, RecoveryStore run/attempt manifest, checkpoint cadence/resume rules, safe-stop current-run identity, D06 evaluation schedule/firewall, poison-stop semantics, resource evidence/execution target/measured resource envelope, random-init/no-foreign-weight truth, and exact training/compute authority evidence.

The independently expected terminal root also carries `base_manifest_sha256`, the canonical hash of the complete manifest body with `terminal_authority` removed. This closes the trust-direction gap for every behavior-affecting base field, including optimizer/scheduler/precision, seed, checkpoint lineage, identities, resource policy and execution backend: changing any of them invalidates the externally rooted terminal authority even when the terminal envelope itself is left byte-for-byte unchanged.

The checked contract is deliberately free-only: resource class is `LOCAL_FREE`, `GITHUB_HOSTED_FREE`, or `FREE_GPU`; maximum material cost is exactly zero; `materially_paid=false`; and final-test payload access must remain false.

`finalize_launch_manifest()` only embeds a structurally closed terminal envelope. It does not authenticate that envelope. A finalized manifest remains launch-blocked until `assess_terminal_launch_authority()` receives an independently expected terminal-authority hash and the envelope reports a strict integer `authorized_optimized_target_exposure > 0`. Zero, booleans, malformed values, a substituted root, foreign pretrained weights, final-test access, paid compute, or any disagreement with the manifest's incumbent identities fails closed.

## Lease lifecycle

`build_training_run_lease()` binds a unique `run_id` and holder to the exact manifest hash plus the exact training/compute authority references and hashes. It remains a local duplicate-guard primitive and does not authenticate scientific authority.

`build_authorized_training_run_lease()` is the finalized composition entrypoint. It canonicalizes the complete caller manifest once into a private JSON snapshot, performs the external terminal-authority assessment against that snapshot, and builds the **same** `TRAINING_RUN` lease from that exact snapshot. Before returning, it requires the lease manifest SHA-256 to equal the manifest SHA-256 authenticated by the assessment. A caller mutation after the assessment boundary therefore cannot change the emitted lease. It still does not execute an optimizer step and does not prove global exclusivity.

A RUNNING lease has a maximum six-hour TTL and an explicit renewal sequence. Expired leases cannot be renewed, including through a manually constructed transition candidate: the candidate renewal instant must still be strictly before the previous lease expiry.

Trusted-time assessment is effectivity-aware. A structurally valid lease whose acquisition time is still in the future, or whose latest renewal is later than the assessor's trusted `now`, remains evidence but cannot open the local duplicate guard and cannot be persisted by the acquisition primitive.

Terminal states are `COMPLETED`, `FAILED`, and `ABORTED`; terminal records cannot transition again. `COMPLETED` is success evidence and therefore its terminal instant must be strictly before the previous RUNNING lease expires. `FAILED` and `ABORTED` may be recorded at or after expiry as post-expiry failure/abort evidence; that distinction does not reopen the lease or authorize a replacement run.

`acquire_local_training_run_lease()` derives exactly one path from the manifest hash and uses an exclusive filesystem create. A second acquisition for the same manifest on that filesystem fails. Existing records are never silently overwritten, including expired or terminal records; they remain evidence requiring an explicit higher-level replacement/relaunch decision.

All enum-like fields supplied through JSON are type-guarded before membership checks. Malformed JSON arrays/objects are denied as candidate data instead of escaping the validator with a Python `TypeError`; the assessor CLI retains machine-readable denial behavior for these cases.

### Critical scope limit

The exclusive-create primitive proves only `SINGLE_SHARED_FILESYSTEM_ONLY` atomicity. Separate GitHub-hosted runners do not share that filesystem. Therefore this primitive **must not** be treated as a global or distributed training lock. The GitHub-hosted execution carrier must add one canonical cross-runner/global exclusivity mechanism and bind it to the same manifest identity before optimizer step 1.

## Idempotency

`training_side_effect_idempotency_key()` deterministically binds the launch-manifest SHA-256, run ID, effect kind, and logical step. The same logical effect gets the same key; changing manifest/run/effect/step changes the key. Consumers should use it for retryable external effects and checkpoint/log publication while still verifying the underlying artifact identity.

## Assessor

```text
python tools/assess_learned20m_training_lease.py MANIFEST [LEASE] [--now YYYY-MM-DDTHH:MM:SSZ]
```

A zero exit from the legacy CLI means the **local duplicate-guard contract** is active for the supplied manifest/lease at that time. It does not mean training is authorized. Missing, malformed, terminal, expired, or not-yet-effective leases fail closed. The canonical launcher must separately call the terminal-authority assessment before it may use the authorized lease-builder path.

## Required launcher composition before real training

Before the first real optimizer step, the canonical launcher must require all of these independently: terminal scientific/corpus/tokenizer/exposure authority; externally verified training and compute authority; exact immutable finalized launch-manifest identity; one global/cross-runner single-training lease; this local duplicate guard where applicable; canonical runtime/recovery/evaluation gates; and stable idempotency for external effects. Any missing gate keeps optimizer execution blocked.

## Current physical status

This code establishes the composition mechanics only. It does **not** fabricate the final production manifest while clean-corpus/exposure and neighboring runtime/recovery/resource authorities are still converging. Until the live authority chain proves positive optimized-target exposure and supplies the exact incumbent roots, the only honest current state remains BLOCKED: authorized optimized-target exposure is zero, tokenizer-fit authority is not established here, optimizer updates/training have not executed, learned weights do not exist, final-test outcomes have not been read, paid compute has not been used, and foreign pretrained weights have not been used.
