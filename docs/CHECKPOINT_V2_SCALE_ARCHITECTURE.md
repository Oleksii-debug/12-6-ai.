# Checkpoint-v2 scale architecture

Status: engineering successor for larger-model training resume. Checkpoint-v1 remains unchanged and remains the S0 authority.

## Why checkpoint-v1 stops scaling

Checkpoint-v1 is deliberately strong for small, portable snapshots: one immutable directory, SafeTensors payloads, canonical JSON state, exact SHA-256 inventory, and a verified in-memory byte snapshot before mutation. Those properties are appropriate for S0, but several implementation choices scale linearly with the entire checkpoint rather than with one rank's local state:

- model save converts every tensor to contiguous CPU/NumPy storage before writing;
- optimizer, scheduler, trainer and RNG state are packed into one state payload;
- verification reads every payload into memory and retains those bytes in `VerifiedCheckpoint`;
- model and optimizer state are published through a single directory writer rather than distributed writers;
- there is no topology/reshard contract for a different world size;
- asynchronous save and retention are intentionally outside the v1 contract.

At 10M parameters, a normal fp32 model plus populated fp32 AdamW moments is already roughly an order of magnitude larger than weights alone. At 100M and 400M, a whole-checkpoint RAM snapshot and one-writer path become an avoidable architectural constraint even before optimizer or mixed-precision master-weight choices grow further.

This is not a defect in v1. It is the point at which a different training-resume format is justified.

## V2 decision

`src/twelve_six/checkpoint/v2.py` is additive. It does not modify checkpoint-v1 or the active v1 hardening, cross-architecture portability, or local-POSIX durability surfaces.

V2 uses:

- PyTorch Distributed Checkpoint (DCP) for model and optimizer state;
- DCP state-dict canonicalization through `get_state_dict()` / `set_state_dict()`;
- SafeTensors plus the existing pickle-free state-tree codec for per-rank trainer, scheduler and RNG control state;
- a manifest-last `COMPLETE` commit record;
- streaming SHA-256 verification of every payload file without retaining checkpoint-sized payload bytes in RAM;
- explicit source topology and writer records;
- exact semantic identity derived from the existing `CheckpointIdentity` fields.

V2 is a trusted project training-resume artifact. DCP-managed metadata is not treated as an untrusted third-party import format. The 12-6 control layer remains SafeTensors/canonical JSON and explicitly does not fall back to pickle.

## Filesystem layout

```text
checkpoint-v2/
  dcp/
    .metadata
    __0_0.distcp
    ...
  control/
    rank-00000.json
    rank-00000.safetensors
    rank-00001.json
    rank-00001.safetensors
    ...
  manifest.json
  MANIFEST.sha256
```

`manifest.json` is written only after DCP and every rank-local control payload is complete. A directory without that commit manifest is incomplete and cannot be resumed.

The manifest binds:

- Git SHA;
- ModelSpec and its hash;
- exact parameter count;
- tokenizer and ordered-vocabulary identities;
- dataset and run-manifest identities;
- training config, optimizer and scheduler descriptors plus their hashes;
- seed, precision, step and tokens seen;
- environment lock identity;
- source world size and declared parallelism geometry;
- every writer rank, backend and environment fingerprint;
- every payload path, byte length and SHA-256;
- migration provenance when the source was checkpoint-v1.

The checkpoint ID hashes semantic identity, topology, writers, storage declaration, file inventory and migration record.

## Integrity and load semantics

V1's strongest property is an immutable checkpoint-sized byte snapshot held in RAM before target mutation. V2 intentionally does not reproduce that property.

V2 instead:

1. requires a `COMPLETE` manifest;
2. validates closed schemas and semantic identity;
3. streams and hashes the entire payload inventory;
4. performs DCP load;
5. streams and hashes the entire payload inventory again;
6. only accepts the resume if the checkpoint ID is unchanged.

A checksum or identity failure is fatal. If storage changes during DCP load, the training process is considered invalid and must not continue. There is no rollback promise after a distributed load has begun. Completed checkpoints therefore require a controlled/immutable namespace; mutating a committed checkpoint in place is unsupported.

This trades checkpoint-sized RAM for two streaming integrity passes while preserving a fail-closed resume boundary.

## Async save

`begin_async_checkpoint_v2()` starts DCP asynchronous save and returns a handle. The `COMPLETE` manifest is not published until every rank calls `wait()` successfully. Only one async save may be in flight per process.

This is intentionally conservative. Async DCP may stage state in CPU memory, especially for accelerator training, so "async" is not a zero-memory feature. GPU staging memory and training-step overlap need separate GPU evidence before an async cadence is frozen.

## Distributed writers and resume topology

`ResumeTopology` records source `world_size` plus a structured parallelism mapping. The current implementation verifies that the declared world size equals the writer process group.

Exact-topology resume can restore each rank's trainer/scheduler/RNG control state. A topology change is rejected unless the caller explicitly sets `allow_reshard=True`. Rank-local RNG is not silently mapped across a topology change; topology-changing resume requires `restore_rng=False`.

DCP provides the model/optimizer resharding mechanism. This package proves replicated multi-rank writer mechanics, but it does **not** claim FSDP/DTensor sharded-state or topology-changing reshard execution until those are run on the project's distributed stack.

## Checkpoint-v1 migration

`migrate_checkpoint_v1_to_v2()` is a one-time bridge, not a reason to modify v1.

It:

1. verifies the source checkpoint-v1;
2. requires semantic identity equality rather than allowing relabeling during migration;
3. loads the source through the normal v1 loader;
4. saves the resulting state as v2;
5. records the source v1 checkpoint ID and source manifest SHA-256.

The migration necessarily pays v1's whole-snapshot RAM cost once. Future v2 resumes do not.

## Retention

`plan_retention_v2()` and `apply_retention_v2()` operate only on fully verified `COMPLETE` v2 directories. An incomplete directory is ignored by retention planning rather than being mistaken for a resumable checkpoint. A corrupted committed checkpoint causes retention planning to fail closed.

The current policy primitive supports:

- keep the newest N checkpoints;
- optionally keep every Nth training step as a sparse recovery series.

Deletion re-verifies each planned checkpoint immediately before removal.

## Object storage

Object storage is deliberately **not implemented** in this package. Treating an S3-compatible bucket or FUSE mount as a POSIX filesystem would create false atomicity and consistency assumptions.

A future store adapter must provide an explicit protocol such as:

- immutable generation/prefix per checkpoint attempt;
- distributed writer objects uploaded under that generation;
- content hashes and sizes collected without relying on mutable directory listing semantics;
- a final immutable/conditional-create commit manifest;
- readers that require that commit record before resume;
- retention that deletes only generations no longer reachable from retained commit records.

Until that adapter is implemented and exercised, the manifest records `NOT_IMPLEMENTED_REQUIRES_STORE_ADAPTER`.

## Scale trigger

Recommended checkpoint choice is stage-dependent rather than a global format replacement:

| Model stage | Training-resume recommendation |
| --- | --- |
| S0 ~10K | checkpoint-v1 remains authoritative |
| S1 ~100K | v1 remains adequate; v2 optional mechanics testing |
| S2 ~1M | qualify both; v2 becomes the scale-readiness path |
| S3 ~10M | prefer v2 for training resume; retain v1 only where compatibility/evidence requires it |
| ~100M+ | v2-class sharded/distributed checkpointing is required before launch |
| distributed/FSDP/DTensor | v2 only after the sharded reshard acceptance gate is green |

Using the current fp32+AdamW representative payload ratio from the 10M probe, a simple linear storage extrapolation is roughly 1.3 GB at 100M parameters and roughly 5.2 GB at 400M. These are storage-planning estimates, not measured checkpoints at those scales.

## Supporting local measurements

Before pushing this package, the prototype was exercised locally with exact S2/S3 parameter geometries and populated AdamW state. These values are supporting evidence only because the local runtime was PyTorch 2.10 rather than the repository's locked PyTorch 2.13 runtime.

| Geometry | Parameters | Full v2 directory | Median save | Median load |
| --- | ---: | ---: | ---: | ---: |
| S2 representative | 1,066,112 | 14,153,525 bytes | 0.179 s | 0.062 s |
| S3 representative | 10,059,840 | 131,647,325 bytes | 2.626 s | 0.711 s |

Save time includes DCP write plus streaming integrity/manifest work; load time includes pre/post integrity passes and state restoration. Filesystem cache produced substantial variance in the S3 local samples, so only the locked workflow should be used for stable engineering comparisons.

A local two-rank Gloo smoke also completed with two writer records, multi-rank DCP payloads, rank-local SafeTensors control files and exact model/AdamW round-trip on both ranks. The model was replicated; this is not FSDP/DTensor reshard evidence.

## Next acceptance gates

Before a large paid run depends on checkpoint-v2:

1. exact locked S2 1M and S3 10M save/load evidence must be green;
2. two-rank distributed writer evidence must be green;
3. FSDP or DTensor sharded save plus changed-world-size resume must be executed, not inferred;
4. accelerator async-save staging memory and overlap must be measured;
5. an object-store adapter is required before remote object storage is called supported;
6. recovery drills must prove selection of the newest valid `COMPLETE` checkpoint after an interrupted save.

None of these mechanics changes Base model behavior, grants stage promotion, or authorizes paid compute.
