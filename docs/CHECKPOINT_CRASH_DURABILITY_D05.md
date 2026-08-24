# D05 checkpoint crash-durability contract

## Status

This contract is an additive D05 hardening layer for checkpoint-v1. It does not
change checkpoint identity, serialization, load verification, model semantics,
or promotion authority.

The existing checkpoint core proves logical atomic publication by staging a
complete verified directory and renaming it into place. Logical rename
atomicity is not by itself a persistence guarantee after a host/kernel/power
failure because dirty file data and directory entries may still exist only in
the page cache.

## Local POSIX protocol

`save_durable_checkpoint()` uses a sibling staging checkpoint in the final
parent directory and does not expose the final path until all of these steps
complete:

1. Serialize checkpoint-v1 through the canonical `save_checkpoint()` path.
2. Re-verify the exact staged checkpoint inventory, manifest, hashes and
   checkpoint identity.
3. `fsync()` the three payloads, `manifest.json`, and `MANIFEST.sha256`.
4. `fsync()` the staged checkpoint directory so its five entries are durable.
5. Atomically rename the staged directory to the immutable final destination.
6. `fsync()` the destination parent directory before returning success.

The destination parent must already exist and must be a real non-symlink
POSIX directory. This avoids making a false recursive-directory durability
claim: persisting newly created ancestor directory entries would require
syncing each ancestor's parent as a separate transaction.

Checkpoint-v1 remains immutable. `overwrite=True` never authorizes replacing
an existing checkpoint.

## Failure semantics

Failures before the final rename are classified as unpublished and the sibling
staging directory is removed. The final destination does not appear.

A failure of the final parent-directory `fsync()` is different. The rename may
already have made a complete verified checkpoint visible. Deleting that path
would destroy useful evidence and would not undo the already-issued rename.
The API therefore raises `CheckpointDurabilityError(published=True)` and
preserves the destination. `confirm_checkpoint_durability()` can verify the
checkpoint again, sync all five files and the checkpoint directory, and retry
the parent-directory sync.

## Truth boundary

A successful return proves that the local POSIX fsync ordering completed. It is
not an empirical power-cut test and is not a universal durability guarantee for
NFS, SMB, object-store/FUSE mounts, distributed filesystems, container storage
with weaker host guarantees, faulty hardware, or storage devices that violate
flush semantics. Windows is not claimed by this layer. The repository's
existing trailing-dot Windows checkout blocker is a separate issue.

The publication namespace is expected to be trusted/single-writer with respect
to non-cooperating processes. The wrapper rechecks the destination immediately
before rename, but Python does not expose a portable POSIX directory
`RENAME_NOREPLACE` primitive across all supported environments. Concurrent
writers must allocate distinct checkpoint destinations or coordinate above this
API.

This work does not claim distributed D08 checkpoint durability, cross-host
atomicity, CANDIDATE/STABLE status, AUDIT-A/B PASS, paid-compute authorization,
or any behavioral Base change.
