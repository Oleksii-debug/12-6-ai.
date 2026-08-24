# D05 first-party inference snapshot consistency

## Scope

The canonical D01+D04+D05 to D07 adapter already verifies checkpoint integrity, reconstructs the accepted `ModelSpec`, binds the canonical byte tokenizer, loads exact weights, and exposes D07 generation/CLI/server semantics.

PR #85 strengthened the D05 core so `prepare_checkpoint_load()` captures one immutable verified in-memory byte snapshot and `load_verified_checkpoint()` consumes only that snapshot. The first-party adapter still had a narrower adapter-level check/use split: it called `verify_checkpoint(path)` to obtain the manifest and later called `load_checkpoint(path)` to load weights. Both core calls were individually safe, but the filesystem path could name a different valid checkpoint between them.

That could produce a backend whose diagnostics reported checkpoint A while its model contained weights from checkpoint B, provided B retained compatible ModelSpec/tokenizer identities. This is an inference provenance defect even though neither checkpoint is corrupt.

## Required invariant

`load_first_party_backend(path)` now performs exactly one D05 snapshot operation:

1. `prepare_checkpoint_load(path)` reads and verifies the exact checkpoint inventory and bytes.
2. ModelSpec/context and tokenizer/vocabulary compatibility are derived from that verified snapshot manifest.
3. `load_verified_checkpoint(snapshot, ...)` decodes and applies the model weights from the same snapshot without reopening the path.
4. Backend diagnostics use the manifest returned from that same verified load result.

After step 1, later filesystem mutation of `path` cannot change either the loaded weights or the checkpoint identities reported by the backend.

## Regression

`tests/test_first_party_snapshot_consistency.py` creates two independently valid canonical S0 checkpoints with the same ModelSpec/tokenizer contract but different Git identity, checkpoint ID, step, and model weights. The test swaps the filesystem path to the second checkpoint after the first verified snapshot has been captured but before the adapter continues loading.

Acceptance requires:

- exactly one adapter-level snapshot preparation;
- backend diagnostics remain bound to the original checkpoint ID/Git SHA/step;
- every loaded model tensor remains exactly equal to the original snapshot weights;
- the live filesystem path is independently verified to contain the replacement checkpoint after the swap, proving that path identity and backend authority have diverged safely rather than accidentally.

## Boundary

This change does not alter D01 architecture, D02 training, D03 data, D04 tokenizer/packing, D05 checkpoint format, D06 evaluation, D07 generation/sampling/server semantics, D08 locks, or D10 promotion authority. It adds no dependency and no Base behavior. Canonical Base remains random-initialized and pretraining-only.

The checkpoint path retained on `FirstPartyInferenceBackend` is informational. Once the backend has loaded, the verified snapshot used during construction is the provenance authority; later path contents are not.
