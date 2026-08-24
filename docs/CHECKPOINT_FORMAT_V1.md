# 12-6 AI checkpoint format v1

D05 owns this serialization contract. The format is deliberately data-only: no checkpoint path uses Python pickle.

A checkpoint directory contains:

- `weights.safetensors` — model state tensor payloads;
- `state.safetensors` — tensor leaves from optimizer, scheduler, trainer/RNG state;
- `state.json` — nested data-only structure that references `state.safetensors` leaves;
- `manifest.json` — lineage, training identity, environment evidence, file hashes and artifact identity;
- `MANIFEST.sha256` — integrity record for the manifest itself.

`manifest.json` format version 1 records full Git SHA, full ModelSpec plus its hash, parameter count, tokenizer-config SHA-256, tokenizer vocabulary-mapping SHA-256, dataset-manifest SHA-256, full launch/run-manifest SHA-256, training config plus hash, seed, optimizer/scheduler metadata plus hashes, precision, step, tokens seen, environment/package snapshot, optional dependency-lock SHA-256, every payload SHA-256/size, and one checkpoint identity hash covering the identity and payload records.

The format was hardened before any canonical S0 checkpoint existed, so the pre-release format number remains v1. Earlier transient test artifacts that lack the vocabulary/run-manifest identities are intentionally rejected by current verification rather than treated as canonical-compatible artifacts.

## Save/load invariants

`save_checkpoint` stages into a sibling temporary directory, verifies the completed bundle, and only then renames it into place. Existing checkpoints are not overwritten unless explicitly requested.

`CheckpointIdentity.validate()` rejects abbreviated Git identities and placeholder/non-hash lineage strings. Required candidate/tokenizer/dataset/run hashes must be exact lowercase hexadecimal values of the documented width. This applies even when callers construct `CheckpointIdentity` directly rather than going through the C01 run-manifest binder.

`verify_checkpoint` validates `MANIFEST.sha256`, exact identity formats, the ModelSpec/training-config/optimizer/scheduler/environment internal hashes, every payload size/hash, and `checkpoint_id`. Only after integrity and optional lineage constraints pass can `load_checkpoint` mutate model/optimizer/scheduler/RNG state.

The state-tree codec preserves mappings with non-string keys, tuples, lists, bytes, NumPy scalars/arrays, and PyTorch tensors when PyTorch is installed. Unknown Python objects fail closed rather than being pickled.

RNG capture covers Python and NumPy. When PyTorch is installed it also records CPU RNG, available CUDA RNG states, and the deterministic-algorithms flag. CUDA RNG restoration fails closed if the runtime device count differs; callers may explicitly load with `restore_rng=False` instead of receiving a false determinism claim.

## Current evidence boundary

Unit tests include a real interrupted NumPy training run: three steps are trained, saved, loaded into a fresh model/optimizer/scheduler with deliberately different RNG state, and continued to step eight. Final weights, momentum state, scheduler state, Python RNG and NumPy RNG must exactly equal an uninterrupted eight-step baseline.

A PyTorch AdamW interrupted-resume test runs when PyTorch is installed and verifies restored model state, optimizer state indirectly through the next exact update, and PyTorch RNG through an identical sampled continuation batch. This proves the generic adapter path, not the integrated 12-6 S0 Base. The S0 Base save/load/generation/resume proof remains pending exact D01-D08 composition and must not be claimed from these adapter tests.

`export_hf_directory` produces the conventional `model.safetensors` + `config.json` layout and preserves the 12-6 manifest. It does **not** by itself prove Transformers compatibility; D01/D07 must supply architecture-specific registration/config mapping and compare outputs.

Large checkpoints are artifacts, not git content. Commit only code, tests, manifests/evidence and small fixtures.
