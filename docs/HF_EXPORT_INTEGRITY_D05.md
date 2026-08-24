# D05 HF-style export integrity

This package hardens the existing `export_hf_directory()` conversion handoff without claiming Transformers compatibility.

## Safety contract

A source checkpoint is verified before export. The export is built in a sibling staging directory and is verified against the source checkpoint before publication. The final destination is immutable: an existing directory, file, or symlink is never recursively removed, including when legacy callers pass `overwrite=True`.

The export inventory is exact and contains only:

- `model.safetensors` — exact byte copy of canonical checkpoint weights;
- `config.json` — caller-supplied HF-style configuration metadata;
- `12-6-checkpoint-manifest.json` — exact copy of canonical checkpoint provenance;
- `12-6-parity-request.json` — D07/independent parity handoff state;
- `12-6-export.json` — export attestation and file hashes.

Every entry must be a regular file, not a symlink or special file. `12-6-export.json` binds the hashes and sizes of the other four files and carries a canonical self-hash. `verify_hf_export()` recomputes the source checkpoint identity, exact weight/provenance equality, parity request semantics, file inventory and attestation self-hash.

## External parity hooks

A parity hook receives the verified source checkpoint and the unpublished staging directory. The hook is treated as evidence-producing but untrusted artifact-authoring code. After it returns, D05 re-verifies the source checkpoint and staged weights/config/provenance before recording the hook result. Mutation, extra files, hook failure, malformed evidence, or any later verification failure prevents publication.

`EXTERNAL_EVIDENCE_ATTACHED` means only that hook output was attached. It does not imply Transformers compatibility, logit parity, audit PASS, candidate promotion, or STABLE status. Runtime compatibility remains owned by D07/independent parity evidence.

## Publication boundary

Publication uses a fresh destination created by the current call. The attestation is moved last as the completion marker, then the final directory is verified again. If publication or final verification fails in-process, the newly created incomplete destination is removed. A process crash may leave an incomplete destination; subsequent calls fail closed instead of overwriting it.

This remains a LOCAL_FREE S0 conversion-integrity contract. It does not implement GGUF/llama.cpp conversion, distributed checkpoint export, foreign pretrained weights, instruction/alignment behavior, or release promotion.
