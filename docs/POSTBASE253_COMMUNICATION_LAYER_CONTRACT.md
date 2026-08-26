# POSTBASE-253 communication-layer contract

Worker: `POSTBASE-253-COMMUNICATION-LAYER-CONTRACT`

This package defines an architectural handoff only. It does not train, fine-tune, align, personalize, or run a communication model.

## Canonical Base boundary

Canonical Base remains a random-initialized pretraining lineage. The retained Base artifact must declare all of the following as false: SFT applied, RLHF applied, DPO applied, personality applied, chat template applied, and external-LLM inference used for Base construction. A violation is rejected before a post-Base contract can be accepted.

The post-Base implementation lives under `src/twelve_six/post_base/`. Its evidence and artifact namespaces are fixed to `evidence/post_base` and `artifacts/post_base`. Those names are intentionally distinct from canonical Base evidence. A post-Base output is always `POSTTRAIN` lineage and cannot be labeled `BASE`.

## Versioned consumption contract

Schema: `12-6.post-base.communication-consumption.v1`.

A future communication experiment must bind:

- immutable Base checkpoint ID, SHA-256, source Git SHA, stage, and `BASE` lineage;
- the canonical Base policy assertions above;
- exact tokenizer ID, tokenizer-config SHA-256, vocabulary SHA-256, and vocabulary size;
- communication-dataset ID plus immutable manifest, source-registry, train-split, and evaluation-split SHA-256 identities;
- a stage identity under the explicit `post_base.communication.*` family;
- separate Base and post-Base evaluation namespaces;
- optionally, a versioned dialogue formatter that remains external to Base and cannot mutate tokenizer IDs, add special tokens, or install a Base chat template;
- an exact rollback target equal to the untouched input Base checkpoint.

POSTBASE-253 v1 has `execution_authorized=false` and rejects any attempt to set it true. A later worker that performs communication training must add separate authorization rather than reinterpret this contract as permission to train.

## Immutability proof

`prepare_post_base_workspace()` refuses any destination equal to, inside, or containing the retained Base checkpoint directory. It snapshots every checkpoint file by relative path, byte size, and SHA-256, rejects symlinks, then copies the checkpoint to `<experiment_root>/input_checkpoint` with independent regular files.

After the copy it requires:

1. the canonical Base snapshot is byte-identical to the pre-copy snapshot;
2. the cloned checkpoint snapshot is byte-identical to canonical Base;
3. no source/clone file pair shares the same device+inode, so the clone is not a hard link;
4. an existing experiment directory is never overwritten.

The regression suite then mutates the cloned weights and proves the retained Base bytes and directory identity remain unchanged. Rollback is therefore the exact original `CheckpointRef`, not a reconstructed or post-trained artifact.

## Evaluation and evidence separation

Canonical Base evidence remains under `evidence/base`. Communication-layer evidence is restricted to `evidence/post_base`. Post-Base results cannot be written into Base evidence or used to silently upgrade a Base checkpoint claim.

Training and evaluation dataset split identities must differ. No final/held-out result is converted into training input by this contract.

## Relationship to D09

This worker stacks on terminal-green D09 post-training infrastructure rather than creating a second post-training framework. D09 owns generic post-training records, provenance, verifiers, and future runtime adapters. POSTBASE-253 adds only the narrower Base-consumption, tokenizer/dialogue compatibility, evidence separation, rollback, and copy-on-write checkpoint boundary.

## Truth boundary

Tests only. No optimizer update. No gradient. No SFT, RLHF, DPO, preference training, reward training, or reinforcement-learning execution. No personality or chat template is added to canonical Base. No external LLM inference is used. No paid compute, paged attention, custom CUDA kernel, or model-weight change is introduced.

Execution profile: `LOCAL_FREE`.
