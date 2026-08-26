# POSTBASE-353 SFT runner mechanics

Worker: `POSTBASE-353-SFT-RUNNER-MECHANICS`

Execution profile: `LOCAL_FREE`.

This worker adds the first supervised fine-tuning orchestration mechanics under the existing POSTBASE-253 communication-consumption boundary. It does not authorize or execute a real communication campaign.

## Scope

The implementation is framework-neutral and lives at `src/twelve_six/post_base/sft_runner.py`. A future model adapter may implement `SFTMechanicsBackend`; POSTBASE-353 itself does not introduce a second model stack, external LLM, hosted trainer, paid service, or foreign pretrained weights.

The runner binds every mechanics run to the existing `PostBaseConsumptionContract` and requires:

- stage `post_base.communication.supervision`;
- exact tokenizer compatibility with the canonical Base contract;
- exact dataset manifest, source-registry, train-split, and evaluation-split identities;
- output lineage and namespace already fixed by POSTBASE-253;
- `LOCAL_FREE` compute;
- fixture-only execution mode;
- no real-campaign authorization ID.

## Fixture-only data gate

POSTBASE-353 can execute only `SFTMechanicsDataset` fixtures. Tests use tiny deterministic synthetic/project-owned rows. Fixture provenance rejects `foreign_model_output=true`; no foreign model output is admitted.

This is intentionally narrower than the future communication dataset contract. A later authorized campaign must consume the separately governed communication-data contract rather than treating these toy fixture records as training data.

The mechanics dataset computes deterministic SHA-256 identities for:

- train split;
- evaluation split;
- source registry;
- complete fixture manifest.

The runner refuses to create a workspace if those identities do not exactly equal the identities bound into `PostBaseConsumptionContract`.

## Canonical Base immutability

The runner calls POSTBASE-253 `prepare_post_base_workspace()` before backend initialization. The backend receives only `<experiment_root>/input_checkpoint`, never the canonical Base path.

Before and after execution the runner re-snapshots both canonical Base and the cloned input checkpoint. Any byte, path, size, or SHA-256 change fails closed. Therefore optimizer-like mechanics may alter only backend state published into post-Base checkpoint generations; neither canonical Base nor the immutable cloned input checkpoint is a mutable update target.

Existing experiment roots are rejected and never overwritten.

## Checkpoint namespace and rollback

SFT mechanics checkpoints are isolated under:

`artifacts/post_base/sft/checkpoints`

Each state publication creates a new immutable directory:

`generation_000000`, `generation_000001`, ...

Generation zero is the pre-update backend state loaded from the cloned input checkpoint. Every later generation binds its parent generation, backend ID, backend-state snapshot hash, immutable Base checkpoint identity, input-checkpoint snapshot identity, and dataset manifest identity.

Generation directories are never overwritten. An `active.json` pointer may move between immutable generations. `rollback_to()` only moves that pointer to an existing older generation; it does not delete or rewrite newer generations. `load_state()` re-hashes backend bytes against the generation manifest before restoration.

The run receipt also preserves the exact POSTBASE-253 canonical Base `CheckpointRef` as the external rollback target.

## Evaluation separation

Evaluation evidence is not stored inside checkpoint artifacts. It is written under:

`evidence/post_base/sft/evaluations`

The runner requires a non-empty evaluation fixture split and performs both baseline and final evaluation. Metrics must be finite numeric values. Each evaluation record binds the immutable evaluation-split SHA-256 and is write-once.

This namespace remains distinct from canonical Base evidence at `evidence/base` and from SFT checkpoint artifacts.

## Backend boundary

`SFTMechanicsBackend` exposes only five operations:

1. load the cloned input checkpoint;
2. execute one supervised mechanics step;
3. evaluate a state;
4. save a checkpoint state;
5. load a checkpoint state.

POSTBASE-353 does not prescribe PyTorch, a future first-party model adapter, or any external training framework. This keeps the mechanics compatible with later learned 10M and compatible 20M adapters without coupling this worker to a model implementation that is being developed separately.

## Tests

`tests/test_post_base_sft_runner.py` uses a deterministic scalar toy backend and tiny fixture records. The suite proves:

- two mechanics updates occur only in post-Base state;
- canonical Base bytes remain unchanged;
- cloned input-checkpoint bytes remain unchanged;
- train/evaluation fixture identities are contract-bound before workspace creation;
- checkpoints and evaluation use separate post-Base namespaces;
- immutable checkpoint generations and rollback work;
- generation-byte tampering is detected before load;
- evaluation evidence cannot be overwritten;
- existing experiment roots are not overwritten;
- foreign-model fixture output is rejected;
- paid compute and real-campaign authorization are rejected;
- non-supervision communication stages are rejected.

## Truth boundary

A passing POSTBASE-353 test demonstrates runner/checkpoint/evaluation/rollback mechanics only. It is not evidence that a learned Base has been communication-trained, that a communication dataset is approved, or that SFT improves assistant quality.

Real communication SFT remains unauthorized. No production dataset, real learned Base campaign, external LLM output, paid compute, RLHF, DPO, preference optimization, or stage promotion is introduced by this worker.
