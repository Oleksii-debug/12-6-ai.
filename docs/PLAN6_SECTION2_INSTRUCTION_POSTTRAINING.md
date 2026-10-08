# Plan 6 / Section 2 — Instruction post-training descendant adapter

## Authority boundary
`src/twelve_six/instruction_post_training.py` is Plan-6-owned. It does not create a second optimizer, model architecture, training scheduler, or checkpoint authority. The bounded fixture runner composes Plan-3/D02 `Trainer` and `TrainerConfig`, and trains an isolated CPU `copy.deepcopy` of the supplied Base module. The original Base content SHA is verified before work and checked again after. This is not a production Base checkpoint verifier; the caller must supply an independently approved canonical parent checkpoint and actual tokenizer binding.

## Contract
- Versioned, immutable recipe with exact Base/model-state content SHA-256, tokenizer SHA, deterministic seed, distinct BOS/SEP/EOS/PAD token IDs, max context, batch, optimizer-step and LR budgets. All bounds fail closed.
- Each training-only instruction example binds source, rights and independent quality-evidence SHA-256 values and typed quality. Callers provide separately managed frozen allowlists for rights and quality plus reserved evaluation content fingerprints. A supplied digest or self-claimed quality alone is not trusted admission.
- Reject unapproved or unknown rights and quality evidence, evaluation/test split, reserved-eval collision, duplicate prompt/target content, malformed identities, injected special tokens, oversized records, and non-finite quality/recipe values.
- Format `BOS + prompt + SEP + target + EOS`; already-aligned `target_ids` supervise only target+EOS and use `-100` for prompt/SEP/padding. `loss_mask` matches the supervised positions. Seeded batching is replayable.
- The engine returns a content-addressed descendant receipt binding parent, recipe, admitted dataset, resulting weights, optimizer-step and optimized-token counts. No recipe or receipt authorizes promotion, real corpus, protected holdout exposure or paid training.

## Qualification
`tests/test_instruction_post_training_plan6_section2.py`: bounded CPU tiny PyTorch model, Base state nonmutation, deterministic rerun and descendant receipt replay; prompt masking, batching and aligned labels; forgery, duplicate, bad split, rights/quality denial, eval leakage, special-token injection, corrupted parent, oversized/context/budget and incompatible recipe. Also reuse `tests/test_trainer_checkpoint_adapter.py` for existing D02 trainer surface.

The Python source/test exact Git blobs were independently tested with a **temporary local test-only D02-interface shim** (24 passing; Python compileall passing). That shim is not checked into the repository and cannot be represented as full D02 integration. Exact-head hosted CI and the real D02 integration test remain qualification evidence to inspect separately. No material GPU/cloud compute, teacher model or production campaign was started.
