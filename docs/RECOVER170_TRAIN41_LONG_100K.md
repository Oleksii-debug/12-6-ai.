# RECOVER-170 — TRAIN-41 long ~100K recovery

`SWARM_WORKER_ID: RECOVER-170-TRAIN41-LONG-100K`

This package recovers the failed TRAIN-41 long ~100K experiment without creating a second model, Trainer, checkpoint format, tokenizer, packing framework, evaluator, or inference backend.

## Historical failure

TRAIN-41 exact-head workflow run `32862102098` stopped before training because `tools/verify_locked_environment.py --profile linux-x86_64` rejected the aggregate profile as stale against the then-current `pyproject.toml`. No long trajectory existed from that run.

RECOVER-170 replaces that obsolete repository-wide gate with the accepted D08 purpose environment authority. Before any training, the dedicated workflow must successfully execute `tools/verify_purpose_environment.py --profile linux-x86_64-cuda-training` on the exact recovery SHA and retain its self-hashed evidence. Training remains CPU FP32 and LOCAL_FREE; the purpose profile supplies exact environment identity and does not assert GPU execution.

## TRAIN-41 semantics retained

The recovery keeps the TRAIN-41 / RESEARCH41 controlled ~100K Base geometry and initialization:

- 95,568 parameters;
- ModelSpec SHA-256 `4f1aaa6821360f0d22033356e011843646c8c14a6b4d20a3ad5b2ad125867470`;
- InitSpec SHA-256 `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`;
- canonical `s0-byte-v1`, vocab 256, no special tokens;
- AdamW LR 3e-4, betas 0.9/0.95, eps 1e-8, weight decay 0;
- constant LR, no warmup, gradient clip 1.0, FP32, seed 1337;
- batch 4 × sequence length 64;
- primary frontier 2,097,152 actual optimized causal targets;
- mandatory fresh-process recovery at the first update at/above 1,048,576 actual targets.

The run is gated by `Trainer.tokens_seen`, not a nominal padded-token estimate. Tail padding is masked and does not count as optimized evidence. Threshold overshoot is explicitly retained and must remain smaller than one batch capacity.

## Deliberate data-authority change

The historical S0 fixture is not used as representative data. RECOVER-170 moves the long experiment to the compatible accepted DATA-25 project corpus, identity `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`, with zero train/validation content overlap.

This preserves the comparison purpose because the model, initializer, tokenizer, optimizer, seed, batch/sequence geometry and causal-target budget remain fixed; only the obsolete tiny fixture is replaced. DATA-25 remains project-authored, so the report explicitly forbids a representative external-corpus quality claim.

## Scientific trajectory

Held-out DATA-25 validation BPB is measured at random initialization and at the predeclared TRAIN-41 budgets: 1K, 2K, 4K, 8K, 16K, 32K, 65K, 131K, 262K, 524K, 1.048M, 1.573M and 2.097M requested actual optimized targets. Each evaluation retains aggregate and UK/EN/code BPB and proves model-state non-mutation.

Training BPB is accumulated with exact causal-token weighting between held-out evaluations. The report retains the heldout-minus-training BPB gap and a deliberately narrow overfit proxy: first held-out rise of at least 0.01 BPB above the best prior point while interval training BPB still improves. This is an optimization/generalization diagnostic only and is not a privacy or memorization-extraction claim.

Raw greedy Base snapshots use fixed prompts `The `, `Україна ` and `def ` at 0, 16K, 65K, 262K, 1.048M, 1.573M and 2.097M requested targets. Snapshots execute through a D05 checkpoint and the canonical first-party inference loader.

Retained D05 checkpoints are captured at the first update at/above 65K, 262K, 1.048M, 1.573M and 2.097M requested causal targets. The report records requested and actual token counts, overshoot, optimizer step, checkpoint ID and save/verify wall time.

## Fresh-process recovery

Phase 1 stops at the first committed optimizer update at/above 1,048,576 actual causal targets. The next workflow step starts a separate Python process, constructs fresh model, Trainer and optimizer objects, verifies and loads the retained D05 checkpoint with RNG restoration, checks exact optimizer-step and `tokens_seen` restoration, recomputes held-out BPB and requires maximum aggregate/stratum drift <= 1e-12, requires exact greedy generation-token parity, reconstructs the deterministic DATA-25 mixture position from optimizer step, and continues to the final frontier.

## Evidence

The exact-head artifact retains purpose-environment evidence, DATA-25 manifest, evaluation identity, run manifest, phase-1 state, complete train curve, final machine report and all retained checkpoint directories. The final report is self-hashed and the validator re-verifies the final checkpoint identity.

No foreign pretrained weights, SFT, RLHF, DPO or paid compute are permitted. No intelligence, production-readiness, alignment or instruction-following claim is permitted. This is learned scratch-Base experiment evidence only and grants no stage promotion.
