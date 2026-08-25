# TRAIN-41 long ~100K Base experiment

## Question

What can the current strongest executable approximately 100K from-scratch 12-6 Base candidate actually learn when it is trained far beyond the short preflight horizon under one fixed recipe and observed densely enough to expose instability, plateau, overfit, checkpoint cost, and resume correctness?

## Incumbent preserved

TRAIN-41 is stacked on RESEARCH06 PR #183 and uses its exact first controlled ModelSpec: 95,568 trainable parameters, vocab 256, context 256, d_model 48, three blocks, four query/four KV heads, head_dim 12, SwiGLU d_ff 128, pre-RMSNorm, RoPE, tied embeddings, no dropout.

The optimizer is not retuned: FP32 AdamW, learning rate 3e-4, betas 0.9/0.95, weight decay 0, constant LR, clip norm 1.0, seed 1337, batch 4 x sequence 64, one optimizer update per strict aligned causal-pair microbatch.

The exact byte tokenizer and S0 train/validation split remain fixed. The strict RESEARCH06 `loss_mask` path counts exactly the causal targets that contribute to loss.

## Why the byte tokenizer and S0 fixture remain selected

The live DATA-10 recipe records a repeatable ByteLevel BPE experiment with actual vocab 472 and substantially lower controlled fertility, but it is explicitly not frozen on representative multilingual data. Switching to it would also alter the approximately 100K parameter allocation and confound this duration experiment.

The repository has no representative approved scale corpus yet. The S0 controlled fixture contains 1,920 UTF-8 train bytes and 406 validation bytes. DATA-10's project-authored mechanics corpus is only 1,454 bytes, and external approved training sources are zero. Therefore the current S0 fixture is the only incumbent with an exact established train/held-out training identity. TRAIN-41 does not relabel it representative.

This means the experiment is strong evidence about optimization, memorization, held-out behavior on the controlled fixture, numerical stability, checkpoint/resume correctness, and systems overhead. It is not evidence of broad Ukrainian/English/code language-model quality.

## Run length and observation plan

Planned maximum: 2,097,152 exact optimized causal targets, about 21.95 optimized byte targets per parameter.

Scheduled held-out evaluations: 1,024; 2,048; 4,096; 8,192; 16,384; 32,768; 65,536; 131,072; 262,144; 524,288; 1,048,576; 1,572,864; 2,097,152 optimized targets, plus an initial zero-token evaluation.

Step telemetry is every optimizer step through step 64, every 8 steps through 512, every 32 through 2,048, then every 128 steps and at every evaluation boundary. The report records train loss/BPB, validation loss/BPB, LR, pre-clip global gradient norm, clip activation/rate, sampled relative L2 update ratio, step time, optimized-token throughput, evaluation time, checkpoint save/load time, and peak process RSS.

Greedy generation snapshots are retained at 0; 16,384; 65,536; 262,144; 1,048,576; 1,572,864; and 2,097,152 optimized targets for three fixed raw-Base prefixes: `The `, `Україна `, and `def `.

## Checkpoint and resume contract

Full D05 checkpoints are retained at 65,536; 262,144; 1,048,576; 1,572,864; and 2,097,152 targets when reached. The workflow ends the first process exactly at 1,048,576 targets, then launches a separate Python process, verifies the D05 identity, restores model/optimizer/RNG/counters, re-evaluates the held-out split, requires validation drift <=1e-12, repeats the deterministic generation snapshot, and only then continues.

The uploaded Actions artifact retains the complete checkpoint directories, not only their manifests.

## Defined early-stop conditions

The run does not stop merely because a curve looks flat by eye. Clean early stop is allowed only for explicit rules:

- divergence: pre-clip global gradient norm >100 for eight consecutive committed updates;
- divergence: scheduled held-out validation loss exceeds the random-init validation loss by more than 2.0 nats;
- no improvement, only at or after 1,572,864 optimized targets: the best validation improvement across the latest four scheduled evaluations is <1e-4 nats.

Non-finite training remains a fail-closed Trainer error rather than a successful early-stop result.

## Truth boundary

This is Base next-token training from random initialization. It adds no instruction/chat/system behavior, foreign pretrained weights, post-training, tokenizer freeze, corpus approval, canonical stage change, promotion, capability claim, or paid compute authorization.

The exact-head `TRAIN-41 Long 100K Base` workflow is the execution authority. A queued or failed workflow is not evidence of a completed long run.
