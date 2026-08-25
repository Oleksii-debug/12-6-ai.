# RESEARCH-06 fixed-token scaling + MODEL-10 ~1M GQA

This successor composes the exact terminal-green RESEARCH41 controlled-scaling snapshot (`9ff78ea31c34fd434015d5bc512596ce5dac766a`) with the MODEL-35 native-GQA/KV-cache incumbent (`75a885e221fdf16f647289df25a7150ef46e7528`). Canonical S0 is not modified.

## Scientific control

The four MHA scaling points remain 95,568 / 267,912 / 467,808 / 1,037,696 trainable parameters. Every candidate uses the same `s0-byte-v1` tokenizer, vocabulary 256, max context 256, training sequence length 64, project-authored S0 train/validation split, normal initialization family, seed 1337, CPU FP32, and AdamW recipe inherited from RESEARCH41.

RESEARCH41 previously observed token checkpoints only after a full 252-target optimizer batch, so nominal 4,096 / 16,384 / 65,536 budgets became 4,284 / 16,632 / 65,772 optimized tokens. RESEARCH-06 replaces only that accounting boundary: aligned causal targets plus a binary `loss_mask` permit the final optimizer step of each budget segment to contain exactly the remaining number of valid next-token targets. Evaluation targets are recorded separately and never mutate `Trainer.tokens_seen`.

The 1M attention comparison reuses the fixed-control 1,037,696-parameter 8Q/8KV MHA model. The GQA comparator keeps D=128, L=5, 8 query heads and head_dim=16, reduces to 4 KV heads, and reallocates projection savings into `d_ff=395`, yielding 1,038,336 parameters (0.0617% above MHA). Data trace, tokenizer, context, initialization and optimizer are unchanged.

## Resume and fail-closed evidence

Each candidate checkpoints at exactly 16,384 optimized tokens using the existing D05 model+Trainer checkpoint adapter, reconstructs fresh model and Trainer objects, verifies all run/data/tokenizer/model identities, and resumes to 65,536. A second uninterrupted trajectory consumes the same exact masked batch trace; final model tensors and full Trainer/optimizer state must be bitwise equal or the run fails.

The report validator rejects token overshoot, evaluation-token contamination, compute-proxy drift, non-finite held-out loss, resume inequality, source-SHA drift and weakened truth boundaries.

## Measurements and ranking

The exact-head workflow retains per-candidate JSON plus resume checkpoints and an aggregate self-hashed report. It records held-out loss and byte-token BPB, `6*N*T` compute proxy, training/evaluation/checkpoint wall time, fresh-process RSS high-water mark, model/optimizer tensor bytes, gradient norm distribution, clipping frequency, parameter movement and relative update ratio.

Scaling candidates are ranked independently by final held-out validation loss, held-out improvement per parameter, held-out improvement per compute proxy, and held-out improvement per primary-trajectory wall second. Train loss is diagnostic only and is not a generalization ranking input.

For a working small-model research vehicle, the report uses an explicit knee heuristic: select the smallest MHA candidate whose final held-out loss is within 5% of the best observed MHA loss. This is a research recommendation, not an architecture freeze.

MODEL-10 additionally measures parameter allocation, full-context KV-cache bytes, exact cached-vs-stateless greedy generation, training loss, gradients and step time. On CPU the inherited grouped-attention path is the explicit expanded-KV reference fallback; CPU timing is therefore not evidence of GPU GQA speed.

## Truth boundary

The corpus is the deliberately tiny recycled project-authored S0 fixture. Held-out measurements are genuine for its isolated validation records, but they are not broad language-model quality evidence. No paid compute, GPU performance claim, stage promotion, tokenizer/corpus freeze, or canonical S0 change is authorized by this experiment.
