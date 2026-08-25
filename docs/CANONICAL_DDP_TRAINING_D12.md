# D12 canonical token-correct DDP execution

## Why this exists

D12 already defines distributed topology, rank identity, checkpoint-layout and backend-adoption contracts. The earlier LOCAL_FREE evidence did not train a canonical `TwelveSixDecoder` through a real process group.

A naive `DistributedDataParallel` wrapper is also numerically wrong for the current D02 token-accounting rule when ranks contain different numbers of valid targets. D02 intentionally backpropagates `mean_loss * local_valid_tokens` and normalizes accumulated gradients by the local valid-token count. DDP averages gradients across ranks before that local normalization. If local token counts differ, the result is not the gradient of the global token-mean loss.

## Contract

For rank `r`, let `n_r` be its valid target count, `L_r` its mean loss over those targets, `N = sum_r n_r`, and `W` the DDP world size. DDP averages rank gradients, so each rank must backpropagate:

`L_r * n_r * W / N`

After DDP's average, the resulting gradient is:

`sum_r grad(L_r * n_r) / N`

which is the same token-weighted objective as one single-process batch containing all rank examples.

`src/twelve_six/distributed/ddp_training.py` implements this rule as an explicit execution adapter. It does not silently reinterpret the single-process D02 Trainer.

## Real proof

The probe deliberately gives ranks unequal valid-token counts by masking different target tails. It then requires all of the following in one real CPU/Gloo process-group execution:

- canonical scratch/random-init `TwelveSixDecoder` construction from an exact stage config;
- two real DDP ranks;
- explicit global valid-token all-reduce;
- one real backward and AdamW optimizer step;
- exact parameter synchronization across ranks after the update;
- non-zero parameter change;
- a fresh single-process model initialized from the same seed;
- one reference optimizer step on the concatenated global batch;
- distributed-vs-reference parameter agreement within the declared tolerance.

The dedicated locked workflow runs the regression suite and then executes this proof on committed S2 geometry: 1,066,112 parameters, two CPU/Gloo ranks, one optimizer step. The controlled synthetic token pattern exists only to test mechanics; it is not an S2 corpus or tokenizer selection.

## What this unlocks

This closes the gap between distributed topology prose/contracts and a real canonical model update. The same token-weighted rule is reusable when later data packing produces unequal valid-token counts across ranks. It gives future GPU/NCCL, FSDP2/HSDP and larger-stage work a correctness reference instead of assuming that equal-size batches prove distributed objective equivalence.

## Truth boundary

This evidence is `LOCAL_FREE_DISTRIBUTED_MECHANICS_NOT_STAGE_OR_CAPABILITY_EVIDENCE`.

It does not claim GPU/NCCL execution, throughput, MFU, multi-node behavior, fault tolerance, FSDP2 parameter sharding, TP/PP/CP execution, distributed checkpoint writing, corpus quality, tokenizer freeze, S2 capability, paid-compute authorization, CANDIDATE/STABLE promotion or an audit verdict.

Canonical Base remains random-initialized and pretraining-only. No foreign pretrained weights and no instruction/alignment/refusal/personality/domain-specialization layer are introduced.
