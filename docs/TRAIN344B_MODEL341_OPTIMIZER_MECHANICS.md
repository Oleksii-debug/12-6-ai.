# TRAIN-344B — exact MODEL-341 optimizer mechanics

## Decision

TRAIN-344B removes the stale `BLOCKED_MISSING_20M_MODELSPEC` condition from the original TRAIN-344 contract only for bounded synthetic mechanics. It does not remove the Research Corpus V1 or D05 checkpoint-integrity gates and does not authorize a learned 20M campaign.

The exact target is MODEL-341 at `e4ff486fd90802fc123bebf60eed4e59196a98df`: 20,613,440 random-init parameters, ModelSpec `fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`, InitSpec `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`.

## Hardening over TRAIN-344

The predecessor accepted any repository ModelSpec inside an 18M–22M parameter window. That is insufficient for a durable scale-transfer claim because a different 20M geometry could pass the gate. TRAIN-344B binds all of the following simultaneously:

- exact MODEL-341 ancestor SHA;
- exact candidate config path;
- exact Git blob SHA-1 and content SHA-256;
- exact parameter count;
- exact ModelSpec and InitSpec identities;
- vocab 256 and context 1024;
- D320 / L16 / 10Q / 2KV / head32 / FFN1080 geometry.

Any drift fails before optimizer execution.

## Frozen optimizer probe

The original preregistration remains unchanged: AdamW, LR arms 1.6e-4 / 2.2e-4 / 2.6e-4, betas 0.9/0.95, eps 1e-8, weight decay 0.1, clip 1.0, constant schedule, no warmup, sequence 256, microbatch 1, accumulation 1, deterministic FP32 CPU.

Each arm receives identical random initialization and an identical deterministic synthetic token trace. Each arm executes exactly 32 optimizer updates / 8,160 causal targets; total bounded mechanics exposure is 96 updates / 24,480 synthetic causal targets. The only positive label permitted is `STABLE_MECHANICS_ONLY`. No arm may be selected as a quality winner from this experiment.

## Remaining hard blockers

Research Corpus V1 is still not materialized and current learned-corpus authorization remains zero. D05 checkpoint integrity remains `RETEST_REQUIRED` while a separate remediation converges the corruption matrix. TRAIN-344B therefore records zero learned-corpus updates, no long training, and no paid compute. Its mechanics evidence must be refreshed or re-bound after data and D05 terminalization before it can participate in a learned-campaign launch record.
