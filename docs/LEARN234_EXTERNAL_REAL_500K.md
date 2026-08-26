# LEARN-234 External-Real 500K

## Current decision

`BLOCKED_NO_TERMINAL_DATA230`

LEARN-234 may start optimizer step 1 only after `DATA-230-CORPUS-V03-EXTERNAL-REAL` is published as a terminal deterministic authority with an exact corpus identity and deterministic final shard inventory. At this cutoff no DATA-230 PR, branch, or commit is published in the live repository. No older DATA-183/DATA-110 corpus is substituted.

`EVAL-233-REAL-HOLDOUT-V2` is also not published at this cutoff. Its absence does not independently authorize or block training; when available, its authorized real external holdout must be consumed without exposing final-test material to checkpoint/tokenizer/hyperparameter selection.

No training was started. Optimizer updates are exactly zero. No checkpoint or model-quality result is claimed.

## Frozen incumbent before DATA-230 arrives

The strongest accepted comparable ~500K scratch-Base geometry remains the MILESTONE-150/MILESTONE-221 467,808-parameter control: D96, L4, 6 MHA heads / 6 KV heads, head dimension 16, SwiGLU FFN 256, vocab 256, ModelSpec `208ac8ca113388e76f280d0154cae815785bee7705546f4d854d9447b9dd1f4a`, common InitSpec `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`.

The tokenizer incumbent at this cutoff is canonical `s0-byte-v1`, config `b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`, vocab `905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`, 256 ordinary byte IDs and no special tokens.

The retained 500K optimizer control is AdamW LR 3e-4, betas 0.9/0.95, eps 1e-8, weight decay 0, constant schedule, no warmup, global clip 1.0, FP32, document-isolated seq128, batch8, seed1337. LEARN-234 must re-resolve these authorities immediately before training and may not retain them if a newer exact accepted authority supersedes them.

## Preregistered budget rule

The requested matched DATA-25 anchor is 948,504 actual optimized targets. The external-real budget is fixed before observing validation quality as:

`min(948504, DATA230_one_pass_unique_train_optimized_targets)`

Artificial corpus repetition is forbidden. If DATA-230 supplies fewer than 948,504 one-pass unique optimized targets, LEARN-234 stops at the one-pass supply ceiling and records the unmatched DATA-25 comparison limitation rather than recycling documents.

Selection-validation checkpoints are preregistered at 0%, 25%, 50%, 75%, and 100% of the realized budget. Best is the minimum frozen selection-validation aggregate BPB; chronological final remains separately retained even when it is not best.

## Required execution evidence after unblock

The run must start from random initialization only and report selection-validation aggregate, UA, EN, code and source-family BPB; EVAL-233 real external holdout when compatible and available; gradient norms; clip rate; update/parameter ratio; optimized-token throughput; memory; checkpoint integrity; fresh-process resume; and memorization/generalization diagnostics. Evaluation must not mutate model or Trainer state.

The DATA-25 matched control anchor is the terminal MILESTONE-150 producer at source `5838cd16869dcfcf762368d8673eddf52d51b7e3`, workflow run `32937411703`, artifact `9595677772`, archive SHA-256 `c00b7e9006320f8916c739a3311e8cc47ad0d0b16957f8ebd7d19233fd9f1c71`, corpus `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`, 948,504 optimized targets and 500K best/final BPB `0.2645455968814711`.

## Execution boundary

The dedicated workflow runs the universal execution bootstrap only to validate this authority gate. It performs no model construction, optimizer creation, checkpoint loading, corpus substitution, or training while DATA-230 is absent.

No foreign weights. No SFT, RLHF, or DPO. No paid compute. `LOCAL_FREE` only.
