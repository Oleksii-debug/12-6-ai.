# RESEARCH-192: clean 1M -> 3M -> 10M scaling transfer

## Scientific question

Measure the quality/efficiency transfer from approximately 1M to 3M to 10M trainable parameters while holding the complete non-size learning recipe fixed. This is a controlled DATA-25 scaling study, not a broad-corpus scaling-law claim.

## Incumbent and transfer decision

The terminal MILESTONE-150 producer is retained as the accepted 1M incumbent control:

- source `5838cd16869dcfcf762368d8673eddf52d51b7e3`;
- workflow run `32937411703`, terminal success;
- artifact `9595677772`, digest `sha256:c00b7e9006320f8916c739a3311e8cc47ad0d0b16957f8ebd7d19233fd9f1c71`;
- ladder report identity `1f8350bed574a7b78778f0ebb7854ca5311173006820ec27110122f8965c9a5a`;
- 1M report identity `1b63e8f5096c43b9a36923ddd9d4b8d8a8d1705559f63080c0a287c5520fc738`.

RESEARCH-192 reruns the 1M seed-1337 trajectory only because the requested transfer table needs end-to-end checkpoint wall time under the same execution campaign and the ambiguous 1M/3M neighbor requires a paired repeat. The accepted M150 artifact is downloaded, digest-checked and used as a bound reproducibility control rather than discarded.

The existing SCALE-141 10M runtime is not a comparable point here. It changes weight decay, sequence length, microbatch geometry and S3 architecture/context. Importing its quality number would confound scale with optimizer/runtime/architecture transfer.

## Frozen recipe

All arms share:

- DATA-25 corpus identity `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`;
- canonical `s0-byte-v1`, vocab 256, no special tokens;
- MILESTONE-150 held-out evaluation identity `7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113`;
- document-isolated seq-128 packing and batch 8;
- AdamW, LR 3e-4, betas 0.9/0.95, eps 1e-8, weight decay 0;
- constant schedule, no warmup, gradient clip 1.0;
- FP32 deterministic CPU execution;
- same deterministic DATA-25 optimizer-step trace;
- random initialization with the common InitSpec `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`;
- fresh-process resume at optimizer step 500.

Seed is the only deliberately repeated nuisance variable. The data trace stays fixed while model/trainer initialization uses the preregistered seed.

## Exact scale family

| scale | parameters | fixed-control ModelSpec | geometry |
|---|---:|---|---|
| 1M | 1,037,696 | `ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07` | D128 L5 H8/KV8 HD16 FF352 |
| 3M | 3,221,184 | `3255ebffea76d17e59a19b4de50be616b27e85593a6eebec0db935d7efebb5ea` | D192 L7 H12/KV12 HD16 FF530 |
| 10M | 10,000,640 | `f01cf22d3a44bd72be74691ca4b4a75b093851f45fc2b252c5116eb72370dc53` | D256 L12 H16/KV16 HD16 FF736 |

All three use MHA, pre-RMSNorm, RoPE, SwiGLU, tied byte embeddings, no biases and max ModelSpec context 256. Training sequence length remains 128. The 3M bridge is the RESEARCH-138 high-information interpolation target transferred into the MILESTONE-150 family. The 10M control deliberately matches the existing S3 parameter count while not importing S3 GQA/context changes.

## Common token checkpoints

The comparison is made only at exact actual optimized-target counts already observed in the M150 deterministic trace:

- optimizer step 500: 474,377 optimized byte targets;
- optimizer step 1000: 948,504 optimized byte targets.

Every arm must hit both counts exactly. Any mismatch is classified as a hidden token advantage and aborts comparison. Evaluation tokens never enter optimized-token accounting.

## Seed plan

The most ambiguous neighboring pair is 1M -> 3M, because RESEARCH-138 identified the ~3.2M bridge as the most informative next interpolation point. RESEARCH-192 therefore runs paired seeds 1337 and 1338 for both 1M and 3M. The 10M control runs seed 1337. Two paired seeds are descriptive evidence only; they do not satisfy the separate RESEARCH-140 three-repeat promotion rule.

## Reported quantities

At both common token checkpoints each arm records:

- aggregate held-out BPB;
- UK/EN/code held-out BPB;
- online training BPB from that optimizer-step minibatch;
- held-out minus online-training BPB generalization gap;
- `6*N*T` compute proxy using actual optimized tokens;
- end-to-end wall time through the checkpoint, including scheduled evaluation/checkpoint overhead;
- FP32 trainable parameter bytes and measured process RSS peak;
- optimized tokens per end-to-end second;
- checkpoint gradient norm.

Pairwise rows additionally report held-out BPB improvement per added parameter and per incremental `6*deltaN*T` compute. Positive improvement means the larger scale has lower held-out BPB.

## Execution integrity

The dedicated workflow uses the ENV-151 universal exact execution bootstrap with the committed CPU-runtime/test capability closure and Python 3.11.16. Every stochastic arm runs prepare, phase1, fresh Python resume, retained-checkpoint verification and arm summarization as separate commands. The final comparison job refuses incomplete arms, non-size recipe drift, corpus/tokenizer/evaluation drift, token-budget mismatch or incumbent-artifact identity mismatch.

## Truth boundary

LOCAL_FREE CPU only. No paid compute. No foreign pretrained weights. No SFT, RLHF or DPO. No stage promotion, production-readiness, instruction-following, intelligence, external-corpus representativeness, Chinchilla or universal scaling-law claim.
