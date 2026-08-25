# TRAIN-126 Weight-Decay Transfer — LOCAL_FREE evidence

Base repository: `Oleksii-debug/12-6-ai.`
Base SHA: `dcc7dfc39299487bca5bdbfe5e6c70eaa6706278`
Experimental branch: `train126/wd-transfer-real-corpus-20260826`

## Truth boundary

DATA-21/22 is a successful rights-approved real bounded UK/EN sample, not a canonical broad representative UK/EN/code corpus; DATA-23 code intake head failed. Local execution used a source-parity adapter rather than importing a git checkout, although 500K checkpoint-key/generation parity with retained LEARN03 evidence was verified.

## Fixed identities

- Tokenizer: `s0-byte-v1`, UTF-8 byte, vocab 256.
- Data: DATA-21/22 accepted external-source intake, 173,358 normalized UTF-8 bytes, 3 accepted records, UK+EN, rights approved.
- Data trace SHA256: `648a2414cf4cbeab51b59301a852e0c38874d8b7055bbfd6f61662db0b134969`.
- 500K geometry: 467,808 parameters.
- 1M geometry: 992,896 parameters.
- AdamW: lr=3e-4, betas=(0.9,0.95), eps=1e-8, clip=1.0, fp32.
- TRAIN-44 grouping: token embedding decay fixed at 0; all non-embedding parameters use tested coefficient.
- Grid: 0, 0.01, 0.1.
- 1,024 optimizer steps; batch 4; sequence 64; 252 scored tokens/step; eval 0/256/512/768/1024.
- Every run stops at 512 and resumes in a fresh process.

## Final results

| Scale | WD | Train BPB | Held-out BPB | Gap BPB | Embedding L2 | Block L2 | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| 500k | 0 | 2.796260 | 2.845061 | 0.048801 | 16.7433 | 16.0506 | ACCEPT |
| 500k | 0.01 | 2.798659 | 2.847379 | 0.048720 | 16.7661 | 16.0251 | REJECT |
| 500k | 0.1 | 2.820711 | 2.868088 | 0.047377 | 16.9659 | 15.8014 | REJECT |
| 1m | 0 | 2.505092 | 2.572581 | 0.067490 | 16.8553 | 21.3119 | ACCEPT |
| 1m | 0.01 | 2.512942 | 2.573797 | 0.060855 | 16.8537 | 21.2691 | REJECT |
| 1m | 0.1 | 2.503476 | 2.587760 | 0.084285 | 16.8766 | 20.9201 | REJECT |

## Decision

Provisional coefficient rule: **WD=0.0 at 500K and ~1M**. Positive decay is rejected because it does not improve held-out BPB. For 10M, keep **0.0 as default** and test **0.01 only as a narrow challenger**; do not carry 0.1 forward without new evidence.

At 500K, WD=0.1 lowers block L2 from 16.0506 to 15.8013 but worsens held-out BPB from 2.8451 to 2.8681. At ~1M, WD=0.01 yields 2.5738 vs 2.5726 at zero; WD=0.1 yields 2.5878. This is regularization cost without held-out benefit.

## Proof checks

- Random initialization: yes; initial greedy generation is spaces for both scales.
- Exact parameter count: yes, 467,808 and 992,896.
- Train and held-out BPB decrease: yes for all conditions.
- Multiple checkpoints: steps 256/512/768/1024 retained for all six conditions.
- Fresh-process resume: yes; phase PIDs differ in all six runs.
- Evaluation non-mutation: yes; tensor fingerprint assertion passed at every train/validation evaluation.
- Generation before/after: yes; initial space-only generation changes to learned repetitive byte-language patterns.
- Exact retained winners: `checkpoints/500k-wd0-step1024.pt`, `checkpoints/1m-wd0-step1024.pt`.
- No CUDA / paid compute: CPU-only PyTorch 2.10.0+cpu.

## Rejections

- TRAIN-44 empirical result is not accepted: its workflow failed Ruff I001 before comparison execution. Only its parameter-group semantics are reused.
- LEARN03 corpus is not accepted for WD transfer: 10 train records / 1,930 unique bytes, repeated fixture.
- DATA-23 code intake is not accepted: published head has failed intake workflow and failed CI.
- MILESTONE-100 broad representative-corpus claim remains open; this experiment is a bounded real-corpus optimization result, not a broad Base promotion.

## Reproduction

```bash
python train126_experiment.py orchestrate --data-root external-source-intake-evidence --out-dir train126_run/checkpoints --summary train126_run/train126-wd-transfer-summary.json --total-steps 1024 --mid-step 512 --eval-steps 0 256 512 768 1024
```

Adapter SHA256: `637c618d7ef54029aefa7b8db5962711b0b04c0069e27eb8d3e788b6fd8f6054`
