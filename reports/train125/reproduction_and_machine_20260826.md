# TRAIN-125 LOCAL_FREE reproduction and machine manifest

## Practical transfer rule

Measured on the fixed byte-tokenizer / bounded real DATA-21/22 recipe with AdamW betas=(0.9,0.95), eps=1e-8, WD=0, warmup=0, constant LR, clip=1, fp32, seed=1337, batch=4, seq=64:

`lr_best ~= 1.81265e-3 * (parameters / 100000)^(-0.461414)`

Measured best values: 95,568 -> 1.8e-3; 267,912 -> 1.2e-3; 467,808 -> 9e-4; 1,037,696 -> 6e-4. Leave-one-scale-out maximum relative prediction error was 12.9%; every measured optimum fell inside predicted LR +/-15%.

Preregistered ~10M candidates: `1.6e-4`, `2.2e-4`, `3.0e-4`. This range is UNPROVEN until a real ~10M run executes.

## Local milestone reproduction

The retained LOCAL_FREE evidence bundle contains `train125_local.py`, the exact DATA-21/22 accepted-text artifact, candidate JSON results, all five milestone checkpoints, and reports. From the extracted bundle root:

```bash
python train125_local.py candidate --scale 100k --lr 0.0018 --steps 128 --data data --out lr_results
python train125_local.py candidate --scale 250k --lr 0.0012 --steps 128 --data data --out lr_results
python train125_local.py candidate --scale 500k --lr 0.0009 --steps 128 --data data --out lr_results
python train125_local.py candidate --scale 1m --lr 0.0006 --steps 128 --data data --out lr_results
rm -rf milestone
python train125_local.py milestone phase1 --data data --out milestone
python train125_local.py milestone resume --data data --out milestone
```

Phase 1 starts from random initialization and writes checkpoints 0/250/500. The second command is a fresh Python process, restores step 500 and continues at step 501, writing checkpoints 750/1000.

## Local machine manifest

- Python: 3.13.5 CPython, GCC 14.2.0
- PyTorch: 2.10.0+cpu
- Platform: Linux 6.18.35 x86_64, glibc 2.41
- CPU: 5 logical / 5 physical cores visible to the container
- RAM: 6,368,833,536 bytes
- torch threads: 2
- CUDA available: false
- paid compute: false

## Invalidation conditions

Direct LR transfer is invalid if tokenizer identity, corpus/source-family recipe, optimizer betas/eps/WD, warmup/scheduler, clip norm, precision, sequence length, effective batch, or token budget changes. Re-preregister if a future best LR lands on a search boundary; if loss/gradients become non-finite; if early loss spikes >25%; if mean update/weight ratio reaches 0.02; if leave-one-scale-out error rises above 25%; if the exponent shifts by >0.15; if model topology changes materially; or if held-out BPB fails to improve despite train-loss decrease.

## Truth boundary

TRAIN-42 itself is rejected as optimizer evidence because its two live LR-range workflow runs failed before the sweep. The valid optimizer center came from successful adjacent stability/optimizer-dynamics evidence. DATA-21/22 is real, rights-approved external training data, but only a bounded two-family sample (Verkhovna Rada legislation + Standard Ebooks manual), not broad representative-corpus evidence. The network-isolated local runtime used a source-equivalent mirror of the selected architecture/AdamW semantics and a local checkpoint envelope. This branch is based on the incumbent MILESTONE-100 composition that imports project Trainer/D05/inference, but the hosted exact-head MILESTONE-100 run was still queued when this evidence was recorded; no queued PASS is claimed.
