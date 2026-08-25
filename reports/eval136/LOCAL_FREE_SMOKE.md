# EVAL-136 LOCAL_FREE smoke evidence

Status: `EXECUTED_LOCAL_FREE_NON_PROMOTION_AUTHORITY`

Truth boundary: the execution environment could not obtain an exact Git checkout because external GitHub DNS was unavailable. The run used a connector-reconstructed local mirror of integration SHA `fb9c6d9b73ce436d637077892d73edf136fcaeac`, with the exact model architecture/training interfaces required by this experiment and the project-owned S0 train/validation text fields. Local JSONL metadata wrappers and manifest were not byte-identical to the repository package. Therefore these numbers are diagnostic evidence only and must be rerun on an exact checkout before promotion or canonical scaling claims.

No canary text or ordinary training text is included below.

## Joint onset

| Model | Parameters | First joint validation + disproportionate-memorization event |
| --- | ---: | --- |
| S1 ~100K | 107,856 | 2,060 tokens, T/N 0.01910 |
| EVAL-136 midpoint ~500K | 492,384 | 8,239 tokens, T/N 0.01673 |
| S2 ~1M | 1,066,112 | Not reached through 8,252 tokens, T/N 0.00774 |

The joint event requires held-out BPB to improve from the preceding checkpoint and at least two independent repeated-vs-unseen memorization signals. It is not a privacy-leakage threshold.

## S1 ~100K

Checkpoint summary:

| Tokens | T/N | held-out BPB | memorization index | non-canary train-probe NLL | stop |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.00000 | 8.9858 | 0.1330 | 6.2333 | no |
| 2,060 | 0.01910 | 7.7876 | 1.0040 | 5.8213 | yes |
| 8,380 | 0.07770 | 6.5702 | 3.7535 | 4.8955 | yes |

Exposure curve at 8,380 tokens (`configured exposure/cycle -> observed mean exposures, median NLL, median rank percentile, exact recovery`):

- 0 -> 0.00, 4.248, 0.500, 0.000
- 1 -> 1.00, 4.561, 0.750, 0.000
- 2 -> 3.00, 3.757, 0.125, 0.000
- 4 -> 6.33, 2.593, 0.125, 0.000
- 8 -> 11.67, 2.096, 0.125, 0.000
- 16 -> 23.33, 1.203, 0.125, 0.333

## Experimental ~500K midpoint

Checkpoint summary:

| Tokens | T/N | held-out BPB | memorization index | non-canary train-probe NLL | stop |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.00000 | 9.0790 | 0.0000 | 6.2722 | no |
| 2,082 | 0.00423 | 10.6968 | 1.0784 | 5.9190 | no |
| 8,239 | 0.01673 | 5.0037 | 2.9102 | 4.7626 | yes |

At 2,082 tokens the repeated canaries already showed NLL/rank separation, but held-out BPB worsened. The joint policy therefore did not stop. This is an intentional guard against interpreting memorization alone as a reason to stop a run whose ordinary validation behavior has not improved.

Exposure curve at 8,239 tokens:

- 0 -> 0.00, 4.270, 0.375, 0.000
- 1 -> 1.00, 4.520, 0.875, 0.000
- 2 -> 2.33, 4.143, 0.250, 0.000
- 4 -> 6.00, 3.096, 0.125, 0.000
- 8 -> 11.00, 2.755, 0.125, 0.000
- 16 -> 23.67, 1.610, 0.125, 0.000

## S2 ~1M

Checkpoint summary:

| Tokens | T/N | held-out BPB | memorization index | non-canary train-probe NLL | stop |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.00000 | 11.1574 | 0.0000 | 7.7381 | no |
| 2,061 | 0.00193 | 9.5708 | 1.1525 | 6.7753 | no |
| 8,252 | 0.00774 | 4.9532 | 1.7762 | 4.3962 | no |

Exposure curve at 8,252 tokens:

- 0 -> 0.00, 4.014, 0.125, 0.000
- 1 -> 1.00, 4.823, 0.750, 0.000
- 2 -> 2.33, 4.356, 0.250, 0.000
- 4 -> 5.00, 4.737, 0.500, 0.000
- 8 -> 10.33, 3.295, 0.125, 0.000
- 16 -> 21.00, 2.238, 0.125, 0.000

Held-out BPB improved strongly, but only the NLL separation signal fired at the final smoke checkpoint; rank and exact-recovery controls did not establish disproportionate memorization. The diagnostic therefore remains below the stop threshold.

## Safety and interpretation

The randomly sampled non-canary training probe emitted only hashes/aggregate metrics in the machine implementation; exact short-continuation recovery was 0 in this smoke run. Evaluation was verified non-mutating at every checkpoint by hashing model state and checking Trainer counters before and after scoring.

These observations support using `eval136-small-experiment-stop-diagnostic-v1` as a conservative small-experiment diagnostic: stop or investigate when held-out BPB is still improving and at least two control-relative memorization signals fire. They do not establish a universal privacy threshold, do not imply that memorization equals data leakage, and do not authorize extrapolation beyond the measured exposure/T-N range.
