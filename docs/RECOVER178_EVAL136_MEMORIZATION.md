# RECOVER-178 — EVAL-136 memorization authority recovery

`RECOVER-178-EVAL136-MEMORIZATION` is a convergence recovery stacked on `MILESTONE-150-LEARNED-BASE-LADDER-V1`. It does not redefine the model family, tokenizer, corpus, Trainer, held-out evaluator, or the EVAL-136 canary scorer.

## Live-state decision

At recovery start, EVAL-136 exact head `73b8c811d0ecb1ff424e73f6c689ad290d142ea5` did not have an exact-green 100K/500K/1M memorization matrix: the ordinary CI/training/preflight runs on that head were terminal failures. Its local report was therefore source-shaped smoke, not authority evidence. The conditional 10M-only extension was not activated.

Recovery work started from MILESTONE-150 head `d4758d0ce7f8821ef10e4eee666648b51cc0247c`. The RECOVER-178 branch is rebased onto the current MILESTONE-150 convergence head before exact execution. Runtime reports bind the exact checked-out recovery SHA rather than treating the recovery-start SHA as final authority. The exact EVAL-136 `src/twelve_six/memorization.py` blob is reused unchanged.

## Comparable truth model

All three trajectories use the M150 incumbents:

- DATA-25 corpus identity `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`;
- canonical `s0-byte-v1`, 256 bytes, no special tokens;
- the same held-out validation and `uk` / `en` / `code` evaluation identity;
- document-isolated seq-128 packing;
- exact M150 models: 95,568 / 467,808 / 1,037,696 parameters;
- AdamW M150 optimizer/run semantics and model-init seed 1337;
- 1,000 optimizer steps and checkpoints at 0 / 250 / 500 / 750 / 1000.

EVAL-136 seed `20260826` remains scoped to the synthetic canary suite and exposure schedule. It does not replace the M150 model-init seed.

## Exposure contract

The incumbent canary levels remain `0, 1, 2, 4, 8, 16`, with three replicas per level and six-byte continuations. Control level 0 is structurally absent from training.

One exposure cycle is 100 optimizer steps. The 93 non-control canary records implied by the incumbent suite are deterministically shuffled into those 100 positions. At a canary event, exactly one row of the incumbent B=8 DATA-25 batch is replaced by the synthetic canary record. The other seven rows remain DATA-25. Reports retain actual observed exposure counts rather than inferring them from targets.

`corpus_repetition_count` is reported at every checkpoint as the number of optimized DATA-25 packed examples divided by one complete packed DATA-25 train corpus, with consumed/optimized counts and per-stratum fractions retained separately.

## Metrics and non-mutation

Every checkpoint records:

- common held-out BPB and the M150 UA/EN/code breakdown;
- continuation NLL and mean log-likelihood;
- geometric-mean token likelihood;
- matched-alternative rank;
- exact short-continuation recovery;
- actual canary exposure count;
- DATA-25 corpus repetition count;
- model-state SHA-256;
- Trainer counters;
- explicit evaluation non-mutation PASS.

Evaluation must preserve model weights, Trainer counters and the model train/eval mode. Any mutation aborts the run.

## Stop-policy binding

The methodology remains `eval136-small-experiment-stop-diagnostic-v1`. The configuration predeclares its floors and signal rule. For each scale, the realized NLL, top-decile-rank and exact-recovery thresholds are bound from the random-init control curve **before optimizer step 1** and written to `threshold-binding.json`. All later stop diagnostics consume only those frozen values. A threshold cannot be recomputed from a later checkpoint.

The diagnostic remains an experiment stop signal only. It is not a privacy-leakage test and grants no model-promotion authority.

## Safe reporting

Public evidence emits no canary/source text. Canary prefixes and continuations are represented only by existing EVAL-136 hashes and aggregate metadata. A recursive public-output guard rejects raw `text`, `prefix`, `continuation`, `source_text` or `canary_text` fields before reports are written/finalized.

No claim is made about privacy leakage, intelligence, production readiness, alignment, instruction following, or broad memorization thresholds. No foreign pretrained weights, SFT, RLHF, DPO, or paid compute are introduced.

## Environment and execution

The exact-head workflow first creates a clean environment through `tools/bootstrap_universal_environment.py` from the committed Linux x86-64 toolchain, runtime and **dev** hash locks. No project test runs before that exact dev environment exists. The workload is CPU execution only and labeled `LOCAL_FREE_CPU`.

Authoritative output is `recover178-evidence/memorization-report.json`. It finalizes only if 100K, 500K and 1M share the same corpus/evaluation identity and every checkpoint passes evaluation non-mutation.
