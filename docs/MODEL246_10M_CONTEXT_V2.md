# MODEL-246 — 10M Context V2

Worker: `MODEL-246-10M-CONTEXT-V2`

## Current execution result

`BLOCKED_MISSING_TERMINAL_OPTIMIZER_AND_DATA_AUTHORITIES`

No MODEL-246 optimizer update was executed. This is intentional fail-closed behavior, not an unsuccessful numerical arm.

At the execution cutoff the repository did not publish a terminal `TRAIN-243-10M-CLIPPING-AUTHORITY-V2` or `TRAIN-244-10M-LR-BETA-V2` branch/PR/commit/artifact. The visible `data230/corpus-v03-external-real-20260826` branch points to `6d994e2aece6c44e28c1a2c344ac98b5a8fd5e08`, whose commit message is `DATA-214 restore retained quality and privacy evidence`; it therefore cannot be treated as terminal DATA-230. `TRAIN-245-10M-EFFECTIVE-BATCH-V2` is also not published at this cutoff, so MODEL-246 does not guess an effective batch unless terminal TRAIN-244 explicitly freezes it.

## Historical MODEL-197 evidence

The prepared predecessor is real executed evidence, not a plan. Exact head `df224d14b11099f3a36cebc5372bb5a869c37ec2`, Actions run `32940725569`, concluded `success`. Artifact `9598153241` has archive digest `sha256:6e407c7a81ffa02e668190755401c9f39f8331e058ce553f3e617d15e7776722`.

Its single-seed old-recipe result was `KEEP_256_NO_LONGER_CONTEXT_BPB_GAIN`:

| Horizon | Common-short BPB | Native BPB | Clip rate | CPU optimized tok/s | CPU max RSS |
|---:|---:|---:|---:|---:|---:|
| 256 | 1.0927784593 | 1.0927784593 | 1.000000 | 1584.159 | 640,598,016 B |
| 512 | 2.0152682730 | 1.2074882416 | 0.984375 | 1550.415 | 732,655,616 B |
| 1024 | 2.2955104382 | 1.4057391072 | 0.937500 | 1530.570 | 821,948,416 B |

The old 256→512 common-short paired-document delta was +0.92249 BPB and 256→1024 was +1.20273 BPB, both regressions. However, this result is not eligible to answer MODEL-246 because almost every update clipped, the optimizer/data contract is the one MODEL-246 is instructed to supersede, and the predecessor used one training seed.

## Frozen V2 comparison once authorities are terminal

The scientific variable remains only trained within-document context: 256, 512, 1024. The exact 10M parameter geometry must be identical across arms. Keeping `max_seq_len=1024` in the ModelSpec is allowed; it is mechanical support only and cannot be called learned long-context capability.

Every arm must consume the same ordered source documents/source bytes. Each causal target must be optimized exactly once. The same source-content group must define each optimizer update across horizons. A valid implementation may split one document into more windows at shorter horizon, but summed NLL must be normalized by the exact causal-target count before the common optimizer update so longer contexts receive neither extra source bytes nor extra optimizer exposure.

The terminal optimizer authority must bind LR, betas, epsilon, weight decay, clipping, schedule and effective batch/accumulation semantics. Tokenizer and corpus identities must be byte-for-byte identical across arms.

Required reports are: common short-horizon BPB at 256, native-horizon BPB, reserved dependency-distance probes with matched short controls, padding/right-trim utilization, actual loss tokens, exact optimizer-update count, CPU wall time/throughput and process RSS, plus gradient-health telemetry. CPU systems numbers are CPU-specific and must not be extrapolated to GPU.

No default context may move away from 256 on a single seed. A replacement requires multiple paired seeds under one immutable terminal optimizer/data/tokenizer contract and positive evidence on learned dependency behavior, not RoPE/KV reach alone.

## Machine evidence

`evidence/model246/readiness.json` is self-hashed and records the exact blocker state plus the predecessor control. `tools/validate_model246_readiness.py` rejects any mutation that silently turns the blocked report into a numerical V2 claim. The dedicated workflow is LOCAL_FREE only and performs no model training while prerequisites are absent.
