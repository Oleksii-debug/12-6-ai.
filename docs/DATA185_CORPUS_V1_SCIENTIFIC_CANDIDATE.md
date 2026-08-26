# DATA-185 Corpus V1 Scientific Candidate

`SWARM_WORKER_ID: DATA-185-CORPUS-V1-SCIENTIFIC-CANDIDATE`

This milestone is a convergence qualification layer over DATA-110. It does not create another preprocessing, Trainer, tokenizer, checkpoint or evaluation subsystem.

## Exact incumbents

The branch is stacked on DATA-110 exact head `e4f8fe7faef93aef9a2d9a00cb8464e900b463e4` and re-executes its composed rights, immutable intake, quality, privacy, exact/near deduplication, D06 decontamination, cluster-safe split, deterministic sharding, document-isolated packing, Product Trainer, checkpoint/resume and held-out evaluation path.

The previous dataset identity is DATA-25:

`422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`

The fixed comparison uses the same approximately-1M Base geometry used by DATA-110: 836,736 parameters with canonical `s0-byte-v1`, random initialization, seed 1337, FP32, AdamW, sequence length 128, batch 8, the same UA/EN/code mixture and 512 optimizer steps.

Candidate and previous-DATA-25 arms are evaluated on one common candidate validation identity. The machine report contains aggregate BPB, UA/EN/code BPB and provenance-bound source-family BPB for both arms.

## Scientific qualification gates

DATA-185 records and validates:

- real external training-byte share, with a local provisional minimum of 1%;
- real source-family diversity, requiring at least two real UA families, two real EN families and one real code family;
- real code presence, or an explicit rights blocker;
- exact two-build corpus and shard determinism;
- zero train-validation cluster straddles;
- D06/reserved-eval training-eligibility evidence;
- streaming Product Trainer evidence and fresh-process resume;
- evaluation non-mutation;
- a matched previous-DATA-25 learning comparison.

The family unit is the exact provenance `source_id` because DATA-110 does not bind a broader semantic family taxonomy. DATA-185 deliberately does not invent one.

## Real-code blocker

DATA-23 exact head `5f223f9ef77762a042e966372fdf9f064b3cc9fe` retained a bounded real mechanical pilot of three Python files / 4,998 bytes, but all three files were blocked by the live training-rights registry and zero real code bytes were training eligible. DATA-110 therefore contains real rights-approved UA and EN input but no rights-approved real code source.

That is an explicit blocker, not a fabricated code-data claim.

## RESEARCH-140

The fixed DATA-185 comparison is one paired trajectory. RESEARCH-140 exact head `c2fa6ba71691c3d8cc86aa0a1c3c83eb10bce98` requires at least three paired repeats before selecting a winner. Therefore the machine report records the one-pair result as `INSUFFICIENT_REPEATS`, descriptive only, with no p-value or asymptotic significance claim.

The single pair is still used as a fail-closed non-regression diagnostic: the candidate must not regress by more than 0.02 BPB on the common held-out identity. Passing that check is not a claim that the candidate is better.

## Current structural decision

The present source composition cannot satisfy the real-code gate, and the admitted external source diversity is only one rights-approved real UA source family and one rights-approved real EN source family. Therefore the expected scientific qualification is:

`RETEST_REQUIRED`

The exact numerical comparison, observed real-byte share and report hash are not claimed until the exact-head workflow executes successfully.

`FREEZE_FOR_RESEARCH_V1` is reserved for a controlled comparable research identity only. It does not mean production readiness, population representativeness, intelligence, alignment or instruction following.

No foreign pretrained weights. No SFT, RLHF or DPO. No paid compute. Execution profile is `LOCAL_FREE`.
