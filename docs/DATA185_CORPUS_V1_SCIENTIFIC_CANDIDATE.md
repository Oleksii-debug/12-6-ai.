# DATA-185 Corpus V1 Scientific Candidate

`SWARM_WORKER_ID: DATA-185-CORPUS-V1-SCIENTIFIC-CANDIDATE`

This milestone is a convergence qualification layer over DATA-110. It does not create another preprocessing, Trainer, tokenizer, checkpoint or evaluation subsystem.

## Exact incumbents

The branch is stacked on DATA-110 exact head `fd60b362c7089e20b3c0e1fb37dc839ae5a17c5c` and re-executes its composed rights, immutable intake, quality, privacy, exact/near deduplication, D06 decontamination, cluster-safe split, deterministic sharding, document-isolated packing, Product Trainer, checkpoint/resume and held-out evaluation path through the canonical `data110_entrypoint.py`.

The previous dataset identity is DATA-25:

`422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`

The fixed comparison uses the same approximately-1M Base geometry used by DATA-110: 836,736 parameters with canonical `s0-byte-v1`, random initialization, seed 1337, FP32, AdamW, sequence length 128, batch 8, the same UA/EN/code mixture and 512 optimizer steps.

## Leakage-safe common evaluation

DATA-110 constructs its project-authored candidate material only from DATA-25 `train` rows, then performs a new cluster-safe candidate split. Therefore candidate-validation cannot be used as the authoritative common selection metric for a DATA-25 baseline, because some candidate-validation project rows originated from DATA-25 train.

DATA-185 instead uses the original DATA-25 `validation` split as the common candidate-vs-previous evaluation identity. Before the comparison is accepted, it explicitly requires zero exact content-hash overlap between the DATA-110 candidate train split and DATA-25 validation. Aggregate BPB, UA/EN/code BPB and source-family BPB are reported for both models on this common identity.

The DATA-110 candidate's own validation split is still evaluated, including every provenance-bound source family. Families without validation examples are reported with `bits_per_byte: null` and `NO_VALIDATION_EXAMPLES`; they are not silently omitted. Candidate-validation results are diagnostic and are not the candidate-vs-previous selection metric.

## Scientific qualification gates

DATA-185 records and validates:

- real external training-byte share, with a local provisional minimum of 1%;
- real source-family diversity, requiring at least two real UA families, two real EN families and one real code family;
- real code presence, or an explicit rights blocker;
- exact two-build corpus and shard determinism;
- zero train-validation cluster straddles;
- D06/reserved-eval training-eligibility evidence;
- streaming Product Trainer evidence and fresh-process resume;
- zero exact candidate-train/common-eval overlap;
- evaluation non-mutation;
- a matched previous-DATA-25 learning comparison.

The family unit is exact provenance `source_id` because DATA-110 does not bind a broader semantic family taxonomy. DATA-185 deliberately does not invent one.

## Real-code blocker

DATA-23 exact head `5f223f9ef77762a042e966372fdf9f064b3cc9fe` retained a bounded real mechanical pilot of three Python files / 4,998 bytes, but all three files were blocked by the live training-rights registry and zero real code bytes were training eligible. DATA-110 therefore contains real rights-approved UA and EN input but no rights-approved real code source.

That is an explicit blocker, not a fabricated code-data claim.

## RESEARCH-140

The fixed DATA-185 comparison is one paired trajectory. RESEARCH-140 exact head `c2fa6ba71691c3d8cc86aa0a1c3c83eb10bce98` requires at least three paired repeats before selecting a winner. Therefore the machine report records the one-pair result as `INSUFFICIENT_REPEATS`, descriptive only, with no p-value or asymptotic significance claim.

The single pair is used only as a fail-closed non-regression diagnostic on the common DATA-25 validation identity. Passing it is not a claim that the candidate is better.

## Machine status

The qualification command always writes one of exactly:

- `FREEZE_FOR_RESEARCH_V1`
- `RETEST_REQUIRED`
- `BLOCKED`

Scientific/data gaps produce `RETEST_REQUIRED` with machine reasons. Execution or evidence-integrity failures produce a minimal `BLOCKED` report and leave unsupported candidate/comparison sections absent.

The present source composition cannot satisfy the real-code gate, and the admitted external source diversity is only one rights-approved real UA source family and one rights-approved real EN source family. Therefore the expected successful exact-head qualification is:

`RETEST_REQUIRED`

The exact numerical comparison, observed real-byte share and report hash are not claimed until the exact-head workflow executes successfully.

`FREEZE_FOR_RESEARCH_V1` is reserved for a controlled comparable research identity only. It does not mean production readiness, population representativeness, intelligence, alignment or instruction following.

No foreign pretrained weights. No SFT, RLHF or DPO. No paid compute. Execution profile is `LOCAL_FREE`.
