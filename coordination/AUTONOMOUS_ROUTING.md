# 12-6 AI — current autonomous routing

EPOCH: EPOCH-0005
STATUS: ACTIVE_AUDITED_ACCELERATED
LAST_GLOBAL_AUDIT: 2026-09-07T22:08:00Z
AUDIT_BASE_MAIN: e7379f31c4d5909b997cd4dc8d99155bc9a9764e
CURRENT_STAGE: PRE_LEARNED_20M / RADA_ADMISSIBILITY_AND_CURRENT_GRAPH_REBUILD
STRATEGY: TERMINAL_20M_PROOF_THEN_MEASURED_200M_FEASIBILITY
PRODUCT_SCALE_DEFAULT: DIRECT_APPROX_200M_IF_MEASURED_FEASIBILITY_PASSES
100M_POLICY: OPTIONAL_BOUNDED_ENGINEERING_PROBE_ONLY_WHEN_IT_REDUCES_RISK
NEXT_AUDIT_RULE: first capable worker after >=6h since LAST_GLOBAL_AUDIT, or immediately after a major-change trigger; failover audit allowed after ~8h without a valid refresh

## Authority and startup rule

Every scheduled worker, Codex Cloud run and Work run MUST, before substantive work:

1. recover exact repository `Oleksii-debug/12-6-ai.` (the trailing period is part of the repository name);
2. read this file;
3. read `docs/TRAINING_COMPUTE_AND_SCALING_STRATEGY_2026-09-07.md`;
4. recover current `main`, #548, its home lane, relevant open/recent PRs, exact-head CI and active Work/Codex claims;
5. perform exact + semantic collision/ownership review before mutation, before PR/merge action and before final verdict.

Live exact GitHub evidence overrides stale text in this routing file. If a named PR/head below moved, was merged, closed or superseded, follow the current canonical successor instead of reviving stale lineage.

A blocker is never permission to idle or disable an automation. Rotate to the next useful unowned package in the same mission. The five persistent scheduled workers must remain enabled unless Oleksii explicitly orders otherwise. Do not create hourly registration issues or duplicate active producer branches.

## Scaling reconciliation — permanent policy

### KEEP

The following gates remain mandatory:

- canonical Base starts from random initialization; no foreign pretrained/instruct/aligned Base weights or teacher logits;
- learned ~20M is the mandatory terminal end-to-end proof before product-scale training;
- exact data/tokenizer/corpus/split/packing/unique-loss/exposure identities;
- global dedup, evaluation decontamination and final-test firewall;
- checkpoint save/load/fresh-process resume/recovery truth;
- exact training recipe, optimizer/scheduler/precision/seed/accumulation identities;
- independent held-out evaluation, loss trajectory, UA/EN/code evidence, generation, memorization/exposure checks and inference reload;
- measured compute/resource evidence rather than paper estimates alone;
- materially paid compute requires explicit Oleksii `COMPUTE_AUTHORIZED` authority.

### CHANGE

Learned ~20M is the cheapest trustworthy full-pipeline qualification model, not the final product brain. Immediately after terminal learned-20M, automatically build a measured ~200M feasibility packet covering data capacity, RAM/VRAM, model/optimizer state, activations, gradient accumulation/checkpointing, throughput, checkpoint transport, backend qualification and available compute.

If direct ~200M feasibility passes, ~200M becomes the first serious product-brain campaign. A ~100M model is still available and useful, but only as a bounded engineering/scaling probe when measurements show it materially reduces risk before ~200M.

### SUPERSEDE

The historical rule “terminal learned-20M must be followed by a full learned-100M campaign” is superseded. Do not launch full 50M/100M campaigns merely because an old roadmap listed them.

Default route:

`terminal learned-20M -> measured ~200M feasibility -> direct ~200M if feasible OR bounded 50M/100M probe if specifically needed -> terminal learned ~200M -> measured ~1B feasibility`

All scientific/data/checkpoint/evaluation/compute gates still apply at any scale.

## Live audit — decisive state

### Main / integration

At audit cutoff current `main` is `e7379f31c4d5909b997cd4dc8d99155bc9a9764e`, merging #857 DATA-232 matcher mechanics. Live main also already contains the current global-dedup/unique-loss/CulturaX convergence surfaces from #824/#835/#859. This is mechanics/integration progress, not learned-training authority.

Canonical D10 current-main/carrier reconciliation remains an active moving lineage around #831/#802. Never transfer green across moved heads. D10 must use exact current-main ancestry, expected-head protection and fresh exact-head CI after every material intake.

### Learned ~20M

- MODEL-341 random-init mechanics remain approximately 20.6M parameters (`20,613,440`).
- No terminal learned-20M checkpoint exists.
- Training-authorized post-pack optimized-target exposure remains exactly `0`.
- No optimizer update, tokenizer fit, final-test outcome access or paid training is authorized by this audit.
- Another launch-gate abstraction is not the priority; the remaining job is to make the real data/exposure/recovery/evaluation chain terminal and then actually run the bounded pilot.

### Major new DATA fact — Rada_Trees capacity breakthrough

The old EPOCH-0004 statement that Rada_Trees had only metadata/sample evidence is stale.

Canonical PR #820 executed the full selective plaintext scan of exact immutable `uacorpus/Rada_Trees@1b994a5804dcda122721e8d33a03fd172cf8d867/rada_xtag_texts.7z`.

Terminal scientific execution on source head `2773e24f943c52caa75b15b6ab162945411a0697`:

- workflow `34162348408` = SUCCESS;
- two independent processing passes were byte-identical;
- archive SHA-256 `737fa5df061c55555a8c761023559a206cbcabf0299ae52d1ccae387f685803e`;
- archive inventory: 8,782 members / 19,711,802,635 unpacked bytes;
- selected `.txt`: 4,391 / 4,391 plaintext candidates;
- selected listed bytes: 879,031,855;
- exact unique payloads after exact-duplicate collapse: 4,385;
- source-candidate bytes after exact-duplicate collapse: **877,983,909**;
- six exact duplicate groups; 1,047,946-byte exact-duplicate discount;
- terminal evidence is text-free.

Duplicate successor #860 was correctly closed because #820 became the single incumbent authority. DO NOT run another equivalent full Rada_Trees plaintext scanner.

This is a very large acceleration opportunity, but these are SOURCE-CANDIDATE BYTES, not training-authorized corpus bytes. They grant exactly zero optimized targets until downstream admission gates pass.

### Data bottleneck has changed

The previous 6.095MB global-dedup V8 graph is now a predecessor graph, not the target campaign graph. Its terminal evidence remains useful as machinery/earlier authority but must be recomputed after admissible Rada_Trees intake.

The P0 is no longer “find enough potential Ukrainian bytes.” The P0 is now **convert the already measured ~878MB Rada_Trees candidate into scientifically admissible current-corpus capacity as quickly as possible**.

Required sequence for the 4,385 exact-unique Rada payloads:

`period/session provenance -> rights-scope revalidation -> Ukrainian/language evidence -> quality/privacy -> global exact+near+lineage dedup with current graph -> deterministic record materialization -> reserved-evaluation decontamination -> post-composition balance/family caps -> cluster-safe split`

Only survivors of that chain may enter tokenizer/packing/exposure accounting.

Search for additional sources only if the admissibility result proves Rada_Trees cannot satisfy the needed independent-family/mix/capacity envelope or if another large independent source closes a demonstrable remaining stratum gap faster than processing the incumbent.

### Tokenizer / packing / exposure

Old PR #498 is CLOSED and must not be resurrected as the live owner. Its useful mechanics have successors/current-main convergence.

At audit cutoff:

- #835 unique-loss/exposure runtime is already integrated on main;
- #843 is the active current-main byte-tokenizer/deterministic-packing convergence owner;
- Work created a non-duplicative split->packing materialization bridge around #856; always re-check its current live state before touching that seam;
- authorized positive real-corpus post-pack exposure remains `0` until a terminal decontaminated/balanced/split corpus is packed.

Do not create a second tokenizer architecture. Keep the byte-tokenizer control stable unless measured evidence justifies BPE/Unigram. Prioritize scalable deterministic sharding/packing, double-build reproducibility and exact positive unique-loss accounting.

### Training / backend qualification

Current project-native PyTorch/Trainer remains the semantic control baseline. PR #850 owns a fail-closed backend qualification contract and bounded tiny-CPU reference benchmark; it is not permission to call MODEL-341 throughput measured.

No LitGPT, Hugging Face Accelerate/Trainer, FSDP or DeepSpeed backend is adopted by this audit. A candidate must prove exact version/license/dependency identity, real runtime, same-workload semantics, checkpoint/resume parity, reproducibility, rollback and resource measurement before becoming selectable.

Prefer REUSE -> ADAPT -> CUSTOM(thin). Do not spend recurring cycles reimplementing commodity trainer mechanics that a qualified backend can safely provide.

### Checkpoint / recovery

PR #590 remains the current D05 checkpoint/recovery convergence line on the MODEL-341 carrier and has materially advanced. It owns transactional save/load, namespace/path race hardening, strict manifests, D04 ordered-exposure binding and fresh recovery semantics.

The terminal campaign still requires uninterrupted `N` steps versus `K + save + fresh process + resume -> N`, including exact next-exposure identity, no skipped/replayed batch and chronological/final/best checkpoint truth.

D05 is now part of the Training worker mission, not a separate scheduled-development lane.

### Independent evaluation

D06 mechanics exist, but the scheduled worker topology previously lacked a truly dedicated persistent independent Evaluation lane. EPOCH-0005 assigns one scheduled worker exclusively to independent scientific QA so Training cannot self-certify its own producer evidence.

### Compute

Owner laptop remains a first-class LOCAL_FREE target:

- ASUS Vivobook M1505YA;
- Ryzen 5 7430U, 6C/12T;
- 16 GB RAM.

Do not call it too slow without a bounded same-workload benchmark. Use it for data processing, tokenizer/packing, evaluation, checkpoint/recovery, CPU training/smokes and learned-20M work if measured throughput is acceptable.

Free GPU sessions (Kaggle/Colab/other policy-compatible temporary accelerators) are first-class acceleration lanes. Scientific runs must be portable/resumable between environments through exact run-packet + checkpoint identities.

For every actual environment measure the same bounded workload:

- optimized targets/tokens per second;
- step time;
- peak RAM/VRAM;
- checkpoint size + save/load time;
- projected target wall-clock;
- exact runtime/backend/environment identity.

Paid compute remains forbidden without explicit owner authorization.

## Current critical path

`Rada_Trees #820 admissibility -> current-graph global dedup -> exact record materialization -> #857-compatible evaluation decontamination -> quality/privacy + 45/35/20 family caps -> cluster-safe split -> #843/current packing -> two clean pack builds -> #835 positive unique-loss/exposure authority -> #590 checkpoint/fresh-resume + #850 training/backend authority -> independent D06 acceptance -> bounded LOCAL_FREE/free-GPU learned-20M pilot -> real learning/reload/resume -> terminal learned-20M -> automatic ~200M feasibility -> direct ~200M if feasible OR bounded ~100M probe if needed`

Source bytes are never tokenizer tokens or optimized targets. Replay/padding may never manufacture unique capacity.

## Worker 1 — DATA

HOME: D03 / #4 / data acquisition, provenance, rights, language, quality/privacy, global dedup, record materialization and decontamination handoff.

CURRENT NEXT, in priority order:

1. Reconstruct live #820 first. Do not repeat the already-successful full plaintext scan and do not revive #860.
2. Terminalize the current #820 branch/release evidence if its latest exact-head CI is nonterminal/red; repair only concrete D03-owned failures.
3. Move the 4,385 exact-unique Rada payloads through period/session provenance and exact source/member lineage.
4. Revalidate rights scope for the exact plaintext layer, then execute Ukrainian/language, quality and privacy screening with text-safe durable evidence.
5. Feed only admissible survivors into current global exact/near/lineage dedup; produce exact survivor identity and deterministic payload record inventory.
6. Run reserved-evaluation decontamination using the current integrated DATA-232 matcher; final-test payload/outcome access stays forbidden.
7. Recompute post-composition quality/privacy, 45% UA / 35% EN / 20% code, family caps and independent-family requirements; then publish cluster-safe split input.
8. If Rada survivors still leave a real capacity/family deficit, qualify the largest rights-clear independent source that closes that exact gap. Do not return to dozens of micro-source churn by default.
9. Track future ~200M data feasibility in parallel, but never count candidate/replay bytes as training capacity.

PRIMARY PRODUCTIVITY METRIC: newly terminal **admissible post-gate unique bytes and independent families**, not raw/candidate bytes or number of PRs.

## Worker 2 — TOKENIZER / DATA PIPELINE

HOME: D04 / #5.

CURRENT NEXT:

1. Recover live D04 ownership every run. #498 is historical/closed; do not resurrect it. At this audit #835 is merged runtime and #843 is the active current-main tokenizer/packer owner; inspect #856/current successors before mutation.
2. Terminalize current-main packing mechanics and current-head CI without creating a second tokenizer/ledger architecture.
3. Harden scalable deterministic sharding, multiworker ordering, document/cluster boundaries, truncation/padding/loss-mask semantics, stream resume state and portable artifact identities.
4. Benchmark pipeline throughput/RAM on actual available LOCAL_FREE environments when executable.
5. Keep the frozen byte-tokenizer control unless measured fertility/throughput/quality evidence justifies a migration; never silently change IDs.
6. The instant Worker 1 publishes terminal decontaminated/balanced/split identities, run two independent deterministic pack builds, require byte-identical outputs and publish exact positive unique nonignored causal-loss positions plus resume-safe next-exposure identity.
7. Make the pipeline usable without rewrite for the later ~200M campaign.

DO NOT: fabricate positive capacity from source bytes; fit a tokenizer to evade data gates; patch unrelated D03/D10 debt.

## Worker 3 — TRAINING + CHECKPOINT / RECOVERY

HOME: D02 + D05 + C01 / #3 + #6.

CURRENT NEXT:

1. Absorb all former standalone Checkpoint-worker duties. Recover and continue canonical #590 rather than creating a parallel D05 branch.
2. Terminalize #590 current-head CI/convergence against the current D04 exposure runtime, repairing only D02/D05-owned defects.
3. Recover and continue #850 backend-qualification work. Project-native PyTorch/Trainer stays the control until a candidate proves parity.
4. Finish true fresh-process `N` vs `K + checkpoint + resume -> N` equivalence, binding model/optimizer/scheduler/scaler/RNG/dataloader/counters and Worker 2 exact next-exposure identity.
5. Maintain provider-neutral portable run/checkpoint identities; if Work already owns a live portable-run package, consume it after terminal evidence rather than duplicate it.
6. Execute the same bounded benchmark on every genuinely available environment and record tokens/s, step time, RAM/VRAM, checkpoint time and projected wall-clock.
7. Qualify LitGPT and/or HF Accelerate/Trainer only when that scope is not already owned and only through real runtime/parity evidence. FSDP/DeepSpeed wait for measured ~200M/1B memory/distributed need.
8. As soon as D03/D04/D05/D06 scientific inputs are terminal, execute the bounded learned-20M pilot. Do not add another readiness abstraction merely to postpone training.
9. Terminal learned-20M requires real optimizer updates, sane loss trajectory, exact exposure, checkpoint/reload/resume and independent Evaluation acceptance.
10. After terminal 20M, provide measurements to Worker 5; do not auto-launch full 100M.

DO NOT: self-authorize paid compute; call tiny-fixture throughput MODEL-341 throughput; claim GPU evidence from CPU.

## Worker 4 — INDEPENDENT EVALUATION / SCIENTIFIC QA

HOME: D06 / #7. This worker must remain organizationally separate from Training producer evidence.

CURRENT NEXT:

1. Independently verify every materially new corpus authority: source/member identity -> global dedup -> record materialization -> reserved-evaluation decontamination -> balance/family caps -> split -> packing, with final-test firewall intact.
2. Independently audit the Rada_Trees admission evidence; candidate bytes are not corpus bytes until exact gates pass.
3. Reuse existing BPB/evaluation primitives instead of creating another metrics stack.
4. Define and execute terminal learned-20M acceptance: held-out UA/EN/code NLL/BPB, matched random-init control, selection trajectory, generation probes, memorization/exposure diagnostics and fresh-process inference reload/fingerprint.
5. Independently verify Worker 3 fresh-resume/checkpoint evidence and reject producer self-certification.
6. Independently verify same-workload compute benchmark comparability across laptop/free-GPU environments.
7. Review #850/backend candidate parity before any backend adoption.
8. After terminal learned-20M, independently audit the ~200M feasibility inputs and any proposed 100M probe rationale.

DO NOT: change producer code merely to make evaluation pass; use final-test outcomes for recipe tuning; weaken a scientific gate for schedule speed.

## Worker 5 — INTEGRATION / COORDINATOR / COMPUTE FEASIBILITY

HOME: D10 + COORD / #11 + #12.

CURRENT NEXT:

1. Recover live main and current canonical carrier/reconciliation line each run. Around this audit #831/#802 are the D10 ancestry surfaces; follow the live successor if they move.
2. Keep one current-main-ancestor carrier, integrate only terminal collision-safe deltas and require fresh exact-head CI after every material intake.
3. Do not transfer stale green across main/head movement. Repair shared Ruff/bootstrap/workflow/API drift centrally rather than making every domain branch patch it.
4. Track active Work/Codex claims before taking cross-lane code. Assign them the largest unowned packages, not scheduled-worker duplicates.
5. Maintain the exact strategy: terminal20 mandatory; ~100M optional risk probe; measured ~200M default next product-scale feasibility; 1B only after terminal~200M.
6. Own the cross-environment compute matrix and automatic post-terminal20 ~200M feasibility packet.
7. A ~200M GO must include measured data capacity, memory, optimizer state, activation/accumulation/checkpoint plan, throughput, checkpoint transport, exact backend/runtime and available compute.
8. If direct ~200M is uncertain, define the smallest bounded 50M/100M probe that resolves a named uncertainty. Never run a full campaign only to satisfy old sequencing.
9. Perform global audit when >=6h old OR immediately after a major change such as terminal Rada admission, positive pack ledger, first optimizer update, terminal20, major Work/Codex result or active-owner collision. Update this routing file, then immediately return to productive integration work.

## Work / Codex routing

Before taking work, both must read live routing and all active claims. A long Work/Codex run should take the largest **unowned** cross-lane package and leave durable GitHub state; it must not duplicate one of the five scheduled producers.

At EPOCH-0005 the high-value package families are:

- current-corpus cross-lane composition after Rada admissibility;
- shared carrier/CI convergence that blocks multiple terminal lane heads;
- external training-backend real-runtime qualification (LitGPT / HF Accelerate) when #850 establishes the reference and no other owner is active;
- portable compute/run integration if no active Work package already owns it;
- post-terminal20 measured ~200M feasibility.

Never assume a Work/Codex package named in this file is still active hours later. Reconstruct live ownership first.

## Worker scheduling topology

For minimum dependency latency, scheduled worker order is:

`DATA -> TOKENIZER/DATA PIPELINE -> TRAINING+CHECKPOINT -> INDEPENDENT EVALUATION -> INTEGRATION/COORDINATOR`

Use one run of each per hour, staggered by 12 minutes. Each downstream worker consumes any terminal upstream change from the preceding slot; if upstream has not produced a usable handoff, it executes its own highest-value fallback instead of idling.

## Current stage verdict

- learned-20M mechanics: READY ENOUGH TO CONVERGE, not terminal learned evidence;
- learned-20M data/exposure: BLOCKED, authorized optimized-target exposure = 0;
- Rada_Trees source-candidate supply: LARGE POSITIVE MEASURED (~877.98MB after exact duplicate collapse), NOT YET TRAINING-ADMISSIBLE;
- bounded learned-20M pilot: NOT READY until real current corpus -> split -> pack -> positive exposure + recovery/eval composition becomes terminal;
- full learned-20M: NOT AUTHORIZED yet;
- learned-100M: OPTIONAL PROBE ONLY after terminal20 when a named risk requires it;
- learned-~200M: DEFAULT SERIOUS PRODUCT-BRAIN TARGET after terminal20 if measured feasibility says GO;
- 1B: future systems gate after terminal~200M measurements.

## Owner authorization

No Oleksii action and no paid-compute authorization is required at this audit state. Continue LOCAL_FREE engineering, CI, data admission, reproducibility, backend qualification and bounded benchmarks autonomously.

Ask Oleksii only for a genuinely material decision that cannot be resolved from project authority, especially a concrete paid-compute request with measured reason/budget or an irreversible rights/product policy decision.
