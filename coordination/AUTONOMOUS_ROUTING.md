# 12-6 AI — current autonomous routing

EPOCH: EPOCH-0004
STATUS: ACTIVE_AUDITED_ACCELERATED
LAST_GLOBAL_AUDIT: 2026-09-07T11:50:00Z
AUDIT_BASE_MAIN: a52c0e621c8127c0c906ab4eef82dfd295b539ee
CURRENT_STAGE: PRE_LEARNED_20M / DATA_CAPACITY_EXPANSION
STRATEGY: TERMINAL_20M_PROOF_THEN_MEASURED_200M_FEASIBILITY
NEXT_AUDIT_RULE: first capable worker after >=6h since LAST_GLOBAL_AUDIT, or immediately after a major-change trigger; failover audit allowed after ~8h without a valid refresh

## Authority and startup rule

Every scheduled worker, Codex Cloud run and Work run MUST read this file first, then read `docs/TRAINING_COMPUTE_AND_SCALING_STRATEGY_2026-09-07.md`, then reconstruct live GitHub before substantive work. Exact live GitHub evidence overrides stale text here.

Do not continue an old task merely because a Scheduled Task prompt names it. Stable prompts define home lanes and recovery behavior; this routing epoch defines the current targets.

Before mutation, before PR/update/merge action and before final verdict: perform exact + semantic ownership/collision review. Reuse canonical branches/PRs. The five persistent workers do not create hourly registration issues. A blocker in one package means rotate to the next useful unowned package in the same mission, not stop.

## 2026-09-07 scaling reconciliation

The accelerated scaling strategy is now executable on `main` through the terminal-green R01 router merged on 2026-09-07. This formally reconciles older 20M→100M roadmaps.

### KEEP

These gates remain mandatory and are NOT weakened:

- canonical Base starts from random initialization; no foreign pretrained/instruct/aligned Base weights or teacher logits;
- learned ~20M remains a mandatory terminal end-to-end proof before product-scale training;
- exact data/tokenizer/corpus/split/packing/unique-loss identities;
- global dedup, evaluation decontamination and final-test firewall;
- checkpoint/save/load/fresh-process resume/recovery truth;
- reproducible training recipe, optimizer/scheduler/precision/seed/exposure identities;
- independent evaluation, loss trajectory, held-out UA/EN/code evidence, inference reload and memorization/exposure checks;
- measured compute/resource evidence rather than paper estimates alone;
- materially paid compute requires explicit owner `COMPUTE_AUTHORIZED` authority.

### CHANGE

- Learned 20M is the cheapest trustworthy full-pipeline qualification model, not a final product brain to polish indefinitely.
- Immediately after terminal learned 20M, automatically build a measured ~200M feasibility packet.
- If direct ~200M feasibility is demonstrated by real data/compute/memory/checkpoint/backend evidence, ~200M becomes the first serious product-brain campaign.
- 50M/100M remain available as bounded engineering probes only when they materially reduce risk, e.g. memory scaling, optimizer stability, throughput extrapolation or architecture/runtime transition.
- Commodity trainer mechanics should default to qualified reuse/adaptation rather than repeated custom reinvention. Project-owned scientific truth remains authoritative.

### SUPERSEDE

The old rule “terminal learned 20M must be followed by a full learned 100M campaign before a serious product-scale brain” is superseded.

Do NOT automatically run full 50M or 100M campaigns merely because an older roadmap listed them in sequence. Their scientific/data/checkpoint/evaluation/compute safety gates are preserved and apply whenever those scales are used as probes.

The new default route is:

`terminal learned 20M -> measured ~200M feasibility -> direct ~200M if feasible OR bounded 50M/100M probe only if needed -> terminal learned ~200M -> measured ~1B feasibility`

## Live audit — decisive current facts

### Learned 20M

- Random-init MODEL-341 mechanics remain established at 20,613,440 parameters.
- No terminal learned 20M checkpoint exists.
- Training-authorized post-pack optimized-target exposure remains exactly 0.
- The launch/readiness/evaluation/recovery control surfaces are substantially built; another launch-gate implementation is not a priority.

### Data is now the dominant blocker

The previous routing state saying “record-level materialization is missing” is stale.

- DATA-526 now has a real frozen 48-record materialization over 2,215,615 normalized payload bytes with exact record/payload identities and two byte-identical materializations. Its dedicated DATA-526 evidence is terminal-success; generic release CI remains separate debt.
- The incumbent decontamination line has a terminal dedicated run that reconstructed the frozen inputs twice, scanned twice, preserved the final-test firewall and produced a clean scoped result over the small candidate.
- This proves the small corpus plumbing, but the corpus is scientifically far too small for the frozen 20M source-floor plan.
- Current source-capacity vector remains approximately UA 100,856 bytes / EN 1,838,293 / code 276,466 = 2,215,615 total.
- Frozen 45/35/20 family-capped no-replay envelope is only 61,440 bytes. Ukrainian is the limiting stratum.
- Raw planning gaps to the current 20,000,000-byte source-floor composition are approximately UA 8,899,144 / EN 5,161,707 / code 3,723,534 bytes.
- Replay/padding may not manufacture capacity.
- High-yield UA acquisition incumbent Rada_Trees has terminal shared-CI metadata/object-pinning evidence, but admitted archive-content bytes remain 0 until actual archive/member provenance/quality/dedup/decontamination gates execute.

Therefore the current P0 is **rapid unique data-capacity expansion, UA first**, not repeatedly rebuilding the same 2.2MB candidate.

Every newly admitted source must re-enter:

`source authority -> global dedup -> exact record materialization -> evaluation decontamination -> post-composition quality/privacy/balance -> cluster-safe split`

Only after the target composition is genuinely feasible should D04 publish real positive post-pack exposure.

### Tokenizer / packing

- Canonical D04 packing/unique-loss lineage has terminal scoped ledger/tokenizer evidence.
- Real target-corpus-bound packing remains `NOT_MATERIALIZED`; authorized exposure = 0.
- D04 should preserve byte-baseline/token-ID safety, deterministic shard/multiworker/resume semantics and become ready to pack immediately when a terminal balanced split arrives.
- Do not churn a learned tokenizer merely because data acquisition is still incomplete.

### Trainer / checkpoint / recovery

- D02 transactional trainer restore and scaler-presence parity have been integrated into the canonical carrier.
- D05 has already demonstrated terminal generic + scoped evidence on a recent exact head; the active D05 line has since advanced again with further hardening and new exact-head CI is nonterminal at this audit cutoff.
- The former standalone checkpoint worker is folded into the Training mission from the next scheduled run onward.
- Exact uninterrupted `N` versus `K + checkpoint + fresh-process resume -> N`, including exact next-exposure identity, remains a terminal campaign proof requirement.

### Evaluation

- D06 terminal-pilot numeric evidence and evaluation-firewall logic are already integrated on the canonical carrier.
- A dedicated persistent independent Evaluation worker is now required so producer evidence is not self-certified.
- The worker will verify decontamination/split isolation, held-out UA/EN/code NLL/BPB, selection trajectory, generation, memorization/exposure diagnostics, fresh reload/inference fingerprint and compute benchmark evidence.

### Integration / Work

- Canonical integration carrier has already absorbed the new accelerated 20M→200M strategy and recent D02/D05/D06 work; its newest exact generic CI is still running at this audit cutoff, so no fresh terminal claim is made for that exact head yet.
- Active Work is already building the provider-neutral learned-20M portable run binding after the newly merged strategy router. Its first exact-head CI is red at this cutoff. This package is ACTIVE/COLLISION-RESERVED for Work; scheduled workers and Codex must not duplicate it.

## Trainer/backend reuse policy

12-6 owns:

- architecture and initialization;
- tokenizer/data/corpus authority;
- training recipe decisions;
- scientific gates and evaluation;
- checkpoint/recovery truth and lineage;
- provenance/reproducibility;
- promotion/rollback and future self-learning policy.

Commodity mechanics may be reused only after exact qualification.

Current qualification order:

1. **Plain PyTorch / current project trainer = control baseline.** It is already integrated enough to remain the reference until another backend proves parity.
2. **LitGPT candidate:** qualify as a compact from-scratch runner/reference; no adoption exists yet.
3. **Hugging Face Accelerate/Trainer candidate:** qualify only if its abstractions preserve exact project-owned identities/checkpoint semantics; no adoption exists yet.
4. **PyTorch FSDP / DeepSpeed:** qualify when ~200M/1B memory/distributed requirements justify them; do not force distributed complexity into the 20M proof without measured need.

A backend may become selectable only after exact version/license/dependency identity, real runtime, same-workload output/loss semantics, checkpoint/resume parity, reproducibility, rollback and measured resource evidence. Framework-local metadata never overrides project manifests.

## LOCAL_FREE / free-GPU compute policy

Owner laptop is a first-class LOCAL_FREE target:

- ASUS Vivobook M1505YA;
- Ryzen 5 7430U, 6 cores / 12 threads;
- 16 GB RAM;
- integrated AMD Radeon graphics.

Use it for 24/7 data processing, tokenizer/packing, evaluation, checkpoint/recovery, CPU smoke/training/benchmarks and learned-20M work if measured throughput is acceptable.

Do not claim the laptop is “too slow” without executing the same bounded benchmark workload and recording evidence. Do not assume integrated Radeon acceleration; GPU support requires actual runtime proof.

Free GPU sessions such as Kaggle/Colab or other policy-compatible temporary accelerators are first-class acceleration lanes. The scientific job must remain provider-neutral and resumable between LOCAL_FREE and free GPU sessions.

For each execution environment, use the same bounded workload and record at minimum:

- tokens/second;
- step time;
- peak RAM/VRAM;
- checkpoint size and checkpoint save/load time;
- projected wall-clock to the target exposure;
- exact runtime/backend/environment identity.

Checkpoint early enough on ephemeral services that session expiry does not erase useful progress. Paid compute remains forbidden without explicit owner authorization.

## Current critical path

`high-yield unique data acquisition (UA dominant) -> global dedup -> deterministic record materialization -> evaluation decontamination -> quality/privacy + 45/35/20 family caps -> cluster-safe split -> tokenizer decision + deterministic packing -> positive exact unique causal-loss ledger -> portable training/checkpoint/resume -> independent evaluation -> bounded LOCAL_FREE/free-GPU 20M pilot -> prove real learning/reload/resume -> terminal learned 20M -> automatic ~200M feasibility packet -> direct learned ~200M if feasible OR bounded 50M/100M risk probe only if required -> terminal learned ~200M -> ~1B feasibility`

Source bytes are not tokenizer tokens or optimized targets. Replay/padding must never manufacture unique capacity.

## Worker 1 — DATA

HOME: D03 / sources / corpus / provenance / dedup / decontamination.

CURRENT NEXT:

1. Maximize **new, independently admissible unique corpus capacity per run**, prioritizing Ukrainian until the limiting-stratum gap materially closes; then code/EN as required by balance.
2. Continue existing high-yield source owners instead of cloning them. Current primary UA lineage is Rada_Trees; move it from metadata pinning to real bounded archive/member materialization, provenance, LID/privacy/quality, dedup and decontamination.
3. Search/qualify additional rights-clear independent UA families when the incumbent path cannot supply enough capacity. Prefer large legitimate sources over dozens of micro-admissions.
4. Reuse the already-proven DATA-526/decontamination machinery; do not spend runs proving the unchanged 2.2MB candidate again.
5. Every accepted delta re-enters global dedup -> record freeze -> decontamination -> quality/privacy/balance before split.
6. Track both the learned-20M source-floor gap and future ~200M data feasibility, but never steal capacity from scientific gates or count replay as unique supply.

PRIMARY PRODUCTIVITY METRIC: newly terminal/admissible unique bytes + independent families, with exact provenance and zero evaluation leakage.

## Worker 2 — TOKENIZER / DATA PIPELINE

HOME: D04.

CURRENT NEXT:

1. Preserve the existing canonical packing/unique-loss lineage; do not restart tokenizer/ledger design.
2. While Worker 1 expands data, harden and benchmark scalable deterministic sharding/packing, multiworker ordering, boundary/truncation/padding/loss-mask semantics, resume-state serialization and portable artifact identities.
3. Keep the byte tokenizer/control usable unless measured evidence justifies BPE/Unigram; never silently change token IDs.
4. Prepare for large-corpus throughput so the same pipeline can grow from terminal 20M qualification toward ~200M without architectural rewrite.
5. Immediately after terminal balanced split identity exists: perform two independent pack builds, require byte-identical outputs, publish exact positive unique nonignored causal-loss positions and resume-safe next-exposure identity.
6. Record measured pipeline throughput/RAM on available LOCAL_FREE resources when executable; do not fabricate laptop measurements from estimates.

DO NOT: pack the known under-capacity 2.2MB graph as if it satisfied the 20M campaign; fit tokenizers to evade the source-capacity blocker; patch unrelated shared CI.

## Worker 3 — TRAINING + CHECKPOINT / RECOVERY

HOME: D02 + D05 + C01 execution seams.

CURRENT NEXT:

1. Absorb the former standalone D05 scheduled-worker mission. Continue the canonical D05 line and finish its currently active exact-head hardening/CI before taking a new overlapping checkpoint patch.
2. Terminalize fresh-process deterministic resume: uninterrupted `N` vs `K + save + fresh process + resume -> N`, binding model/optimizer/scheduler/scaler/RNG/dataloader/counters and Worker 2 exact next-exposure identity.
3. Keep plain PyTorch/current Trainer as the control. Consume independent backend qualification evidence rather than rebuilding every commodity primitive.
4. Integrate a **provider-neutral portable run packet** after active Work's current package becomes exact-green; do not duplicate Work while it is active.
5. Build/execute the same bounded compute benchmark workload on every actually available environment and record tokens/s, step time, memory, checkpoint time and projected wall-clock.
6. Use the owner laptop as a real LOCAL_FREE target whenever the execution environment actually has access to it; no unmeasured “too slow” conclusion.
7. Free GPU sessions are valid acceleration. Resume the same scientific job through exact checkpoint/run-packet identities rather than restarting per provider.
8. As soon as terminal D03/D04/D05/D06 launch inputs exist, execute the bounded learned-20M pilot; do not add another readiness layer merely to delay training.
9. Terminal learned 20M requires real optimizer updates, sane loss, exact exposure, reload/resume, independent evaluation and measured compute evidence.
10. After terminal 20M, hand measurements to Worker 5 for ~200M feasibility; do not automatically launch full 50M/100M campaigns.

DO NOT: self-authorize paid compute; adopt LitGPT/Accelerate/FSDP/DeepSpeed without qualification; claim GPU evidence from CPU.

## Worker 4 — INDEPENDENT EVALUATION / SCIENTIFIC QA

HOME: D06 + independent verification. Organizationally separate from Training producer evidence.

CURRENT NEXT:

1. Independently verify the full pretraining/evaluation isolation chain after each materially new corpus freeze: reservations -> decontamination -> split -> packing, with final-test firewall intact.
2. Reuse the already-integrated BPB and D06 pilot-evidence machinery; do not create another metric layer.
3. Define and execute terminal learned-20M acceptance: held-out UA/EN/code NLL/BPB, matched random-init comparison, selection trajectory, generation, memorization/exposure diagnostics, fresh-process inference reload/fingerprint and no final-test recipe tuning.
4. Independently verify Worker 3 checkpoint/resume and compute benchmark evidence without changing producer implementation just to make verification pass.
5. Verify backend parity/reproducibility evidence produced by Codex/Training before any backend adoption decision.
6. For the same bounded compute benchmark, check measurement comparability across laptop/free-GPU environments and reject mismatched workload identities.
7. After terminal 20M, independently audit inputs to the ~200M feasibility decision. A paper memory estimate alone is not enough.

DO NOT: run the same producer code then call it independent audit; use final-test outcomes for recipe tuning; lower gates to accelerate schedule.

## Worker 5 — INTEGRATION / COORDINATOR / COMPUTE FEASIBILITY

HOME: D10 + COORD.

CURRENT NEXT:

1. Keep a single current-main-ancestor integration carrier and obtain fresh exact-head CI after each material intake.
2. Integrate only collision-safe terminal deltas; no stale-green transfer or wholesale divergent merges.
3. Maintain exact strategy reconciliation: terminal20 mandatory; 50/100 optional probes; ~200M feasibility next; 1B only after terminal200.
4. Track active Work/Codex packages before taking cross-lane work. Active Work currently owns portable run binding and must not be duplicated.
5. Repair shared CI/fanout/bootstrap debt centrally when it actually blocks the critical path.
6. Own the cross-environment compute benchmark matrix and, after terminal20, the automatic ~200M feasibility decision using measured data capacity, memory, optimizer state, activations, gradient accumulation/checkpointing, throughput, checkpoint transport and available LOCAL_FREE/free-GPU compute.
7. A ~200M GO decision must name the qualified backend and measured evidence. If direct ~200M is not yet defensible, choose the smallest bounded 50M/100M probe that resolves the specific uncertainty; do not default to a full intermediate campaign.
8. Perform global routing audit only when >=6h old or a major-change trigger fires. After audit, return to productive integration/feasibility work.

## Work — active and next

ACTIVE / RESERVED:

`PORTABLE-LEARNED20-RUN-BINDING` — current Work/R01 follow-up binds terminal learned-20M authorities into a provider-neutral LOCAL_FREE/free-GPU run packet with fresh/resume semantics. Its first exact-head CI is red at this audit cutoff. Work owns repair/terminalization of that same package. Do not duplicate it.

WORK NEXT AFTER CURRENT PACKAGE:

`COMPUTE-BENCHMARK-AND-200M-FEASIBILITY-INTEGRATION` — only after checking current routing/ownership. Connect the portable packet to a single bounded benchmark protocol, measured laptop/free-GPU evidence and fail-closed ~200M feasibility inputs. If another worker/Codex already owns an equivalent package, take the largest disjoint cross-lane blocker instead.

A major Work result triggers immediate routing refresh; do not wait six hours.

## Codex Cloud — next large disjoint package

NEXT RECOMMENDED PACKAGE:

`TRAINING-BACKEND-QUALIFICATION-MATRIX-V1`

Goal: independently qualify commodity trainer reuse without changing the canonical trainer or active Work portable-run binder.

Required arms:

- project current plain PyTorch Trainer = control;
- LitGPT candidate;
- Hugging Face Accelerate/Trainer candidate;
- record FSDP/DeepSpeed as future ~200M/1B candidates unless the bounded environment makes a meaningful qualification possible now.

For each executable candidate: pin exact version/source/license/dependency identity; run the exact same project-owned bounded workload; compare loss/update semantics, state serialization, fresh-process resume, deterministic/reproducible outputs, checkpoint portability, tokens/s, step time, peak RAM/VRAM and checkpoint time; verify rollback to the project control. If runtime/dependency is unavailable, record `RETEST_RUNTIME_REQUIRED`, not PASS. No canonical adoption, model training campaign, final-test access or paid compute authority is granted by the qualification package.

Codex must NOT take Worker 1 source acquisition, Worker 2 packing, Worker 3 canonical checkpoint/pilot, Worker 4 independent stage verdict, Worker 5 carrier/feasibility orchestration, or active Work portable-run binding.

## Integration queue

1. Keep the current carrier reconciled with live main and exact-head CI.
2. Let active Work repair/terminalize portable run binding; consume it only after exact evidence.
3. Grow real corpus capacity aggressively, especially UA, and rerun the existing dedup/decontamination/balance chain on each material source expansion.
4. Once terminal balanced corpus/split exists: immediate deterministic pack -> positive unique-loss ledger.
5. Complete fresh-process checkpoint/resume and portable provider-neutral run authority.
6. Independent evaluation verifies isolation and pilot acceptance.
7. Execute bounded LOCAL_FREE/free-GPU learned-20M pilot as soon as gates permit.
8. Finish terminal learned-20M end-to-end proof; do not over-polish it as the final product model.
9. Automatically build the measured ~200M feasibility packet.
10. Direct ~200M if feasible; otherwise only the smallest useful 50M/100M probe that resolves a named risk.
11. Terminal learned ~200M -> measured ~1B feasibility.

## Scale state

- 20M mechanics: READY / random-init mechanics only.
- 20M corpus target: BLOCKED primarily on unique data capacity, UA dominant.
- 20M post-pack exposure: 0 / NOT MATERIALIZED for the target campaign.
- 20M bounded pilot: NOT READY until terminal corpus/split/packing/recovery/evaluation composition.
- 20M learned terminal proof: NOT COMPLETE.
- ~200M: FEASIBILITY PREPARATION ONLY before terminal20; no learned campaign authority yet.
- 50M/100M: OPTIONAL BOUNDED ENGINEERING PROBES ONLY after terminal20 when a specific risk justifies them; no automatic full campaigns.
- ~1B: CLOSED until terminal independently verified learned~200M + measured data/compute/checkpoint/evaluation feasibility.

## Owner compute / authorization state

LOCAL_FREE engineering is authorized by project policy. Owner laptop and free GPU services may be used only when actually accessible to the executing environment and measurements are recorded truthfully.

Materially paid cloud/GPU compute remains NOT AUTHORIZED. No owner action is currently required because paid compute is not the sole blocker.

## Acceleration objective

The owner wants a major speedup in project progress. Do not promise a synthetic multiplier. Achieve acceleration by removing structural waste:

- pipeline-ordered worker starts: Data -> Pipeline -> Training -> Independent Evaluation -> Integration;
- stable prompts that always read this live plan;
- no repeated terminal proofs;
- no duplicate branches/PRs;
- large high-yield data sources instead of micro-task churn;
- qualified commodity backend reuse;
- portable LOCAL_FREE/free-GPU checkpoint/resume;
- same-workload compute benchmarking before provider decisions;
- skip unnecessary full 50M/100M campaigns;
- immediate 200M feasibility after terminal20.

## Short owner-readable state

20M remains mandatory and is still blocked, now mainly by insufficient unique corpus capacity rather than missing plumbing. The small record/decontamination path works; Ukrainian data supply is the largest gap. The project has formally changed the post-20M route: measured ~200M feasibility is next, while 50M/100M are optional probes. Work is actively building portable LOCAL_FREE/free-GPU run binding. No third-party high-level trainer is adopted yet; plain PyTorch/current Trainer remains the control while LitGPT/Accelerate are qualified independently. No paid compute or owner action is required now.