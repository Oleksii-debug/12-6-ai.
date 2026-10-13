# Owner addendum 2026-10-13 — agent-first autonomous research product and learning priorities

Status: **OWNER REQUIREMENTS RECORDED / IMPLEMENTATION NOT CLAIMED**. This is a new, versioned product requirement overlay, **not** a retroactive DONE claim, a change to already frozen section acceptance, permission to spend, or proof that any external service or model is connected. Existing Plan 1–10 authorities and Google Drive source plans remain unchanged until a planned owner-approved scope allocation. No current worker is to overwrite active section implementations.

## Owner's intended end product

A **single accessible, persistent, autonomous AI research organism** with a replaceable 12-6 brain, memory, tools, multiple independently verified reasoning roles/models, and a research loop that can work for hours/days under resource limits. The goal is **not** an encyclopedia, next-token pretraining alone, a fixed macro, or an endless internal dialogue. The owner may assign long and demanding technical problems without selecting one fixed scientific discipline.

**Priorities** (owner clarification): (1) long-running research and accomplishing tasks through the Internet, (2) rigorous transferable reasoning, logic, causal analysis, uncertainty and error correction, (3) programming/software engineering, defensive cybersecurity, computing/IT and engineering, (4) agent + multi-model orchestration, (5) cumulative memory and verified improvement, (6) optional multimedia/device interaction. Other domains remain accessible through retrieval/tools/teachers, not mandatory memorization. Priorities are curriculum/evaluation and product policy, not a claim of achieved model capability.

## Three boundaries — do not collapse them

1. **Canonical Base weights:** train from random initialization and authorized lawful corpora; current decoder-only baseline stays versioned. No foreign pretrained weights as Base ancestors. Do not covertly merge outside checkpoints or change the tokenizer. Training completion requires actual optimizer updates, checkpoints, and independent held-out evaluation.
2. **Post-Base learned descendants:** instruction/reasoning/agentic/skill-targeted curricula, adapter/specialized descendants and possible bounded weight updates; original Base checkpoint immutable. Specialized learning must measure retention and negative transfer on old skills. Candidate weights require provenance, authorization, tests, evaluation, rollback and deliberate promotion.
3. **Agent/runtime organism:** tools, browsers, external local/API AI models, memory, scheduling, planning, dialogue, sensors, resource management and autonomous loops can work with the currently available model(s) **without** changing Base weights. Never confuse new memory with new trained weights. Existing Nika Core/Autopilot assets are integration candidates only, not evidence of shipped features.

## Required capabilities and acceptance contracts

### A. Goal-driven long-running research — priority P0
- The owner sets a goal, constraints, available models, authorization envelope, budget, requested output and optionally deadline, with no need to choose a domain.
- Durable task record: goal, plan, subtasks, sources/URLs, exact retrieval timestamps, claims and evidence, hypotheses and counterexamples, tests, tool/effect receipts, intermediate artifacts, spent resources, current state, next action, and restart cursor.
- Loop: clarify goal when essential -> plan -> search primary and secondary sources -> extract/compare -> design or implement -> execute permitted tests -> independent verification/critique -> revise hypotheses -> save checkpoint -> continue/report. Research can last 2–5 hours, days or weeks *only while authorized compute/runtime is active*; no fictional background progress.
- Progress and stopping: bounded steps, time/resource limits, actual information gain, duplicate detection, no-progress recovery/replan, user stop/pause, crash recovery, and no blind repeat of unknown external side effects.
- Success evidence: a restart-interrupted multi-hour scenario resumes without losing sources/evidence or redoing irreversible effects; objectively completes a complex IT/engineering deliverable with provenance and tests.

### B. Strong, verified, transferable reasoning — priority P0
- Independent planner, researcher, hypothesis generator, logical/deductive checker, causal critic, contradiction finder, code/test executor and final acceptance auditor. Roles can be contexts over one model; **do not require one neural checkpoint per role**.
- Formal solvers, compilers, tests, calculators and ground-truth measurements outrank model self-confidence. A second model's agreement is **not proof**. Claims marked verified only when supported by applicable evidence.
- Distinguish induction/deduction/causation/correlation; require counterfactual/intervention support when claiming causality. Store uncertainty and alternative hypotheses; explain unresolved gaps, not fabricate certainty.
- Success evidence: held-out suites for logic, causal transfer, coding, cyber-defense, tool use, error detection, proof/checking, and out-of-domain generalization, with cost, latency, ablations vs a single-model baseline and regression tests. More time/teachers must be measured, not asserted equivalent to 20B/50B/100B weights.

### C. Multi-model and modular cognition — priority P0
- One user-facing organism dynamically invokes a subset of specialized modules/models; routes need **not** be round-robin or linear. Allowed next recipient, order, retries, re-entrancy, critique loops and termination depend on task state, intermediate evidence, capability and cost.
- Include both (i) possible **within-one-model** learned specialist blocks/router (research architecture), and (ii) **separate** local/API/browser-accessed models coordinated by a dispatcher (product runtime). Do not confuse sparse active parameter count with total disk-installed parameters or with quality of one giant dense model.
- Providers: canonical 12-6 model, separately authorized local open-weight teachers/critics, external APIs and permitted browser-mediated services. Every model call records exact identity/version, task-specific context, scope, confidence, cost, errors and source lineage. Third-party responses are candidate evidence, not facts.
- Resource governor: explicit live RAM/CPU/GPU/storage/network constraints; limited simultaneous active models; measured load/unload/quantization overhead, KV-cache and swap use; flexible sequencing on owner laptop and server. No guarantee an arbitrary 15B/20B model will run well on 16-GB RAM.
- Dispatcher can learn useful routing policies *separately* from models: optimize correctness and total resource budget, evaluate different paths, avoid loops and detect stagnation; model-to-model handoff preserves structured claims, artifacts, unknowns and verification receipts rather than relying only on lossy summaries.
- Success evidence: same benchmark in single-model, fixed-route and adaptive-route modes; reproducible gains after accounting for RAM, cold model loads, elapsed wall time, calls and cost.

### D. Tool-mediated real-world action — priority P0/P1
- Versioned, typed, least-privilege, permissioned tools for browser search/navigation, web reading, lawful downloading, document/PDF/spreadsheet parsing, filesystem, GitHub/code changes, Python/calculators, tests, datasets/DB/RAG, email/communication connectors and permitted external websites.
- Use existing source-audited/reviewed components where possible. Nika Core, Autopilot and other owner repositories are **candidate integrations** after exact API/compatibility, license/security and tests; no blind copying into canonical Base or duplicate execution authority.
- Remote registrations, purchases, messages, data deletion, privileged commands and physical actuation require appropriate user authorization; protect credentials from model/context injection, use explicit effects ledger; never allow untrusted page content to grant itself permissions. Respect third-party terms and access controls.
- Success evidence: full trace of authorized multi-step research -> retrieval -> code/test/document production, exact effect receipts, failure/restart and malicious-web-content tests, and NVDA-readable progress and controls.

### E. Memory, preferences, knowledge and continual learning — priority P0
- User-controlled separate memory domains: personal facts/preferences/instructions; project/task memory; episodic research log; verified semantic/procedural knowledge; hypotheses and rejected claims; tool/source provenance. Inspect, correct and **delete** selected user memory, with policy-defined downstream/index/cache deletion and clear limits for immutable audits/backups.
- Working context vs durable facts vs trained weights have separate lifecycle/permissions; preserve attribution, expiry/staleness, contradiction, confidence and privacy classification. No automatic promotion of web/teacher text into trusted memory or weights.
- Offline/controlled true weight-learning loop: collect verified successes -> quality/license/privacy review -> curate domain-balanced candidate training set -> test unchanged Base/parent -> train a new descendant or adapter -> independent held-out/anti-forgetting evaluation -> promote/rollback. No self-granted permission to change deployed weights.
- Success evidence: multi-session memory read/correction/deletion and retrieval, stale-evidence rejection, checkpoint-resume; full candidate-weight update and rollback receipts where separately authorized.

### F. Technical curriculum and skill acquisition — priority P0 for learning design
- Current Base trainer supplies next-token/casual-LM objective and optimizer mechanics, **not** a complete owner-approved syllabus nor a proven technical-reasoning brain.
- Candidate post-Base syllabus: logic and formal verification; causal reasoning and hypothesis testing; mathematics/algorithms; Python and software engineering; tests/debugging/reproducibility; operating systems/networking and authorized defensive cybersecurity; API/browser/tool orchestration; research/source criticism; long-horizon planning and recovery; uncertainty and self-correction; transfer to unseen technical tasks.
- For each domain: data provenance and licenses, training objective and allowed split, negative examples, baseline, metrics, minimum generalization performance, replay/retention checks and checkpoint identity. Modular/curriculum/teacher-assisted approaches are hypotheses to A/B test, not automatic production mandates.
- Success evidence: measured increase over comparable Base and non-modular descendants on isolated test sets, not only lower training loss.

### G. Perception, media and device interfaces — priority P1
- Support images/screenshots, audio/STT/TTS, video via approved external or local adapters, transcription, accessible documents, and optional camera/microphone/sensor streams. The compact Base does not have to internalize all modalities.
- Device control uses a typed service boundary: model decision -> authorized controller/program -> hardware interface/microcontroller/driver -> actuator, sensor return -> verified observation. No direct arbitrary electrical commands generated by model; simulator first, physical interlocks and user consent.
- Success evidence: independently tested accessible media pipelines with provenance, owner permissions, and negative tests for unsolicited camera/microphone/hardware access.

### H. Portable deployment, autonomy and accessibility — priority P0/P1
- Same model/agent contracts operate on laptop, private server and supported cloud subject to resources; deployment to an ordinary web-hosting account is **not** automatically possible without a continuously available compute process and adequate RAM.
- Owner may supervise via NVDA keyboard, text-based CLI and accessible web interface, inspect what the agent did for hours, pause/resume or revoke grants and memory. User-facing logs distinguish installed, configured, implemented, tested, and independently accepted capabilities.
- No automatically incurred paid compute, external-model charges, unauthorized registration, irreversible remote action, or hidden always-on agents.

## Current repository mapping (2026-10-13 audit; do not conflate scaffolding with product)
- Base architecture and objective: `src/twelve_six/model.py`, `src/twelve_six/training/trainer.py`, `src/twelve_six/training/loss.py`. **Reference implementation exists**, learned large-scale product not established.
- Model serving/orchestration: `docs/PLAN4_SECTION8_MULTIMODEL_SERVING.md`. **Versioned serving/routing contracts**, not a verified autonomous multi-model organism.
- Agent task state, scheduler and tool registry: `src/twelve_six_agent_runtime/{task_state,scheduler,tools}.py`; `docs/PLAN5_*.md`. **Component reference implementations**, not proof of actual browser/OS/external tool deployment.
- Multimodal: `src/twelve_six_agent_runtime/multimodal.py`; `docs/PLAN5_SECTION9_MULTIMODAL.md`. **Typed contracts only**, not actual video/vision/audio devices.
- Model-to-model review: `src/twelve_six/teacher_council.py`; `docs/PLAN6_SECTION6_TEACHER_COUNCIL.md`. **Fixture-qualified gateway protocol**, not real remote teacher integrations or fact certification.
- Long-term organism architecture: `docs/LIVING_AGENT_DEVELOPMENTAL_COGNITION_VISION.md`, `docs/PROJECT_INSTRUCTIONS.md`. **Vision and contracts**, not release-ready user features.
- Integration Nika/Autopilot and physical Windows/NVDA: Plan 10 owner boundary; no new unverified cross-repository imports here.

## Delivery discipline

This addendum records the **owner's 2026-10-13 clarified product priorities** and explicit missing acceptance requirements. It must be decomposed and scheduled in the appropriate **existing unfinished** sections of Plans 4/5/6/9/10 or versioned future research; no retroactive status changes or ownership collision. At each claimed implementation require exact SHA, CI/local check receipts, resource measurements, independent acceptance and actual user-visible end-to-end behavior. Record "planned", "reference-only", "candidate", "verified" separately. The baseline and source-data/trainability gates take precedence over speculative self-training claims.  
