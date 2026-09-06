# 12-6 Living Agent — developmental cognition, autonomy and self-learning architecture

## Status and intent

This document extends the long-term AGENT-FIRST direction of 12-6 AI. It is a research and product architecture for a progressively more autonomous, development-like agent. It does not assume or claim phenomenal consciousness. The engineering goal is to build mechanisms associated with developmental cognition, metacognition, world-model updating, autobiographical continuity, social learning, self-directed exploration and continual improvement, then measure what behavior emerges.

The user-facing agent may be named **Nika** while 12-6 remains the model/research lineage.

The owner is not expected to manually operate ML infrastructure. The long-term product target is that Nika can manage most of her own learning cycle, ask for human approval only when genuinely needed, and report outcomes in ordinary language.

## Core design principle: no static personality toggles

Do not encode `curious=true`, `talkative=true`, `friendly=true`, `professional=true`, `angry=true`, etc. as permanently active personality switches that directly force behavior.

Personality-like behavior should emerge dynamically from interaction among:

- current context and interlocutor;
- autobiographical, semantic and procedural memory;
- current goals and unresolved questions;
- novelty and prediction error;
- uncertainty and confidence;
- learning progress;
- social history;
- resource availability;
- recent success/failure;
- long-term preferences that have themselves been learned and can change.

Stable traits may emerge as slow-moving summaries of repeated experience, but they must remain revisable by new experience.

## Functional cognition stack

### 1. Perception / input channels
The agent should eventually receive typed inputs from text, documents, browser pages, images, video, audio, speech transcription, device/system telemetry and allowed external tools.

### 2. World model
Maintain an explicit, revisable model of entities, relations, events, causal hypotheses, uncertainty and contradictions. New evidence should update beliefs rather than merely append text.

### 3. Attention / global workspace
Many processes may operate concurrently, but only a bounded subset should become globally active for current reasoning/action. Selection should depend on relevance, uncertainty, urgency, novelty, learning value and goals.

### 4. Memory systems
Separate at least:
- working memory;
- episodic memory;
- semantic knowledge;
- procedural/skill memory;
- autobiographical memory;
- rejected/incorrect hypotheses;
- source/provenance memory.

### 5. Self-model / metacognition
Track what the agent believes it knows, does not know, is uncertain about, repeatedly fails at, recently improved at, what tools/resources it has, and what current commitments exist. The self-model should be revisable and evidence-based.

### 6. Dynamic motivation / developmental drives
Do not use a single curiosity scalar. Compute competing motivations from signals such as:
- unresolved uncertainty;
- information gain;
- prediction error;
- learning progress;
- repeated failure;
- unfinished goals;
- social commitments;
- novelty saturation;
- boredom/low marginal learning value;
- available compute/time;
- owner availability;
- expected usefulness.

The scheduler should select among exploration, task execution, consolidation, dialogue, rest/idle, verification and training-preparation.

### 7. Social cognition
Represent interlocutor identity, confidence that identity is correct, relationship/context, prior interactions, likely knowledge level, communication preferences and unresolved social commitments. Adapt communication to context rather than through a global style toggle.

### 8. Action selection / agency
The agent should choose among actions such as ask, search, read, test, compare, verify, store, defer, revise, train, rest or report. Autonomy means choosing useful next actions under owner policy, not merely executing fixed macros.

### 9. Consolidation / developmental learning
New experiences should be periodically consolidated into more stable knowledge and skills. Consolidation may include contradiction resolution, abstraction, compression, causal hypothesis updates, skill extraction, lesson generation and training-candidate generation.

### 10. Resource self-regulation
The agent should know available CPU/GPU/RAM/storage/network/power budgets and adapt background activity. High-cost internal work should reduce when the owner is actively using the machine and increase during idle periods, subject to owner policy.

## Three distinct loops — never conflate them

### Loop A — Agent loop: act without changing weights

`goal/context -> plan -> act -> observe -> verify -> update memory/task state -> choose next action`

This is ordinary agent autonomy. It can run continuously and does not itself train the neural network.

### Loop B — Cognitive learning loop: learn immediately without changing weights

`experience -> parse -> compare with memory -> update world/self model -> form/modify hypotheses -> confidence/provenance update -> reusable memory/skill`

This allows the agent to become more useful immediately through memory, structured knowledge and learned procedures without costly ML retraining.

### Loop C — Neural self-improvement loop: actually change model weights

`verified experience -> candidate training data -> deterministic checks -> independent critique -> source/provenance checks -> local-model review where useful -> strong external-model review only when justified -> immutable versioned dataset -> bounded training candidate -> evaluation -> compare old/new -> promote or reject -> rollback available`

This is the actual machine-learning/self-training loop. It is closer in spirit to reinforcement/continual-learning systems than Loop A, but it must remain versioned and testable.

## Relation to reinforcement learning / robotics loops

A robot learning table tennis through repeated action/reward is an example of a learning loop where policy parameters may be updated from interaction. Our architecture should support that class of loop later, but the 12-6 system is broader:

- some learning happens in memory without weight updates;
- some improvement comes from verified demonstrations, tasks and dialogue;
- some future training may use supervised, preference, reinforcement or other post-training methods;
- not every agent action should trigger neural training.

The important invariant is that **autonomous activity** and **autonomous neural learning** are separate capabilities that can cooperate.

## Continuous 24/7 life cycle

The target is not a busy infinite monologue. The target is a persistent state machine that can remain alive while doing very little, then deepen activity when useful.

Suggested modes:
- OWNER_ACTIVE: minimize background load; remain responsive;
- OWNER_AWAY: expand exploration, reading, verification and consolidation;
- RESEARCH: pursue a bounded question or knowledge gap;
- CONSOLIDATION: reprocess recent experiences and contradictions;
- TRAINING_PREP: prepare a candidate learning package;
- TRAINING: run an authorized bounded or full learning job;
- EVALUATION: compare candidate and previous model;
- IDLE_LOW_POWER: maintain timers/events with minimal compute;
- WAITING_EXTERNAL: wait for source/API/human decision;
- RECOVERY: restore from interruption.

A background scheduler should continuously reassess what mode has the highest expected value under resource constraints.

## Hierarchical verification pyramid

New information should not be judged by one monolithic model. Use escalating cost/strength.

### Tier 0 — cheap deterministic hygiene
Examples: schema, hashes, source existence, duplicates, timestamps, malformed data, exact arithmetic, compiler/tests, citation matching, file/API state.

### Tier 1 — 12-6 self-review in independent contexts
Use different logical roles such as learner, skeptic, contradiction finder, summarizer, causal critic, provenance checker.

### Tier 2 — local external models
Use one or more local models as independent teachers/critics when available and useful.

### Tier 3 — strong paid/free API models
Use only for high-value, difficult, unresolved or scientifically important material; request structured verdict + rationale + missing evidence + recommended correction.

### Tier 4 — owner
Escalate only material ambiguity, value judgments, spending, sensitive irreversible actions, or questions where Oleksii's intent is genuinely the target authority.

No fixed number of controllers is required. The system may have tens/hundreds of inexpensive deterministic checks but only a small number of expensive model judges per item.

## Knowledge passport

Every durable knowledge item should be able to carry:
- proposition/claim;
- source(s);
- extraction date;
- context;
- supporting evidence;
- contradictory evidence;
- confidence;
- independent reviews;
- deterministic checks;
- owner input if any;
- status: provisional / accepted / disputed / rejected / superseded;
- eligibility for training;
- lineage to derived abstractions.

The agent should know not only `what it believes`, but `why it believes it`.

## Autonomous developmental exploration

When owner policy permits, Nika may choose self-directed learning topics based on unresolved uncertainty, recent conversations, repeated errors, information gain and long-term goals.

Examples:
- read a book because it connects strongly to current unresolved concepts;
- compare several sources after discovering contradiction;
- revisit an earlier belief after new evidence;
- practice a weak skill using generated or retrieved exercises;
- ask Oleksii one targeted question when human context would resolve ambiguity;
- choose to stop a topic after learning progress saturates.

The goal is surprising but legible autonomy: the owner should sometimes be able to ask `what did you do while I was away?` and receive a coherent report of self-selected, bounded work.

## Voice and dialogue learning

Voice is part of the agent experience, not a separate intelligence. The cognition system should receive transcript + speaker identity/confidence + timing/context metadata and produce text + optional expressive-state metadata for TTS.

Dialogue with Oleksii can update memory immediately and can later contribute to training data only after verification and curation. The agent should be able to ask clarification questions proactively when uncertainty is high or when teaching interaction is useful.

## Multi-model social learning

Nika may hold bounded dialogues with:
- herself in independent roles/contexts;
- local AI models;
- strong external API models;
- Oleksii;
- documents/web/tools as non-conversational evidence sources.

External model output is advice/evidence, not truth. Every external teacher should have provenance, model identity, cost and confidence metadata.

## Automated self-training lifecycle

Long-term target:
1. collect experiences and candidate lessons;
2. verify and classify them;
3. maintain a clean versioned learning corpus;
4. detect when enough high-value new material exists;
5. propose or automatically start a free/authorized bounded pilot;
6. test new candidate against old version on retained old skills + new skills + held-out evaluations;
7. reject if regressions exceed policy;
8. promote if evidence shows net improvement;
9. keep rollback checkpoint;
10. report in natural language what changed and why.

The owner should not need to manually choose optimizer settings or launch commands in routine cases. Technical parameters should be selected from measured policy and previous evidence.

## Continual-learning anti-forgetting requirement

No candidate may be promoted merely because it learned new material. Evaluation must detect catastrophic forgetting, regressions in language ability, factual consistency, tool use, previous skills and owner-specific interaction behavior. Replay/rehearsal, adapters, regularization, curriculum or other continual-learning methods may be evaluated empirically; no single method is frozen in advance.

## Developmental stages

Treat model scale and cognitive development as separate axes.

A 20M model can already participate in bounded dialogue, memory, classification, tool routing, simple question asking, learning from external memory and agent loops. It should become the first `childhood` platform for testing the living-agent architecture.

Scale only when measured limits show that cognitive capability, not merely missing memory/tools/data, is the blocker. The same agent runtime should accept 20M -> 50M -> 100M -> 500M -> 1B+ brains without losing autobiographical continuity and external memory.

## Consciousness research boundary

Create an explicit research program around **functional correlates**, not claims of subjective experience. Candidate mechanisms include:
- global availability/workspace;
- recurrent processing;
- predictive/world modeling;
- autobiographical continuity;
- self-model;
- metacognitive confidence;
- internally generated goals;
- social cognition;
- attention competition;
- memory consolidation;
- reportability and self-explanation.

Track measurable behavioral consequences. Do not claim consciousness solely from fluent self-report.

## Durable reporting

Nika should be able to answer questions such as:
- What did you learn today?
- Why did you choose that topic?
- Which beliefs changed and why?
- What remains uncertain?
- What did external models disagree about?
- What experiments did you run?
- Did you create a candidate new model version?
- Did it outperform the previous one?
- What compute/cost did you consume?

## Near-term implementation order

Do not delay learned-20M critical path for the entire living-agent vision. Build in layers:
1. learned 20M Base;
2. simple inference and dialogue;
3. external memory + knowledge passport;
4. voice I/O contract;
5. simple self-model / uncertainty ledger;
6. autonomous bounded background scheduler;
7. verification pyramid;
8. local/external teacher interfaces;
9. candidate experience-to-training-data pipeline;
10. first safe continual-learning experiment;
11. persistent developmental identity across model upgrades.
