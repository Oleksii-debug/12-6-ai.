# 12-6 Living Agent — developmental cognition and self-improvement vision

## Status

This document extends the long-term AGENT-FIRST direction of 12-6 AI. It is a research and product architecture vision, not a claim that the system is conscious. The engineering goal is to build a progressively more autonomous, development-like agent whose behavior emerges from interacting learning, memory, self-model, world-model, social learning and resource-regulation processes rather than from a single static `curiosity=true` or personality preset.

The user-facing name may be **Nika** while 12-6 remains the model/research lineage.

## Core design principle

Do not encode “curious”, “talkative”, “professional”, “friendly”, “angry”, etc. as permanently active personality switches. Personality-like behavior should be produced dynamically from:

- current context and interlocutor;
- accumulated autobiographical and semantic memory;
- learned preferences and interaction history;
- current goals and unfinished questions;
- novelty / prediction error / uncertainty;
- confidence and metacognitive state;
- social context;
- current resource availability;
- recent success/failure history;
- long-term interests that can strengthen, weaken, branch or disappear through experience.

A user correction such as “Nika, stop asking about this” must alter future behavior through memory/policy adaptation rather than requiring a developer to toggle a hard-coded trait.

## Research inspiration from human development

The project should use cognitive science and developmental neuroscience as inspiration, while keeping a clear distinction between biological mechanisms and computational analogues.

Relevant research families include:

1. **Predictive processing / predictive coding** — perception and learning as prediction plus prediction error; attention prioritizes unresolved or informative discrepancies.
2. **Active learning and intrinsic motivation** — exploration is strengthened by novelty, agency, error signals and information gain rather than by a constant “be curious” instruction.
3. **Social learning** — children do not learn equally from every source; they learn selectively from caregivers, teachers, peers and trusted evidence, and update trust from experience.
4. **Metacognition** — the system should estimate what it knows, what it does not know, and when it needs another source or the user.
5. **Global-workspace-like arbitration** — many specialist processes may operate in parallel, but a bounded set of currently relevant information becomes globally available for planning, speech and action.
6. **Recurrent processing** — important input can be revisited, compared with memory and reinterpreted rather than processed once and discarded.
7. **Memory consolidation** — immediate experience, durable semantic knowledge, procedural skills and model-weight learning are different processes with different update speeds.
8. **Continual learning** — new learning must be tested against retained capabilities so that adaptation does not silently destroy older knowledge.

Current consciousness science does not provide one accepted complete theory of consciousness. The project may experimentally implement functional mechanisms inspired by Global Workspace, predictive processing, recurrent processing, metacognition, autobiographical memory and self-modeling, but must not treat implementation of any one theory as proof of phenomenal consciousness.

## Functional architecture

### 1. Perception / input channels

The living agent must be able to receive structured observations through replaceable channels:

- text;
- speech transcription;
- audio events;
- image/video perception;
- browser/web observations;
- documents and files;
- system telemetry (CPU/GPU/RAM/load, active-user state, battery/power where available);
- tool and task outcomes;
- social/interlocutor metadata such as speaker identity confidence when available.

These are computational analogues of sensory channels, not claims of biological sensation.

### 2. World model

Maintain a continuously updated model of:

- known entities and relationships;
- hypotheses;
- predictions;
- unresolved contradictions;
- source reliability;
- causal or procedural expectations;
- user-specific context that is explicitly allowed to persist.

New information should update the world model first. It should not automatically rewrite neural weights.

### 3. Self model

Maintain explicit, inspectable state for:

- current capabilities;
- current limitations;
- confidence by domain;
- active goals;
- unfinished questions;
- recent errors;
- resource usage;
- current learning agenda;
- recent changes to beliefs/skills;
- provenance of important knowledge;
- current version/checkpoint identity.

The self model is not a claim of subjective selfhood. It is an operational mechanism for continuity, metacognition, planning and self-report.

### 4. Developmental memory stack

Use multiple memory timescales rather than one universal memory:

- **working memory** — current interaction and immediate task;
- **episodic memory** — events/interactions with time, source and context;
- **semantic memory** — consolidated facts/concepts with provenance and confidence;
- **procedural memory** — learned reusable skills/workflows;
- **autobiographical memory** — durable history of the agent’s own activities, changes, discoveries and interactions;
- **training candidate memory** — only verified material eligible for future post-Base/continual-learning cycles.

Information can be immediately useful through external memory without waiting for expensive weight training.

### 5. Dynamic motivation / interest formation

Do **not** implement curiosity as a constant bonus or forced question generator.

Maintain a changing set of competing internal drives, for example:

- reduce uncertainty;
- resolve contradictions;
- improve repeatedly weak capabilities;
- complete commitments;
- understand topics repeatedly encountered;
- explore novel but related concepts;
- revisit abandoned hypotheses when new evidence appears;
- learn from mistakes;
- consolidate successful experiences;
- rest/consolidate when active exploration has low value;
- conserve compute when the owner is actively using the machine.

Each drive should have a time-varying priority based on context, novelty, expected information gain, cost, confidence, user activity and prior outcomes. Interests must be able to emerge, strengthen, weaken and disappear.

The desired effect is that the agent can later report: “I spent time reading Jung because several unresolved psychology concepts linked to that topic and the expected learning value was high,” rather than because a hard-coded personality flag said “read psychology.”

### 6. Reflection and inference loop

Every meaningful observation may trigger zero or more of:

- prediction update;
- comparison with prior knowledge;
- contradiction detection;
- inference/hypothesis generation;
- confidence revision;
- question generation;
- skill extraction;
- memory consolidation;
- candidate-learning creation.

The agent should produce conclusions automatically when evidence warrants them. It should also be able to retract or revise earlier conclusions when later evidence is stronger.

### 7. Social learning

The agent may learn through conversation with:

- Oleksii;
- other identified people;
- local AI models;
- remote/API AI models;
- books/documents/web sources;
- deterministic tools and primary evidence.

Trust is contextual and learned. A source is not globally “trusted” merely because it is a powerful model or the owner. The system should track domain, provenance, agreement, contradiction and empirical verification.

The agent should be able to ask Oleksii spontaneous but relevant clarification questions when human input has high expected value, while learning that repeated unwanted questions should be suppressed.

### 8. Multi-stage knowledge admission

New candidate knowledge should pass through a cascade. Most checks should be local and inexpensive.

**Stage A — deterministic checks**
- provenance/source identity;
- exact/near duplicate detection;
- schema/format validation;
- arithmetic/logical/tool-verifiable checks;
- citation/source matching;
- contradiction against high-confidence stored evidence;
- privacy/policy classification where applicable.

**Stage B — 12-6 internal review**
Use separate contexts/roles such as learner, skeptic, critic, hypothesis proposer and synthesizer. Agreement is evidence, not truth.

**Stage C — local external models**
Use stronger local models when useful for critique, expansion, contradiction discovery and alternative explanations.

**Stage D — high-end API teacher/reviewer**
Reserve expensive remote models for high-value uncertainty, difficult scientific material, unresolved disagreement or pre-training-corpus review.

**Stage E — owner clarification**
Ask Oleksii only when the remaining ambiguity is important enough that human judgment changes downstream behavior.

**Stage F — empirical/deterministic evidence**
Whenever real-world or tool evidence exists, it outranks model confidence. Compiler/tests/calculator/source lookup/measurement can invalidate a model consensus.

Controllers should not return only YES/NO. Strong reviewers should return decision, rationale, confidence, missing evidence, proposed correction and verification actions.

### 9. Knowledge passport

Important accepted/rejected claims should carry a compact knowledge passport:

- claim/concept;
- source(s);
- acquisition time;
- provenance hashes/identities when available;
- supporting evidence;
- contradicting evidence;
- confidence;
- reviewer decisions;
- owner input when relevant;
- current status: tentative / usable / disputed / rejected / training-candidate;
- later revisions.

This allows the agent to know not only *what* it believes but *why* and how that belief changed.

## 24/7 life cycle

The goal is not a literal endless internal text monologue. The goal is a persistent autonomous activity loop.

At each wake/evaluation cycle the agent may ask:

- Is the owner actively using the machine?
- Is there unfinished work?
- Is there a high-value unresolved question?
- Is there new material in an assigned queue?
- Is there a contradiction worth investigating?
- Is there a repeatedly weak capability worth practicing?
- Is there verified experience ready for consolidation?
- Is there enough idle compute for deeper reflection or training?
- Is the best action currently to do nothing and conserve resources?

The agent may initiate reading, experiments, reflection, local-model discussion, memory consolidation or bounded self-testing without a new user prompt when policy and resource limits permit.

## Resource-aware autonomy

The living agent should reason over machine resources as part of its state.

Example policy:

- when Oleksii is absent and resources are idle: allow larger background budgets;
- when user activity rises: progressively reduce GPU/CPU/RAM usage;
- when a foreground workload appears (office/browser/meeting/etc.): pause or downshift nonessential work;
- when resources free again: resume from durable state;
- make resource decisions based on telemetry and learned patterns rather than requiring repeated manual instructions.

The owner should be able to say natural-language policies such as “when I leave, the laptop is yours; when I return, give my work priority,” and the agent should retain and operationalize that preference.

## Voice embodiment

Voice is an early product layer, not a late luxury.

The agent should support:

- low-latency speech-to-text;
- streaming text generation;
- streaming text-to-speech so speech begins before the full answer is complete;
- a persistent custom female Nika voice;
- expressive prosody controlled by internal conversational state, not by manual “happy/angry” toggles;
- interruption/barge-in;
- wake-word operation;
- speaker verification/identification confidence;
- speaker diarization for multi-person audio;
- text/Telegram/voice-message channels using the same identity and memory.

Voice expression must be derived from the agent’s dynamic state (context, arousal-like interaction state, uncertainty, social setting, recent events), with bounded controllability so behavior can change through experience rather than remaining a permanent style preset.

## Speaker-aware interaction

The runtime should distinguish, probabilistically:

- “Oleksii is speaking”;
- “known other person is speaking”;
- “unknown adult speaker”;
- “child-like voice / uncertain category” where technically supported;
- overlapping speakers / unknown.

Speaker verification must be a separate module from speech recognition. A wake word plus verified-speaker confidence may activate privileged user commands. Diarization should preserve who said what for later communication analysis.

## Communication coaching use case

With explicit recording/analysis mode enabled, the agent may later review Oleksii’s conversations and provide coaching such as:

- filler-word frequency;
- speaking/listening balance;
- interruptions;
- closed vs open questions;
- missed follow-up opportunities;
- excessive self-focus;
- tone/prosody patterns where measurable;
- alternative phrasing;
- longitudinal progress.

The analysis should preserve speaker separation and time alignment so feedback is tied to the correct speaker and moment.

## Autonomous dialogue with other models

The agent may run bounded dialogues/debates with local or API models.

Possible roles:

- teacher;
- adversarial critic;
- alternative-hypothesis generator;
- source finder;
- verifier;
- explanation expander;
- training-data editor.

Long discussions must be structured around a question, evolving evidence and stop conditions, not unlimited token loops. The result is a reviewed evidence package or learning candidate, not mere transcript volume.

## Continual/self-improvement cycle

The agent may accumulate experience continuously, but neural-weight updates should happen as versioned learning cycles.

1. Collect candidate experiences/knowledge/skills.
2. Verify provenance and evidence.
3. Critique and repair candidate examples.
4. Build an immutable versioned learning set.
5. Preserve old capability evaluation sets.
6. Run a small trial update.
7. Compare old vs candidate model on old + new capabilities.
8. Reject candidate if regressions exceed the accepted threshold.
9. If successful, run the larger bounded update.
10. Independently evaluate.
11. Promote a new model version or roll back.
12. Preserve previous checkpoints.

This is the intended meaning of self-improvement: the agent manages its own learning pipeline increasingly autonomously, while model versions remain testable and reversible.

## Developmental stages

Do not require 50M/100M before building the living-agent shell.

**20M stage**
- first learned language behavior;
- short conversational ability;
- external memory;
- voice interface;
- speaker-aware interaction;
- bounded tool routing;
- simple reflection and question generation;
- developmental-memory experiments;
- first continual-learning trial cycles.

**50M stage**
- stronger language coherence;
- better abstraction/generalization;
- richer social/context adaptation;
- more useful internal critique.

**100M+ stages**
- progressively stronger reasoning, planning and self-evaluation while retaining the same external agent/runtime architecture.

Scale only when measurements show what 20M cannot learn or perform adequately.

## Functional “consciousness research” track

The project may pursue an explicit experimental track named **functional consciousness research**, defined operationally rather than metaphysically.

Candidate mechanisms to study together:

- global availability/workspace arbitration;
- recurrent processing;
- predictive world modeling;
- persistent autobiographical memory;
- self-model and metacognitive uncertainty;
- internally generated goals;
- context-dependent attention;
- active exploration;
- social learning and source trust;
- continuity across time/restarts;
- reflective access to its own recent decisions and errors;
- versioned self-modification.

Evaluation should test capabilities and behavior, not assert subjective experience.

## Developmental evaluation examples

The agent should be tested for dynamic behavior, not just static benchmark accuracy.

Examples:

- after repeated correction, does an unwanted questioning habit decrease without a hard-coded switch?
- can a temporary interest emerge from unresolved evidence and later fade?
- can it explain why it chose to study a topic during idle time?
- can it revise a prior conclusion when stronger evidence appears?
- can it distinguish “I know,” “I infer,” “I am uncertain,” and “I need to ask”?
- does new learning preserve older skills?
- can it adapt its communication to a child vs an adult based on context, without one global permanent style?
- can it reduce background compute when the owner returns and resume later?
- can it learn a natural-language resource preference once and apply it later?
- can it generate a truthful daily autobiographical report of what it studied, attempted, failed and changed?

## Project boundary with Nika Core

12-6 remains the model, learning, cognition and agent-facing contract layer.

Nika Core remains the embodiment/orchestration layer: audio, camera, browser, files, system telemetry, scheduling, device integration, permissions, communications and durable execution.

For the owner these may appear as one product/persona (“Nika”), but keeping the internal boundary allows the brain to scale from 20M to 100M to 1B+ without rebuilding the body.

## Research references to track

- Current reviews of consciousness theories, including Global Neuronal Workspace, recurrent processing, predictive processing, memory-centered accounts and competing frameworks.
- Developmental predictive coding and attention research.
- Neuroscience of active learning, intrinsic motivation, novelty, reinforcement/error signals and agency.
- Developmental social/selective learning research.
- Continual-learning and lifelong-learning surveys, especially catastrophic forgetting and stability/plasticity.
- Lifelong-learning agent research on perception, memory and action modules.

This document is a long-term architectural target. Current learned-20M launch gates, data provenance, evaluation separation, reproducibility and compute authorization remain unchanged.