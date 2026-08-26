# POSTBASE-256 Hypothesis Search V1

Worker: `POSTBASE-356-HYPOTHESIS-SEARCH-V1`, implementing the missing component originally planned as POSTBASE-256.

## Scope

This component is an additive post-Base reasoning substrate stacked on POSTBASE-255. It does not modify Base weights, tokenizer, trainer, checkpoint format, or Base evidence.

Runtime implementation is Python stdlib only and has no model, Torch, network, external-LLM, subprocess, or shell dependency.

## State model

Each hypothesis records:

- stable deterministic ID;
- statement;
- explicit assumptions;
- optional parent hypothesis ID;
- status: active, rejected, or revised;
- complete score history with sequence, reason, and optional evidence link;
- linked evidence IDs;
- linked contradiction IDs;
- linked critiques;
- linked deterministic tests.

Evidence records support or contradiction, weight, hard/soft status, source, and owning hypothesis. Contradictions are first-class records linked back to the exact evidence that created them. Rejection never erases the rejected hypothesis, its assumptions, evidence, contradictions, or score history.

## Operations

`HypothesisSearch` supports:

- `propose`: create an independent candidate hypothesis;
- `branch`: create a competing child while retaining the parent;
- `critique`: record a critique and optionally adjust score;
- `test`: compare a declared prediction with an objective observed value and emit support/contradiction evidence;
- `reject`: explicitly reject an active hypothesis, optionally binding the rejection to evidence owned by that hypothesis;
- `revise`: create a new child hypothesis from an active or rejected parent without rewriting history.

The search selector considers active hypotheses only and deterministically ranks by score then ID. More search work is not represented as proof of better reasoning.

## Objective falsification fixture

The deterministic probe deliberately starts with the wrong hypothesis as preferred:

- wrong: “addition is evaluated before multiplication”, initial score `0.85`;
- correct: “multiplication is evaluated before addition”, initial score `0.60`.

The objective code fixture uses ordinary Python arithmetic:

`2 + 3 * 4 == 14`

The wrong hypothesis predicts `20`, receives hard contradiction evidence, and is explicitly rejected. The correct hypothesis predicts `14`, receives support evidence, and becomes the final preferred hypothesis. The probe also exercises branch, critique, and revise on the same search graph.

The fixture is logic/code evidence, not an LLM judgment and not a broad reasoning-capability claim.

## Truth boundary

This is hypothesis-state/search mechanics only. It does not claim autonomous scientific discovery, general intelligence, calibrated probabilities, or correctness outside supplied evidence/tests. Scores are bounded deterministic search priorities, not statistical posterior probabilities.

Execution profile: `LOCAL_FREE` only. No external LLM and no unrestricted runtime shell.
