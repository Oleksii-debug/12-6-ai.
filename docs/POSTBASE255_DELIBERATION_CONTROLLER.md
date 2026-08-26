# POSTBASE-255 Deliberation Controller V1

This component is deliberately outside canonical Base. It changes no model weights, ModelSpec, tokenizer, checkpoint format, training behavior, or first-party Base generation semantics.

The controller accepts a model adapter, verifier, optional synchronous tool executor, and explicit budgets for wall time, model calls, generated tokens, tool calls, and candidate branches. Its mechanically verified loop is propose -> verify/compare -> critique -> revise -> verify/compare -> retain.

Every model and tool budget unit represents an actual synchronous invocation. There is no simulated waiting. Generated-token and candidate-branch counters are also fail-closed. The controller stops immediately when a verified candidate reaches the configured score/confidence target, including during the initial proposal fan-out.

The wall deadline is passed into every model request and checked before and after every model/tool call. A synchronous invocation that returns after the deadline is still counted as consumed work but its response/result is not accepted into a candidate or reused by a later turn. The controller never leaves untracked background work.

Private scratch is never copied into the public trace, even as a hash, and is not retained in candidate state. Public model-call records contain only stage/identity metadata, generated-token counts, public-response hashes, requested tool-call counts, and measured duration. Tool arguments/results are represented by hashes and sizes rather than payload text. Adapters remain responsible for keeping private scratch out of their public `text` field.

Trace schema `12-6.postbase-deliberation-trace.v1` records candidate versions, verifier scores/confidence, comparisons/rejections, model calls, tool calls, budget consumed, stop reason, and the selected retained candidate.

The LOCAL_FREE probe compares a 3-model-call/2-candidate run with a 7-model-call/4-candidate run under a constant verifier. The larger budget must execute strictly more real model calls and create strictly more verified candidate branches while both runs retain the same quality score. This proves only that additional budget performs more search work; it deliberately does not claim that larger budget improves quality.

## POSTBASE-357 compatibility intake

Before terminalization, the independent terminal verifier authority was re-read at head `7eac24e250c0853745208bab8ba9b2d3d104fbf5`. Its independently pinned production verification blob is `e3c0504c1a6b3768c8aaea1aaaa3b3eb637eaab7`, and its exact-head convergence workflow is terminal success.

POSTBASE-357 exposes an evidence-object API, `verify(VerificationRequest) -> EnsembleResult`, with categorical `PASS` / `FAIL` / `INCONCLUSIVE` / `CONFLICT` outcomes and deterministic-failure precedence. POSTBASE-255 intentionally exposes a text-level `evaluate(task, text, branch_id, iteration) -> Verification` seam. Therefore the terminal verifier is not silently substituted: integration requires an explicit evidence-builder/adapter and an explicit status-to-ranking policy. POSTBASE-255 does not invent that policy, which prevents a categorical verifier result from being weakened or mis-scored merely to obtain direct API compatibility.

No terminal POSTBASE-356 hypothesis-search authority was present at the convergence intake point, so no newer hypothesis-search interface is claimed as consumed.

No external teacher API. No external LLM. No paid compute. No asynchronous background reasoning. No canonical Base capability claim.
