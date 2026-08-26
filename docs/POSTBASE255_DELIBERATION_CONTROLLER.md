# POSTBASE-255 Deliberation Controller V1

This component is deliberately outside canonical Base. It changes no model
weights, ModelSpec, tokenizer, checkpoint format, training behavior, or
first-party Base generation semantics.

The controller accepts a model adapter, verifier, optional synchronous tool
executor, and explicit budgets for wall time, model calls, generated tokens,
tool calls, and candidate branches. Its loop is propose, verify/compare,
critique, revise, verify/compare, retain, repeat.

Every model and tool budget unit represents an actual synchronous invocation.
There is no simulated waiting. The controller stops early when verifier
score/confidence reaches the configured target or when retained-best scores
converge.

The wall deadline is passed into every model request and checked before every
new model/tool call. A compliant adapter must also honor that deadline inside
one in-flight invocation; the controller does not abandon a running in-process
call because doing so would leave untracked background work.

Private scratch is stored only in private candidate state. Public traces record
its SHA-256, not its text. Tool arguments/results are likewise represented by
hashes and sizes. The final artifact exposes only selected public candidate
text, verifier score/confidence, and the machine trace. Adapters remain
responsible for not copying private scratch into their public text field.

Trace schema `12-6.postbase-deliberation-trace.v1` records candidate versions,
scores, rejections, model calls, tool calls, budget consumed, stop reason, and
the selected final candidate.

The LOCAL_FREE probe compares a 3-model-call/2-candidate run with a
7-model-call/4-candidate run under a constant verifier. The larger budget must
execute more search work, while both keep the same quality score. This proves
that the controller does not encode the false assumption that more time or
compute automatically improves quality.

No external teacher API. No paid compute. No asynchronous background
reasoning. No canonical Base capability claim.
