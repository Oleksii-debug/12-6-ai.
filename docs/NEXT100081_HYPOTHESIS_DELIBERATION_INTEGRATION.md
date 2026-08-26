# NEXT100-081 Hypothesis/Deliberation Integration

This integration consumes the terminal POSTBASE-255 deliberation controller and terminal POSTBASE-256 hypothesis-search V1 without changing canonical Base model behavior.

The integrated controller is `HypothesisDeliberationController` in `src/twelve_six/postbase_hypothesis_deliberation.py`. It preserves POSTBASE-255 synchronous budget enforcement for model calls, generated tokens, tool calls, candidate branches, and wall time, while binding each public candidate to POSTBASE-256 hypothesis state.

The integration records multiple hypotheses, score histories, critique identifiers, deterministic evidence, contradictions, revisions, rejections, retained-best history, and exact local accounting for critiques/evidence tests/contradictions/rejections/revisions. Hard deterministic contradictions reject the affected hypothesis. The first rejected hypothesis may be revised, subject to the incumbent branch/model budgets. Retained-best selection is always taken from active POSTBASE-256 hypotheses.

Public trace policy is deliberately narrower than raw hypothesis-search export. Hypothesis statements, statuses, score histories, evidence summaries, contradiction summaries, critique IDs, and test pass/fail state may be public. Critique text, test prediction/observation payloads, tool payloads, and model private scratch are not published. The terminal POSTBASE-255 controller does not place private scratch or a private-scratch hash into its public trace.

The objective fixture begins with a wrong preferred hypothesis: `addition before multiplication` receives verifier score 0.85, while `multiplication before addition` begins at 0.60. Deterministic local evaluation of `2 + 3 * 4` observes 14. The wrong hypothesis predicts 20, receives hard contradiction evidence, and is rejected to score 0.0. The correct hypothesis predicts 14 and rises to 0.90. A revision of the rejected branch is also generated, but its lower score does not replace the retained correct branch.

LOCAL_FREE validation uses only deterministic project-authored adapters and Python tests. No external teacher API, external LLM, paid compute, asynchronous worker, or hidden long-running search is required.

Consumed authority heads at integration construction time:

- POSTBASE-255: `486bd91ca03bed41750c638d702f557f320b780a`
- POSTBASE-256: `ea1d8fff0d3235660dffe7ba411e192df83f5e1d`

The integration commit is constructed with both authority heads as parents and restores the exact terminal POSTBASE-255 source/test/doc/probe blobs while retaining POSTBASE-256 hypothesis-search files.
