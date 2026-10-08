# Model-neutral agent runtime

This package is outside 12-6 model weights. It does not alter Base behavior, add instruction tuning, or claim that the current tiny model can autonomously engineer software.

## Current boundary

`src/twelve_six/agent_runtime/` provides a model-neutral orchestration seam:

- proposer: any future checkpoint adapter or external test double may emit `Proposal` + structured `ToolCall` objects;
- executor: resolves registered tools, stops on failure, binds every result to a trace ID, and writes JSONL execution logs;
- verifier: separate interface and execution phase; executor success is never itself a verified success;
- selector: selects only independently verified candidates;
- dataset builder: serializes selected episodes with `DATASET_RECORD_ONLY_NOT_MODEL_UPDATE`; it does not call Trainer or mutate weights.

The intended later flow is:

`Proposer -> Executor -> Verifier -> Selector -> dataset builder -> Trainer`

The last arrow is intentionally absent here. D09/post-training owns any future decision to consume these records.

## Tools and isolation

The initial tool set is deliberately narrow:

- `files`: UTF-8 read/write/existence operations restricted to a workspace root. Absolute and escaping paths fail closed.
- `terminal`: argv-only subprocess execution, no shell, configured executable allowlist, bounded timeout, cancellation, captured stdout/stderr/exit code.
- `git`: fixed allowlist of local repository operations (`init`, `status`, `diff`, `diff_cached`, `add_all`).
- `browser_mcp`: adapter seam only. `DeterministicMockMCP` supports hermetic tests; no browser dependency or live network authority is introduced.

The workspace boundary is filesystem path isolation, not an OS/container security sandbox. A production executor should place the same tool contract behind a stronger container/VM sandbox before accepting untrusted model-generated commands. The current terminal allowlist is therefore mandatory rather than treating a temporary directory as a security boundary.

There is no self-rewrite primitive. Tools operate on the supplied workspace, not on the installed 12-6 source tree unless a caller deliberately chooses that tree as the workspace.

## Timeouts, cancellation, logs, results

Every tool returns `ToolResult` with stable success/error fields. Terminal processes have a configured maximum timeout and cooperative runtime cancellation. Execution is trace-bound and appended to `.agent_runtime/execution.jsonl` inside the workspace.

The current trace ID is an orchestration identity, not a cryptographic provenance proof. Higher-assurance runs should additionally bind source SHA, model/checkpoint identity, tool implementation identity, environment lock, and immutable artifact hashes.

## Real toy development proof

Run:

```bash
PYTHONPATH=src python -m twelve_six.agent_runtime.toy_workflow --workspace /tmp/12-6-agent-toy
```

The deterministic proposer does not use any model weights. It performs a real isolated development sequence:

1. initializes a Git repository;
2. writes `calculator.py`;
3. writes a unit test;
4. executes the unit test through the terminal tool;
5. inspects Git status;
6. invokes a separate verifier that reruns the test;
7. emits a selectable candidate and dataset-record structure without training anything.

This proves orchestration mechanics only. It is not evidence of autonomous software-engineering capability.

## Audit findings and scale-forward decisions

Current 12-6 infrastructure already has substantial D05/D07 checkpoint/inference/server work. Agent orchestration should not be folded into those model-serving modules because doing so would couple tool authority to Base semantics and create later migration pressure when checkpoints change.

The durable interfaces to preserve across 100K -> 1M -> 10M -> 100M+ stages are structured proposals, stable tool schemas, external workspace authority, trace identity, verifier independence, and model adapter neutrality. A future checkpoint should be swappable behind `Proposer`; tool implementations and verifier policy should not need model-weight changes.

Before this runtime is allowed to operate on real repositories or network services, add an OS-level sandbox/container executor, explicit resource quotas, secret redaction policy, immutable source/checkpoint/environment binding, and an auditable MCP capability allowlist. Those are runtime controls, not Base-model behavior.
