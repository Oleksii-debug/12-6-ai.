# Plan 5 / Section 8 — semantic browser and computer-control reference

**Owner:** Plan 5 / agent-facing contracts; conflict keys `agent-runtime` and `tool-contracts`. This is not Nika Core, a browser plugin, an OS automation driver, a Windows UI package or permission to run remote effects.

## Canonical reference

`src/twelve_six_agent_runtime/ui_control.py` reuses the unique Plan-5 Section-7 `ToolRegistry` and Section-2 `TaskStore`, with a single trusted-host-registered, fixed `ui.perform` `ToolDescriptor`.

- **Semantic observation and identity:** `UISnapshot` carries a backend (`dom`, `ax`, or `uia`), application/window/view IDs, generation, tree hash, provenance evidence, and bounded typed nodes (node ID, role, accessible name, state). A separately trusted host verifies the observation. Ambiguous or missing role/name results require explicit stable node identity; never silently choose the first matching element.
- **Host-authorized intent:** `prepare()` binds the exact tree/node digests, requested action, expected postcondition and method to a versioned typed tool call, its task/effect ID, epoch/revision, explicit grant evidence, and host permission verification. No actual browser/OS dispatch occurs inside this module.
- **Fresh target preflight:** `issue()` compares a separately trusted fresh current snapshot with the exact captured identity immediately before using `TaskStore.issue_effect`. A stale node, changed view/epoch, duplicate issue or forged intent fails closed. The trusted adapter must issue before any external action; if the process fails after issuing, its effect remains `unknown`, not eligible for automatic retry.
- **Postcondition and result proof:** `complete()` accepts a typed result only if expected postcondition is independently observed and both call and execution evidence are verified by the host. `ToolRegistry.accept_result` durably records the receipt via the sole TaskStore authority.
- **Ambiguous/restart reconciliation:** `reconcile_unknown()` resolves an unknown effect (including after a legitimate control epoch bump) only with independent trusted evidence binding the exact original intent, current task, external receipt and expected observed postcondition. It never repeats the effect.
- **Fallback never primary:** `coordinate` and `vision` are opt-in fallback *methods*, not selectors. They still require an exact semantic target and separately attested fallback evidence, and this evidence ID is bound into the typed call. Raw screen-coordinate targets are unsupported.
- **Scope:** no provider credentials or real DOM/UIA/Win32 execution, UI model bypass, physical Windows/NVDA claim, product-specific Nika state, paid infrastructure or final system qualification. The host adapter is interchangeable only if it preserves this contract.

## LOCAL_FREE qualification

```bash
PYTHONPATH=src python -m compileall -q src/twelve_six_agent_runtime
PYTHONPATH=src pytest -q tests/test_agent_ui_control_section8.py tests/test_agent_tools_section7.py tests/test_agent_task_state_section2.py
```

Tests include positive DOM/AX/UIA paths, ambiguous semantic targets, stale-view refusal before issue, unsupported backends/actions, corrupt/malformed tree identity, forged call/target/postcondition, evidence-less fallback refusal, independent grant/receipt failure, restart/epoch fencing, unknown-effect reconciliation without replay, and duplicate completion resistance.

This is component-level **reference adapter** evidence. Browser/computer controls remain fake host fixtures until a future Plan-10 product integration supplies real trusted providers and accessibility acceptance.
