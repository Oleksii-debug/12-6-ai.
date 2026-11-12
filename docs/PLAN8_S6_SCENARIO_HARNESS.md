# Plan 8 Section 6 — versioned scenario harness

The three manifests under `configs/plan8/scenarios/v1/` pin deterministic
component fixtures for the reserved evaluation firewall, an agent goal and
observation, and a SIL synthetic journey. Each declares a capability,
prerequisite, step adapter, expected observation digest and required Q2 level.

`parse_scenario_bytes` rejects ambiguous or oversized manifests.
`run_scenario` takes a repository root and expected Git SHA. It verifies the
checkout before execution and after each adapter, then binds the receipt to the
scenario and exact source. It rejects missing prerequisites, changed
observations and adapter failures. Built-in adapters use the verified root.

The reserved evaluation adapter checks the fixture's no-training and
no-tokenizer-fit flags. The SIL adapter calls the accepted scenario parser.
The agent adapter runs the accepted Plan 5 fixture runtime, reopens its durable
state and records the completed task, verified effect, memory evidence and
dispatch count. Its temporary state is outside the checkout.

`coverage_map` takes `expected_source_sha` and rejects receipts from any other
candidate. It lists the scenarios still missing evidence for each capability.
All three checked-in journeys require Q2 and currently produce Q0 fixture
observations. Physical host and whole-product qualification remain separate
gates.

Run `pytest -q tests/test_plan8_scenario_harness.py` against the exact source
commit. Future integrated journeys can supply named adapters that receive
the same verified repository root and use the existing manifest contract.
Higher evidence levels require the applicable independent qualification.
