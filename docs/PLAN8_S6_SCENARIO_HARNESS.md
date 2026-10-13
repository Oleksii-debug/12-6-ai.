# Plan 8 Section 6 — versioned scenario harness

The three manifests under `configs/plan8/scenarios/v1/` pin deterministic
component fixtures for the reserved evaluation firewall, an agent goal and
observation, and a SIL synthetic journey. Each declares a capability,
prerequisite, step adapter, expected observation digest and required Q2 level.

`parse_scenario_bytes` rejects ambiguous or oversized manifests.
`fixture_journey_adapters(repo_root)` reads the existing repository fixtures,
checks their schemas and the reserved evaluation no-training boundary, and
returns exact content fingerprints. `run_scenario` refuses missing prerequisites,
adapters, changed observations and adapter failures. `coverage_map` verifies
receipts and lists the Q2 gaps. A Q0 fixture receipt is never a physical host,
integrated product, learned model or final-test PASS.

Run `pytest -q tests/test_plan8_scenario_harness.py` against the exact source
commit. Future integrated journeys can supply adapters with the same manifest
and receipt contract, then qualify their higher evidence levels separately.
