# Section 2 — Executable capability map and acceptance graph

## Authority

This delta prepares the current canonical:

- **Section 2 — Executable capability map і acceptance graph**
- plan document `16KotBqgSyf3A0FEWpN8Pnobgf2ZJ1ecT8MLgHoQXibY`
- observed plan revision
  `AHj4eMRLaHccAytgXlAMJ8VWh7OI8IBzEvpQtlHf2LCzdZ-MS4Asxbyf4K6DsKkQqi43tx_lYVEkT5UEAut8l8X7EhXLj3Zuv65xmgfsdw`

This branch is prepared only under the emergency later-section ladder while Sections 0 and 1
remain the numerical frontier. It cannot become canonical DONE before either predecessor.

## 2.1 — machine-readable current capability registry

The checked-in registry is:

`configs/control/product_capabilities_v1.json`

Its validator is:

`src/twelve_six/capability_map.py`

Each capability binds:

- a versioned capability id and component contract;
- explicit dependencies;
- user/operator journeys;
- supported and unsupported environments;
- executable test vectors;
- evidence targets;
- either an integrated result or an explicit unavailable reason.

The registry is bound to accepted
`main@019944d5fe12334791f05f1232d13de4a12e37d3` and its terminal successful push
CI run `37248299503`. Candidate Section-0/1 behavior is not resealed as accepted main truth.

The first registry intentionally distinguishes mechanics from complete product claims.
Examples:

- ModelSpec identity, byte-tokenizer runtime, deterministic packing mechanics,
  checkpoint-integrity mechanics and the Python Windows-oriented operator CLI mechanics are
  AVAILABLE with concrete checked-in test commands and main-CI evidence.
- physical Windows 11 support is **not** inferred from Python CLI tests;
- the replaceable cognitive-core shell remains UNAVAILABLE until Section 0 is integrated;
- unified generation identity remains UNAVAILABLE until Section 1 is integrated;
- learned-20M Base remains UNAVAILABLE because canonical learned weights do not yet exist;
- the packaged Windows/NVDA whole-product journey remains UNAVAILABLE pending its later
  physical gates.

The registry is designed to grow when real functionality grows; adding a capability does not
require changing the schema.

## 2.2 — executable acceptance path and fail-closed availability

For an AVAILABLE capability, the schema requires all of the following:

1. a concrete component contract;
2. at least one supported environment;
3. at least one executable component test vector;
4. at least one executable integration test vector;
5. at least one evidence target;
6. a non-empty integrated result.

`CapabilityRegistry.acceptance_path()` returns that contract → tests → evidence →
integrated-result path.

For an UNAVAILABLE capability, the schema requires an explicit reason and forbids an
`integrated_result`. Its acceptance path exposes the blocker and `integrated_result=null`;
it does not surface candidate tests or planned evidence as if they were acceptance.

Dependencies are closed-world and cycle-checked. An AVAILABLE capability cannot depend on an
UNAVAILABLE capability. Journeys are bidirectionally bound to their capabilities and
`journey_available()` is true only when every required capability is AVAILABLE.

Permanent regressions in `tests/test_capability_map_section2.py` cover:

- exact accepted-main/CI binding;
- executable test paths that must exist in the checkout;
- required component+integration acceptance vectors;
- unavailable-result fail-closed behavior;
- mechanics-versus-physical-Windows separation;
- unavailable dependency rejection;
- dependency-cycle rejection;
- journey/back-binding integrity;
- unknown top-level schema rejection;
- registry-identity drift under availability resealing.

## Truth boundary

This Section-2 candidate does not make Section 0 or Section 1 DONE and does not create learned
weights, corpus admission, tokenizer-fit authority, optimizer exposure, final-test access,
paid-compute authority, Windows/NVDA physical acceptance, server acceptance or release
authority.

Its only purpose is to make current capability truth executable and machine-readable so later
functionality must enter the graph as AVAILABLE with evidence or remain explicitly
UNAVAILABLE.
