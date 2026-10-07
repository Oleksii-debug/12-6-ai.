# Section 2 — Executable capability map and acceptance graph

## Authority

This delta prepares the current canonical:

- **Section 2 — Executable capability map і acceptance graph**
- plan document `16KotBqgSyf3A0FEWpN8Pnobgf2ZJ1ecT8MLgHoQXibY`
- observed plan revision
  `AHj4eMRLaHccAytgXlAMJ8VWh7OI8IBzEvpQtlHf2LCzdZ-MS4Asxbyf4K6DsKkQqi43tx_lYVEkT5UEAut8l8X7EhXLj3Zuv65xmgfsdw`

Sections 0 and 1 are now terminally closed on accepted main. Section 2 is the numerical
PRIMARY and this candidate refreshes its evidence against that accepted predecessor state.

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

The checked-in registry still carries historical Section-1 predecessor evidence from candidate
`49218c0c581b73bcd0985646f48bf35300b1948c` / CI `37608406911`. That evidence was valid
for the earlier closure but is no longer current authority because Section 1 has been REOPENED.
This staged Section-2 repair is converged on exact Section-1 candidate
`005da1c86768af95737ea29c7583387d7c99797f` / tree
`1f7b7a118a3b262e02ea394d562ebd2afc5ad8ed`; its CI run `37610826590` is currently
QUEUED / NOT PASS. Section 2 must refresh the registry/inventory predecessor SHA, tree and terminal
CI binding only after that Section-1 candidate is terminally qualified and integrated. Historical
green evidence is not transferred to the repaired predecessor.

The capability registry is paired with `configs/control/product_source_surface_inventory_v1.json`. The inventory binds the exact accepted-main Git tree and classifies every production Python source surface. Candidate authority distinguishes new `stacked_candidate` paths from existing accepted-main paths explicitly changed as `modified_candidate`; changed existing source bytes must match that modified set exactly. This keeps repair qualification fail-closed after a Section's source has already landed on main instead of silently treating a modified accepted surface as unchanged. At the original candidate head the coverage was 116 accepted-main surfaces + 1 candidate overlay = 117/117 current Python surfaces. Those surfaces map into 19 registered capability families, and every capability is reciprocally bound to at least one user/operator journey.

Examples:

- ModelSpec identity, byte-tokenizer runtime, deterministic packing mechanics,
  checkpoint-integrity mechanics and the Python Windows-oriented operator CLI mechanics are
  AVAILABLE with concrete checked-in test commands and main-CI evidence.
- physical Windows 11 support is **not** inferred from Python CLI tests;
- the replaceable cognitive-core shell is AVAILABLE from terminally closed Section 0;
- unified generation identity is AVAILABLE from terminally closed Section 1;
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

Required test levels are evaluated from sealed canonical wire strings rather than Enum
`__eq__` / `__hash__` behavior. Runtime dunder rebinding therefore cannot make one test
level satisfy both the component and integration acceptance requirements.

`CapabilityRegistry.acceptance_path()` returns that contract → tests → evidence →
integrated-result path.

For an UNAVAILABLE capability, the schema requires an explicit reason and forbids an
`integrated_result`. Its acceptance path exposes the blocker and `integrated_result=null`;
it does not surface candidate tests or planned evidence as if they were acceptance.

Dependencies are closed-world and cycle-checked. An AVAILABLE capability cannot depend on an
UNAVAILABLE capability. Journeys are bidirectionally bound to their capabilities and
`journey_available()` is true only when every required capability is AVAILABLE.

Identity-bearing registry, capability, journey and source-surface objects are revalidated whenever
they are serialized or traversed through acceptance/accessor paths. Canonical registry/inventory
serialization and hashes use sealed stored-state helpers rather than dispatching mutable class-level
validators or serializers, and acceptance paths serialize test/evidence state through the same
stored-state authority. Constructor-time validation is not trusted after object creation: low-level
mutation of an exact dataclass instance fails closed before stale test, evidence, journey or
source-surface state can produce a new accepted identity.

Permanent regressions in `tests/test_capability_map_section2.py` cover:
- AVAILABLE component contracts are resolved by the registry constructor itself, and the resolved module/symbol must actually be owned by the `twelve_six` namespace; imported external objects cannot be re-exported through a `twelve_six.*` attribute path to bypass the executable-contract gate;
- source-surface paths must be canonical POSIX paths beneath `src/twelve_six/`, rejecting traversal and alternate separators before inventory identity is accepted;

- exact accepted-main/CI binding;
- executable test paths that must exist in the checkout and remain canonical POSIX paths strictly beneath `tests/` (no traversal or alternate-separator escape);
- required component+integration acceptance vectors;
- unavailable-result fail-closed behavior;
- mechanics-versus-physical-Windows separation;
- unavailable dependency rejection;
- dependency-cycle rejection;
- journey/back-binding integrity;
- unknown top-level schema rejection;
- registry-identity drift under availability resealing.

## Truth boundary

This Section-2 candidate consumes terminal Section 0 only. Section 1 is currently REOPENED,
so Section 2 is also REOPENED and cannot consume historical Section-1 evidence as current
predecessor authority. The staged repair does not create learned weights, corpus admission,
tokenizer-fit authority, optimizer exposure, final-test access, paid-compute authority,
Windows/NVDA physical acceptance, server acceptance or release authority.

Its only purpose is to make current capability truth executable and machine-readable so later
functionality must enter the graph as AVAILABLE with evidence or remain explicitly
UNAVAILABLE. Section 2 remains IN_PROGRESS until exact-head shared CI is terminal PASS and
its unfinished predecessors close; the static capability/source/journey coverage gap itself
has been closed on this candidate.
