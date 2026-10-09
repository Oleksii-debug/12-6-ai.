# Section 2 — Executable capability map and acceptance graph

## Authority

This delta prepares the current canonical:

- **Section 2 — Executable capability map і acceptance graph**
- plan document `16KotBqgSyf3A0FEWpN8Pnobgf2ZJ1ecT8MLgHoQXibY`
- observed plan revision
  `ANLCKQnXY8VKCJDDnoyswSWTUVvktY41ODOKs8vJB9_b2whb1GpVUU2nUEZxcqkQXTS7YYo7IcAlrRhEcbUHlYiysiUA00rP9QpL5qUJ0A`

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

The registry's predecessor evidence is rebound to the exact qualified Section-1 candidate
`0e1f301c5123b4e52c111cb94264cfd61b60bf4b` / tree
`4fd06e8836450e61ab47e39657c96c6b6f76792e`, whose CI run
`37615156740` is terminal SUCCESS. Section 1 merged as
`main@9ef945ff977dd4674a9b7cf4d6fd2efff6302eeb` with the same tree byte-for-byte.
The subsequent closure-ledger commit
`main@cb94ca7a0c2b9a453356db45ef4d228c44e0ee21` / tree
`17edf21d66709f6e8a7c217e138b33c0bf9a0217` changes control evidence only. The
repository-equivalence gate authenticates both the pinned current-main receipt and the live
`main` ref against the qualified predecessor by capability-bearing blobs; at this baseline all
237 capability-bearing `src/twelve_six/**/*.py`, `tools/`, workflow and `pyproject.toml`
surfaces remain equivalent.

The capability registry is paired with
`configs/control/product_source_surface_inventory_v1.json`. This reopened repair candidate
classifies 116 production Python surfaces as accepted-main plus exactly one
`modified_candidate`: `src/twelve_six/capability_map.py`. The modified surface remains
mapped to the UNAVAILABLE `executable-capability-map` capability until the repaired bytes are
integrated. The declared modified set must equal the exact set of changed existing source blobs,
preventing a repair from silently inheriting accepted-main authority. All 117 source surfaces
remain mapped into registered capability families, and every capability is reciprocally bound
to at least one user/operator journey.

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
they are serialized or traversed through acceptance/accessor paths. Canonical serialization and
identity hashing use sealed stored-state helpers rather than mutable public class methods.
Constructor-time validation is not trusted after object creation: low-level mutation of an exact
dataclass instance fails closed before stale test, evidence, journey or source-surface state can
produce a new accepted identity. Rebinding public `__post_init__`, serializer methods, the
dependency-cycle checker, or the public component-contract resolver cannot manufacture accepted
authority, and public APIs do not expose caller-supplied seal/validator override hooks.

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
- registry-identity drift under availability resealing;
- class-validator and serializer rebinding against stored-state authority;
- source-inventory validator/serializer rebinding;
- dependency-cycle-checker rebinding;
- public component-resolver rebinding;
- rejection of caller-supplied authority override hooks.

## Truth boundary

This Section-2 candidate consumes the already-DONE Section 0 and Section 1 truth without
reopening or widening it, and does not create learned weights, corpus admission, tokenizer-fit
authority, optimizer exposure, final-test access,
paid-compute authority, Windows/NVDA physical acceptance, server acceptance or release
authority.

Its only purpose is to make current capability truth executable and machine-readable so later
functionality must enter the graph as AVAILABLE with evidence or remain explicitly
UNAVAILABLE. All predecessors are now terminally closed. During this repair qualification,
`executable-capability-map` deliberately remains UNAVAILABLE because its authority-bearing
source differs from accepted main. Section 2 remains REOPENED/non-DONE until this exact
candidate receives fresh terminal exact-head CI, the repaired source is integrated onto
then-current `main`, and durable closure evidence is recorded. No earlier Section-2 green is
transferred across the reopen.
