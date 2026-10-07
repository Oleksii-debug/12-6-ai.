# Section 3 — GitHub Software-in-the-Loop Live Qualification Plane

## Authority and sequential position

This candidate implements the current canonical **Section 3** while the numerical closure
frontier remains Section 0. It is later-section fallback work only and cannot become canonical
DONE or merge ahead of Sections 0, 1, or 2.

The project CI policy permits no dedicated workflow. SIL therefore runs as
`sil-current-capability-journeys` inside the single shared
`.github/workflows/ci.yml`.

## 3.1 — reproducible exact-head clean runner

The SIL job:

1. checks out the exact pull-request head SHA or exact push SHA with credentials disabled;
2. proves `git rev-parse HEAD` equals that expected SHA and that tracked, index, and
   non-ignored untracked state is clean;
3. fetches immutable historical environment authority
   `029514654829cebc149cff6fc1fea2a8ba4fa566` from `refs/pull/402/head`, requires
   `FETCH_HEAD` to equal that exact commit, extracts the accepted toolchain/CPU-runtime/dev
   lock bytes, and verifies their pinned SHA-256 identities;
4. creates a fresh CPython 3.11.16 virtual environment, installs only those exact
   `--require-hashes --no-deps` locks, then installs the exact candidate editable package
   with `--no-deps --no-build-isolation` so no dependency resolver can silently select newer
   package bytes;
5. validates the exact final distribution/version set and emits a canonical environment receipt
   binding Python version, authority commit, lock identities, installed versions, and receipt
   identity;
6. loads the current Section-2 capability registry and derives **all journeys whose
   capabilities are currently AVAILABLE**;
7. requires every AVAILABLE journey to match a sealed Section-3 end-to-end contract naming the
   exact ordered integration steps that jointly constitute that journey under one shared SIL
   input envelope;
8. executes every declared journey step without a shell, in declared journey order;
9. records journeys blocked by explicit UNAVAILABLE capabilities rather than simulating them;
10. independently verifies the produced environment/evidence/log bindings; and
11. uploads the exact-head environment receipt, evidence, and log artifact even when execution
    fails.

The registry remains the authority for which journeys are AVAILABLE, while Section 3 explicitly
owns the end-to-end interpretation of those journeys. This is intentionally fail-closed: adding,
removing, or reordering an AVAILABLE journey or its integration vectors does **not** silently
inherit SIL end-to-end status. The sealed journey contract must be updated deliberately and must
match the registry exactly. Rebinding the module-global policy after import cannot reseal the
validator because the canonical policy object is captured by the plan builder.

The current fixture policy is `DETERMINISTIC_SYNTHETIC`. It uses a deterministic project-owned
ModelSpec/InitSpec identity and fixed synthetic data bytes. This is qualification input only:
it is not corpus admission, tokenizer fitting, optimizer execution, learned weights, final-test
access, or a substitute for later physical/device qualification.

## 3.2 — evidence contract

`src/twelve_six/sil_qualification.py` emits a closed evidence envelope that binds:

- exact Git SHA;
- tracked package-source manifest identity over `pyproject.toml`, `src/twelve_six`, and
  packaged `configs/research` files;
- deterministic environment-receipt identity over CPython 3.11.16, immutable historical
  authority commit, exact accepted lock SHA-256 values, and exact installed distribution
  versions;
- canonical capability-registry identity;
- deterministic synthetic ModelSpec and InitSpec identities;
- synthetic data identity;
- scenario identity;
- complete available and explicitly unavailable journey sets;
- the sealed end-to-end contract for every AVAILABLE journey, including ordered vector IDs,
  `SEQUENTIAL_SHARED_INPUT_ENVELOPE` execution mode, and
  `ALL_DECLARED_STEPS_PASS_IN_ORDER` completion rule;
- every executed journey/capability/vector and argv;
- per-execution return code, stdout/stderr hashes and duration;
- a strict JSONL execution log whose raw stdout/stderr and execution metadata are
  cross-checked record-by-record against the evidence envelope;
- aggregate input/output/log identities;
- run timings and final PASS/FAIL verdict; and
- a fail-closed scientific-boundary receipt.

PASS is impossible without at least one AVAILABLE journey and at least one executed integration
vector, and every executed vector must return zero. Component/unit green alone is not SIL PASS.

Before and after every integration vector, the runner re-probes the exact Git SHA and complete
non-ignored checkout cleanliness, including untracked files. A vector that mutates the checkout
or advances HEAD invalidates the run immediately, so later vectors cannot silently execute a
different tree under the original SHA receipt.

The verifier rebuilds the exact tracked package-source manifest, loads and validates the
canonical environment receipt against the accepted historical lock authority and current
interpreter/distribution set, reloads the capability registry and SIL scenario, recomputes
package/environment/registry/model/init/data/scenario identities, reconstructs the complete SIL
plan (including all end-to-end contracts) and input identity, requires every execution record
to match that plan exactly, parses
the strict JSONL log and cross-checks its return code/input binding/stdout/stderr/duration
against each evidence execution, then recomputes the evidence identity, output identity and
log hash. It rejects exact-SHA mismatch,
authority resealing, a widened scientific boundary and FAIL evidence when PASS is required.

## Durable surfaces

- `configs/control/sil_scenario_v1.json`
- `src/twelve_six/sil_qualification.py`
- `tests/test_sil_qualification_section3.py`
- shared `.github/workflows/ci.yml`
- immutable environment authority commit `029514654829cebc149cff6fc1fea2a8ba4fa566`
  (toolchain, CPU-runtime, and dev hash locks are reconstructed from that Git object at run time)
- `SEQUENTIAL_CLOSURE_STATE.md`

## Truth boundary

Section 3 creates no data admission, tokenizer-fit, optimizer, training, learned-weight,
final-test, paid-compute, scale-promotion, server, Windows/NVDA physical, or release authority.
GitHub SIL is software-in-the-loop qualification only. Later Sections retain their own higher
physical and product gates.

The verifier treats timing evidence as a closed schema: start/finish nanosecond timestamps and duration must be non-negative integers, and wall-clock finish cannot precede start. A recomputed self-hash cannot reseal impossible timing metadata into a PASS artifact.
