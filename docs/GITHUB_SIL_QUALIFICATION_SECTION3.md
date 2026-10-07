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
2. proves `git rev-parse HEAD` equals that expected SHA and that tracked/index state is clean;
3. installs the project in Python 3.11.16 using the repository's current package authority;
4. loads the current Section-2 capability registry;
5. derives **all journeys whose capabilities are currently AVAILABLE**;
6. executes every integration-level test vector for every such journey without a shell;
7. records journeys blocked by explicit UNAVAILABLE capabilities rather than simulating them;
8. independently verifies the produced evidence and log binding; and
9. uploads the exact-head evidence/log artifact even when execution fails.

The runner is deliberately registry-driven. A future capability promoted to AVAILABLE must have
an integration vector under the Section-2 contract and therefore becomes part of the SIL plan
without hard-coding a Section-3 allow-list.

The current fixture policy is `DETERMINISTIC_SYNTHETIC`. It uses a deterministic project-owned
ModelSpec/InitSpec identity and fixed synthetic data bytes. This is qualification input only:
it is not corpus admission, tokenizer fitting, optimizer execution, learned weights, final-test
access, or a substitute for later physical/device qualification.

## 3.2 — evidence contract

`src/twelve_six/sil_qualification.py` emits a closed evidence envelope that binds:

- exact Git SHA;
- `pyproject.toml` package identity;
- canonical capability-registry identity;
- deterministic synthetic ModelSpec and InitSpec identities;
- synthetic data identity;
- scenario identity;
- complete available and explicitly unavailable journey sets;
- every executed journey/capability/vector and argv;
- per-execution return code, stdout/stderr hashes and duration;
- aggregate input/output/log identities;
- run timings and final PASS/FAIL verdict; and
- a fail-closed scientific-boundary receipt.

PASS is impossible without at least one AVAILABLE journey and at least one executed integration
vector, and every executed vector must return zero. Component/unit green alone is not SIL PASS.

The verifier reloads the exact-head package metadata, capability registry and SIL scenario,
recomputes package/registry/model/init/data/scenario identities, reconstructs the complete SIL
plan and input identity, requires every execution record to match that plan exactly, then
recomputes the evidence identity, output identity and log hash. It rejects exact-SHA mismatch,
authority resealing, a widened scientific boundary and FAIL evidence when PASS is required.

## Durable surfaces

- `configs/control/sil_scenario_v1.json`
- `src/twelve_six/sil_qualification.py`
- `tests/test_sil_qualification_section3.py`
- shared `.github/workflows/ci.yml`
- `coordination/SECTION_CLOSURE_REGISTRY.json`

## Truth boundary

Section 3 creates no data admission, tokenizer-fit, optimizer, training, learned-weight,
final-test, paid-compute, scale-promotion, server, Windows/NVDA physical, or release authority.
GitHub SIL is software-in-the-loop qualification only. Later Sections retain their own higher
physical and product gates.
