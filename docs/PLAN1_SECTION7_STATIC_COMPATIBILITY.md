# Plan 1 Section 7 — static backend and environment compatibility v1

Authority is a small, versioned, machine-readable file:
`configs/compatibility/plan1_backend_matrix_v1.json`.
The parser is `src/twelve_six/backend_compatibility.py`; it reads the already
accepted Plan-1 S4 lock index through the incumbent `validate_lock_index`
function. It neither modifies nor reads the executable Plan-8 capability map.

Each entry records a locked platform profile, a backend, a feature, a static
status (`supported`, `experimental`, `unsupported`) and optionally an opaque
SHA-256 evidence reference. Such a reference is **not** independent evidence
verification. A `supported` static declaration describes a declared contract
combination, **not** actual installed-machine qualification. All lookup
results explicitly return `qualification="UNQUALIFIED"`; Plan 8 owns the
executable acceptance/evidence graph and alone decides actual readiness.

Only three exact CPython **3.11.16** lock profiles are declared:
Linux x86_64, Linux aarch64 and Windows x86_64. They have a static supported
dependency-environment contract; CPU fixture runtime remains experimental and
CUDA training explicitly unsupported. Other backend/profile/feature triples,
including ROCm, MPS and foreign versions, fail closed to
`unsupported / UNQUALIFIED`. Neither the static supported flag nor fixtures
prove installed runtime, GPU, cloud or model/training readiness.

Any matrix with an unrecognized schema, altered lock index, missing profile,
unknown status, duplicate/ambiguous JSON key, path substitution or tampered
frozen-entry state is rejected. No paid compute, real training, model-weight
import, alternate acceptance authority or Plan-10 final release is introduced.

Qualification commands:
`python -m pytest -q tests/test_plan1_section7_backend_compatibility.py
tests/test_dependency_lock.py`
and `python -m ruff check src/twelve_six/backend_compatibility.py
tests/test_plan1_section7_backend_compatibility.py`.
