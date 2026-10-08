# Plan 1 Section 6 — Migration Contract Baseline v1 packaging

## Authority and zero duplication

The v1 package under src/twelve_six/contracts is a pure data/serialization facade. It imports the previously terminal-DONE architecture and artifact authorities directly from twelve_six.system_architecture and twelve_six.artifact_identity. It does not reimplement identity hashing, parent binding, canonical SystemPlane interfaces, training, evaluation, agents, inference, provider authorization, scheduling, checkpointing or Plan8 readiness.

The baseline manifest computes its digest on demand from the original architecture and runtime-shell identities and the original ArtifactKind enumeration. This is a packaging descriptor, not another registry or new source of truth. The accepted Migration Contract Baseline v1 exists irrespective of this package: Plans 2–8 are not blocked by S6 closure.

## Wire and lifecycle

The v1 serializer accepts only canonical artifact reference/manifest (and future existing canonical manifests), or data-only EvidenceRef, ErrorRecord and LifecycleObservation values. Every packet is bound to baseline hash and explicit version. Byte-for-byte canonical serialization is required. Strict decoding rejects unknown fields, duplicate JSON keys, invalid UTF-8, foreign versions, bool-as-version, mismatched baseline and ambiguous noncanonical encodings.

EvidenceRef always says admitted=false; errors say policy_authority=false; lifecycle observations say effect_authority=false. They cannot themselves grant evidence acceptance, decisions, scheduling, permission, runtime effects or cross-plan readiness.

## Evolution rule

Changes require a new integer schema version, explicit compatibility-test evidence SHA256, and a named sorted impact-plan list. Breaking field removal/type change additionally requires a migration-fixture SHA256 and a nonempty deprecation notice. Policy returns REVIEW_REQUIRED with auto_migrate_existing_artifacts=false and reopen_terminal_plans=false; no migration or other plan mutation is executed. Accepted v1 decoders never silently accept v2 fields/semantics.

## Tests

Focused: python -m pytest -q tests/test_plan1_section6_contract_package.py tests/test_artifact_identity_section1.py tests/test_system_architecture_section0.py

Static lint: python -m ruff check src/twelve_six/contracts tests/test_plan1_section6_contract_package.py

Test scenarios: v1 source class identity and baseline hash; deterministic roundtrip and restart; unknown version/kind; duplicate fields/UTF-8/noncanonical bytes; fake permission/evidence authority; breaking changes without migration/deprecation/impact; forged declarations.

S6 cannot be marked terminal DONE until exact candidate qualification, main merge and readback, plus durable Plan1 GitHub/Drive status update. No paid compute or production release claims.
