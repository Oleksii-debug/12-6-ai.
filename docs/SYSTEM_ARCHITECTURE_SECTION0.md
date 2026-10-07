# Section 0 — Full-system architecture contract

## Authority

This document implements the current canonical Section Plan requirement:

- **Section 0 — Цільова архітектура 12-6 як повної AI-системи**
- plan document: `16KotBqgSyf3A0FEWpN8Pnobgf2ZJ1ecT8MLgHoQXibY`
- observed plan revision at implementation start:
  `AHj4eMRLaHccAytgXlAMJ8VWh7OI8IBzEvpQtlHf2LCzdZ-MS4Asxbyf4K6DsKkQqi43tx_lYVEkT5UEAut8l8X7EhXLj3Zuv65xmgfsdw`

This is **not** the historical `S0 ~10K` experiment described by
`docs/S0_EXECUTION_PLAN.md`. The old S0 remains historical factory evidence. The
current Section 0 is the product architecture contract introduced by the 96-Section plan.

## Acceptance mapping

### 0.1 — explicit product planes with stable typed boundaries

`src/twelve_six/system_architecture.py` defines exactly seven architectural planes:

1. Base Model
2. Post-Base Learning
3. Model Gateway
4. Persistent Cognition
5. Tools
6. Live Agent Plane
7. Evolution Plane

The canonical manifest is machine-checkable. It rejects missing/duplicate planes,
missing/duplicate required boundary contracts, self-edges, invalid contract versions and
boundaries whose endpoints are not part of the manifest.

The canonical boundary policy is sealed after module initialization, and the manifest
validator captures that sealed policy. Callers therefore cannot mutate or rebind the
module-level policy and make a different producer/consumer/interface tuple become canonical.

Closed architecture, shell, core-binding and replacement-evidence schemas also reject
behavioral subclasses at canonical boundaries, preventing validation-facing state from
diverging from overridable identity serialization.

Canonical objects are revalidated again whenever they cross an assembly, replacement,
serialization or evidence boundary. Constructor-time validation is therefore not trusted after
object creation: post-validation mutation of an exact dataclass instance (including mutation via
low-level `object.__setattr__`) fails closed before its stale state can be hashed or promoted.

System-plane wire identities are read from the immutable underlying `str` payload and
cross-checked against the Enum member's stored `_value_` without dispatching the mutable
`.value` descriptor. This prevents class-level descriptor rebinding from creating a
validation/serialization split while still failing closed on low-level Enum value mutation.

Authority-bearing gateway compatibility uses revalidated exact scalar contract signatures
directly rather than dispatchable dataclass equality or a mutable module helper. Rebinding
`InterfaceContract.__eq__` or a helper global therefore cannot make an incompatible core
gateway appear compatible with the persistent shell.

The canonical typed boundaries are:

- Base Model -> Model Gateway: `twelve_six.model_gateway.v1`
- Post-Base Learning -> Base Model: `twelve_six.descendant_model.v1`
- Model Gateway -> Persistent Cognition: `twelve_six.inference_exchange.v1`
- Persistent Cognition -> Tools: `twelve_six.tool_invocation.v1`
- Persistent Cognition -> Live Agent Plane: `twelve_six.cognition_state.v1`
- Evolution Plane -> Post-Base Learning: `twelve_six.training_candidate.v1`
- Evolution Plane -> Model Gateway: `twelve_six.model_promotion.v1`

The manifest defines architectural contracts only. It does **not** claim that every later
capability is already implemented or physically qualified. Those claims remain governed by
their own later Sections and evidence gates.

### 0.2 — replaceable cognitive core without shell rewrites

The replaceable `CognitiveCoreIdentity` binds the exact ModelSpec, InitSpec, checkpoint,
tokenizer and parameter count. The persistent runtime shell is deliberately separate and
binds stable contracts for:

- model gateway;
- memory;
- tools;
- voice;
- UI;
- orchestration.

Each surface role is fail-closed to its canonical contract name, so a caller cannot silently
reseal the memory/tools/voice/UI/orchestration fields to one another or to an unrelated
interface while retaining a superficially valid shell identity. Contract schema versions
remain independently versioned.

`replace_cognitive_core()` accepts a new checkpoint or scale only when its gateway contract
is exactly compatible with the shell. The replacement receipt binds the old and new core
identities and the unchanged shell/surface identities. It carries immutable snapshots of both
cores and the preserved shell, cross-checking each claimed core hash, the shell hash, and every
surface hash against those snapshots so a resealed receipt cannot rename either core or claim
unrelated surface identities. A gateway generation mismatch
fails closed before the assembly is changed.

The regression suite proves both important replacement classes:

- different checkpoint, same parameter scale;
- different model generation/parameter scale.

In both cases memory/tools/voice/UI/orchestration contract identities remain unchanged. A
candidate requiring a new gateway generation is rejected instead of silently rewriting the
shell.

## Evidence and truth boundary

Permanent Q0 evidence for this Section is
`tests/test_system_architecture_section0.py`, exercised by the repository-wide Ruff/pytest
CI on the exact candidate SHA. Section 0 may be recorded as DONE only after this delta is
integrated into accepted live project state and the exact-head evidence is terminal.

This architecture work grants **no** lawful-corpus, tokenizer-fit, optimized-target,
optimizer-step, learned-weight, final-test, paid-compute, scale-promotion, Windows/NVDA,
server or release authority. Existing scientific controls remain fail-closed.
