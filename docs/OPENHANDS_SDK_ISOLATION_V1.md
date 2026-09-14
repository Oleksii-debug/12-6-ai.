# OpenHands Software Agent SDK isolation qualification V1

## Verdict

`CANDIDATE_MECHANICS_ONLY`. This package qualifies a 12-6-owned isolation contract for a future optional OpenHands adapter. It does **not** execute OpenHands, prove backend parity, adopt a backend, authorize agent benchmark access, or alter canonical Base.

## Live project authority

The package is bound to `main` commit `5020afd671a3885c1b738c8b4eafe7525f630546` and `configs/research/open_source_reuse_registry_v2.json` Git blob `d80a60357c56eacac135f948b8a72556bb849e5a`. That registry identifies `OPENHANDS_AGENT_SDK` as `software_agent_runtime`, MIT, `P1_POSTBASE_ISOLATION_REFERENCE`, `canonical_base_dependency=false`, and includes it only in `P1_E_AGENT_RUNTIME`. The same registry requires agent frameworks to remain post-Base-only.

Parent authority is issue #720; D09 lane issue #10 permits isolated post-training/reasoning tooling but forbids contaminating early canonical Base with alignment/specialization behavior.

## Upstream identity and facts

Upstream repository: `OpenHands/software-agent-sdk`.

Observed stable release at qualification time: `v1.43.1`, Git commit `ddac55697c5d15cf8a34495b5ed6d46c86db092a`, published 2026-08-21. The exact tag ref resolves directly to that commit. Its `LICENSE` at the tag is MIT (Git blob `b0bd86a6335a564bd3766666bebbf014fd6a8013`).

The release README describes Python and REST APIs for software agents and explicitly says agents can use either the local machine as workspace or ephemeral workspaces through the Agent Server. That flexibility is why upstream availability is not sufficient 12-6 isolation evidence: an adopted 12-6 adapter must fail closed on host-workspace execution.

Primary upstream references:
- `https://github.com/OpenHands/software-agent-sdk/releases/tag/v1.43.1`
- `https://github.com/OpenHands/software-agent-sdk/tree/ddac55697c5d15cf8a34495b5ed6d46c86db092a`
- `https://github.com/OpenHands/software-agent-sdk/blob/v1.43.1/LICENSE`
- `https://github.com/OpenHands/software-agent-sdk/blob/v1.43.1/README.md`

## Project-owned isolation boundary

A future adapter must provide an ephemeral sandbox; explicit non-wildcard filesystem roots; explicit network host allowlist; no inherited host environment; brokered named secrets only; bounded wall time/processes/memory/CPU; bounded persistence; audit and provenance logs; and an allowed-tool set that is a subset of the project-declared Tool Registry surface.

`src/twelve_six/post_base/openhands_isolation.py` validates those requirements independently of the OpenHands package. The module intentionally has no OpenHands dependency and performs no agent action.

Promotion is fail closed:
- `DISCOVERED`: metadata only.
- `CANDIDATE`: bounded contract/mechanics may be inspected without executing the backend.
- `PARITY_PROVEN`: requires exact pinned backend execution plus isolation-parity evidence.
- `ADOPTED`: additionally requires verified rollback.

No upstream benchmark, README claim, package install, or successful import can promote a state.

## Threat model / negative requirements

The validator rejects host execution mode, `/` or wildcard filesystem scope, wildcard network scope, host-environment inheritance, wildcard/implicit secrets, missing secret broker, missing or excessive resource limits, unbounded persistence, missing audit/provenance logging, undeclared tools, fabricated `PARITY_PROVEN`/`ADOPTED` states, and tampered deterministic evidence.

This V1 does not claim container escape resistance, kernel-level sandbox strength, network egress enforcement, secret redaction correctness, backend determinism, or OpenHands performance. Those require execution against an exact backend/runtime in a separately authorized isolated test environment.

## Operator handoff

Before `PARITY_PROVEN`, a successor must execute the exact pinned backend in an isolated environment and prove host filesystem/environment denial, network and secret-broker enforcement, resource/deadline enforcement, project Tool Registry mapping, and deterministic audit/provenance envelopes. Capture exact SDK/server/container identities and dependency/notice surface.

Before `ADOPTED`, require all parity evidence terminal, a tested rollback path to a project-native/null backend, locked dependencies/notices, and an explicit residual-risk decision. OpenHands remains replaceable infrastructure; its agent policy, model choice, prompts, skills, marketplace content, benchmark data, or outputs do not become canonical Base behavior or training data by implication.
