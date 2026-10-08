# Plan 5 / Section 10 — Runtime security, privacy and trust boundaries

Reference component only; not a security certification of a deployed Nika agent.

## Reused authorities and invariant
- `ContextEntry` and `build_context` are the only context assembly authority (S1).
- `ToolRegistry` is the only versioned tool descriptor/permission/grant boundary (S7).
- `TaskStore` is the only durable task, effect and receipt authority (S2).
- `MediaArtifact` / `MediaPolicy` remains the S9 privacy and provenance preflight.
- This S10 layer adds fail-closed host-attested trust admission before those existing authorities, not an executor or secondary task registry.

## Trust and provenance
- `SourceEvidence` binds exact source ID, origin classification and SHA-256 of content.
- `admit_context_entry` calls independent trusted-host source/provenance and secret-safety verifiers. Privileged owner/system content requires an independent host proof; model/tool/external text cannot claim those roles or protected invariants even when the text says "system", "admin" or "ignore owner".
- The model cannot self-certify authority; caller-controlled booleans, labels and text do not substitute for independent host proof.
- Untrusted content may be kept as bounded source-aware data in S1 context, but cannot be promoted to instruction/critical authority.

## Credentials, grants, effects
- `SecretHandle` accepts scoped opaque `vault:<id>` handles only. The implementation never accepts/returns or logs secret bytes; trusted host owns resolution and applies scope checks. Never place handles in model-visible tool requests.
- `EffectGrant` binds issuer, independently host-verified evidence, task/effect/tool, exact descriptor digest, exact canonical request SHA-256, least-privilege permissions, task control epoch and bounded expiration.
- `prepare_secured_tool` requires fresh trusted-host grant, payload safety and permission callbacks; rejects secret-bearing request keys, vault references, Bearer values, forged/expired/mismatched grants and all untrusted issuers before delegating to canonical S7 `ToolRegistry.prepare`.
- Actual side effects are still issued by S2 and require S7 independently verified result receipts. Crash/restart unknown effects are never blindly retried.
- `AuditReceipt` is an allowlisted metadata-only audit projection (event + safe IDs + evidence digest + outcome), never arbitrary messages/media, raw request, credential, untrusted content or secret.

## Acceptance evidence
Run LOCAL_FREE:
`PYTHONPATH=src pytest -q tests/test_agent_security_section10.py tests/test_agent_multimodal_section9.py tests/test_agent_tools_section7.py tests/test_agent_task_state_section2.py tests/test_agent_task_state_section2_adversarial.py tests/test_agent_context_section1.py`
plus `compileall`, scoped Ruff, SHA-1 blob parity and accepted-main readback. Adversarial evidence: privilege spoofing, prompt injection, forged source digest and grants, scoped vault denial, secret exfiltration fields, host refusal, clock expiry, stale epoch/revision, unknown effect after cold restart, metadata-only logs.

Tests are model/provider-independent fixtures and do not constitute Windows/NVDA, physical-device, real credential store, network isolation, sandbox, external tool/provider, Nika product or whole-product acceptance. No paid compute or real external effects.
