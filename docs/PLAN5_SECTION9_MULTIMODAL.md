# Plan 5 / Section 9 — Multimodal and voice reference contract

Scope: 12-6 agent-facing typed reference adapter only. It does not perform actual media I/O, ASR, TTS, vision inference, model calls or Nika runtime actions.

- `MediaArtifact` v1 covers image/audio/ASR/TTS/vision input and output, bounded MIME/type/size, source and content SHA-256, source ID, host evidence, timestamp, privacy and origin locality.
- `MediaTransform` chain binds every transformation to previous output SHA-256 and monotonic timestamps, preserving traceable source/transform evidence. Chain length and metadata size are bounded; malformed/missing identity fails closed.
- `MediaPolicy` checks kind, direction, privacy, size and remote egress before any model transfer. Exact declared model capability is mandatory; no implicit permission is derived from discovery. Trusted callbacks independently verify actual content bytes, provenance and per-target permission.
- `MediaAdmission` is a reference intent/receipt, not a grant. A separate trusted host admission check is mandatory when building a tool call to prevent model-created/forged admission.
- `media.transfer` is a versioned closed-schema external-effect descriptor registered only through existing Section-7 `ToolRegistry`; `prepare_media_transfer` requires already reserved Section-2 `TaskStore` effect, epoch/revision, host grant and admission proof. Before any real effect, the host uses incumbent `issue_effect`; completion requires incumbent independent `make_result` / `accept_result` receipts. Unknown effects after crash/restart remain unknown and never blindly retry.
- LOCAL_FREE test fixtures intentionally use host callbacks; they are not evidence that media bytes were decoded, actual providers were contacted or real devices tested.
- No credentials, user media data, paid compute, cross-plan mutation or production champion use.

Qualification: `PYTHONPATH=src python -m pytest -q tests/test_agent_multimodal_section9.py tests/test_agent_tools_section7.py tests/test_agent_task_state_section2.py`, `compileall`, Ruff, exact-head readback and accepted-main readback. Never represent queued CI as PASS.
