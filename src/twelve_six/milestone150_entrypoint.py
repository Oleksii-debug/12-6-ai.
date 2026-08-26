"""Fresh-process-safe entrypoint for MILESTONE-150.

The core ladder manifest is hash-canonical through JSON, but TrainerConfig contains
Python tuples (notably AdamW betas).  JSON persistence turns those tuples into
lists.  A fresh process therefore must compare the same JSON data model rather
than Python container implementation details.

This shim deliberately does not weaken any manifest field or identity check.  It
normalizes the already self-hashed run manifest through a deterministic JSON
round trip before phase1/resume code sees it, preserving its identity SHA-256.
"""

from __future__ import annotations

import json
from typing import Any

from twelve_six import milestone150_learned_base_ladder as ladder

_ORIGINAL_RUN_MANIFEST = ladder._run_manifest


def json_normalize(value: Any) -> Any:
    """Return the canonical JSON data-model representation of ``value``."""
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def normalized_run_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Build the incumbent run manifest and normalize only its representation."""
    value = _ORIGINAL_RUN_MANIFEST(*args, **kwargs)
    normalized = json_normalize(value)
    if not isinstance(normalized, dict):
        raise TypeError("MILESTONE-150 run manifest must normalize to an object")
    if normalized.get("identity_sha256") != value.get("identity_sha256"):
        raise RuntimeError("JSON normalization changed run-manifest identity")
    return normalized


def install_fresh_process_manifest_normalization() -> None:
    """Install the representation-only compatibility repair for this process."""
    ladder._run_manifest = normalized_run_manifest


def main(argv: list[str] | None = None) -> int:
    install_fresh_process_manifest_normalization()
    return ladder.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
