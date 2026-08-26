"""Fresh-process-safe entrypoint for DATA-110 Corpus V1 RC execution.

DATA-110 persists its self-hashed run manifest as JSON between phase1 and resume.
TrainerConfig contains Python tuples (notably AdamW betas), while JSON represents
those values as arrays.  Canonicalizing the already-hashed manifest through the
JSON data model keeps strict fresh-process equality without weakening any field,
identity, checkpoint, corpus, tokenizer, or source binding.
"""

from __future__ import annotations

import json
from typing import Any

from twelve_six import data110_release_candidate as candidate


_ORIGINAL_RUN_MANIFEST = candidate._run_manifest


def json_normalize(value: Any) -> Any:
    """Return the deterministic JSON data-model representation of ``value``."""
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def normalized_run_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Normalize representation only while preserving the incumbent identity hash."""
    value = _ORIGINAL_RUN_MANIFEST(*args, **kwargs)
    normalized = json_normalize(value)
    if not isinstance(normalized, dict):
        raise TypeError("DATA-110 run manifest must normalize to an object")
    if normalized.get("identity_sha256") != value.get("identity_sha256"):
        raise RuntimeError("JSON normalization changed DATA-110 run-manifest identity")
    return normalized


def install_fresh_process_manifest_normalization() -> None:
    """Install the representation-only recovery for this CLI process."""
    candidate._run_manifest = normalized_run_manifest


def main(argv: list[str] | None = None) -> int:
    install_fresh_process_manifest_normalization()
    return candidate.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
