"""Fresh-process-safe, launch-gated entrypoint for MILESTONE-150.

The core ladder manifest is hash-canonical through JSON, but TrainerConfig contains
Python tuples (notably AdamW betas). JSON persistence turns those tuples into
lists, so fresh processes normalize the already self-hashed manifest through the
JSON data model.

CI-165 additionally makes every training phase fail closed unless a cheap launch
gate envelope exists and is bound to the exact current Git SHA and scale config.
Prepare/finalize/verification commands remain non-training operations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from twelve_six import milestone150_learned_base_ladder as ladder
from twelve_six.launch_gate import require_launch_envelope_from_env


_ORIGINAL_RUN_MANIFEST = ladder._run_manifest
_LONG_TRAINING_COMMANDS = {"phase1", "resume"}


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


def _option_value(argv: list[str], option: str, default: str | None = None) -> str | None:
    try:
        index = argv.index(option)
    except ValueError:
        return default
    if index + 1 >= len(argv):
        return default
    return argv[index + 1]


def enforce_launch_gate(argv: list[str]) -> None:
    """Refuse M150 training when the CI-165 envelope is absent/stale/misbound."""
    if not argv or argv[0] not in _LONG_TRAINING_COMMANDS:
        return
    scale = _option_value(argv, "--scale")
    if scale not in ladder.SCALE_ORDER:
        return
    repo_root = Path(_option_value(argv, "--repo-root", ".") or ".").resolve()
    require_launch_envelope_from_env(
        repo_root,
        expected_binding={
            "workflow": "milestone150-learned-base-ladder-v1",
            "scale": scale,
        },
    )


def main(argv: list[str] | None = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    install_fresh_process_manifest_normalization()
    enforce_launch_gate(actual_argv)
    return ladder.main(actual_argv)


if __name__ == "__main__":
    raise SystemExit(main())
