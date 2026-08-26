"""MILESTONE-150 recovery entrypoint with JSON-stable run-manifest identity.

The underlying ladder runtime remains authoritative. This entrypoint applies the
same JSON-native normalization already accepted by SCALE-141 so tuple-valued
TrainerConfig fields survive write/read boundaries without weakening identity
checks.
"""
from __future__ import annotations

import json
from typing import Any

from twelve_six.checkpoint import hash_json
from twelve_six import milestone150_learned_base_ladder as base

_BASE_RUN_MANIFEST = base._run_manifest


def _json_normalize(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _json_stable_run_manifest(*args, **kwargs) -> dict[str, Any]:
    value = _BASE_RUN_MANIFEST(*args, **kwargs)
    value.pop("identity_sha256", None)
    value = _json_normalize(value)
    value["identity_sha256"] = hash_json(value)
    return value


def _install() -> None:
    base._run_manifest = _json_stable_run_manifest


def main(argv: list[str] | None = None) -> int:
    _install()
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
