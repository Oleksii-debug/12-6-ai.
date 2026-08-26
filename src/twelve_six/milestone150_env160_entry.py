"""ENV-160 execution shim for MILESTONE-150.

This is deliberately narrow: it JSON-normalizes identity-bearing payloads before
self-hashing/comparison and attaches a stable runtime fingerprint to experimental
reports. It does not change model, data, optimizer, evaluation, or checkpoint math.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import milestone150_learned_base_ladder as impl
from .environment_parity import environment_fingerprint


def json_normalize(value: Any) -> Any:
    """Round-trip through JSON so tuples/lists compare by persisted semantics."""
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _is_experimental_report(payload: dict[str, Any]) -> bool:
    schema = str(payload.get("schema", ""))
    return schema in {
        impl.RUN_SCHEMA,
        "12-6.learned-base-ladder-phase1.v1",
        "12-6.learned-base-ladder-scale-report.v1",
        impl.SCHEMA,
    }


def install(repo_root: Path, source_sha: str) -> None:
    """Install the minimal convergence patches in the already-imported M150 module."""
    fingerprint = environment_fingerprint(repo_root, source_sha=source_sha)
    original_machine = impl.m100._machine

    def normalized_self_hashed(
        payload: dict[str, Any], key: str = "identity_sha256"
    ) -> dict[str, Any]:
        value = json_normalize(payload)
        if _is_experimental_report(value):
            value["environment_fingerprint"] = fingerprint
        value[key] = impl.hash_json(value)
        return value

    def fingerprinted_machine(machine_source_sha: str, locks: dict[str, Any]) -> dict[str, Any]:
        value = original_machine(machine_source_sha, locks)
        value["environment_fingerprint"] = environment_fingerprint(
            repo_root, source_sha=machine_source_sha
        )
        return value

    impl._self_hashed = normalized_self_hashed
    impl.m100._machine = fingerprinted_machine


def main(argv: list[str] | None = None) -> int:
    parser = impl.argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-sha", required=True)
    known, _ = parser.parse_known_args(argv)
    install(known.repo_root.resolve(), known.source_sha)
    return impl.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
