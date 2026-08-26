from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from twelve_six.postbase_evidence_firewall import (  # noqa: E402
    NamespaceViolation,
    validate_artifact_dict,
    validate_audit_manifest,
)

DEFAULT_MANIFEST = (
    ROOT / "configs/post_base/next100_092_evidence_namespace_audit_v1.json"
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_artifacts(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        artifacts: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise NamespaceViolation(
                    f"{path}:{line_number}: JSONL artifact must be an object"
                )
            artifacts.append(value)
        return artifacts

    value = _load_json(path)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return list(value)
    raise NamespaceViolation(
        f"{path}: artifact input must be an object, object list, or JSONL"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail closed on Base/post-Base evidence namespace crossing."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="NEXT100-092 audit manifest",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        action="append",
        default=[],
        help="Optional EvidenceEnvelope JSON/JSONL to validate; repeatable",
    )
    args = parser.parse_args()

    manifest = _load_json(args.manifest)
    if not isinstance(manifest, dict):
        raise NamespaceViolation("audit manifest must be a JSON object")
    validate_audit_manifest(manifest)

    validated = 0
    for artifact_path in args.artifact:
        for artifact in _iter_artifacts(artifact_path):
            validate_artifact_dict(artifact)
            validated += 1

    print(
        "NEXT100-092 EVIDENCE NAMESPACE GATE: PASS "
        f"components=10 artifacts={validated} profile=LOCAL_FREE"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
