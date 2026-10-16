from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
from twelve_six.langgraph_qualification import (  # noqa: E402
    UPSTREAM_COMMIT,
    UPSTREAM_LICENSE,
    UPSTREAM_LICENSE_BLOB,
    UPSTREAM_REPOSITORY,
    UPSTREAM_TAG,
    validate_task_state,
)


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def validate(manifest_path: Path, evidence_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["upstream"]["repository"] == UPSTREAM_REPOSITORY
    assert manifest["upstream"]["tag"] == UPSTREAM_TAG
    assert manifest["upstream"]["commit"] == UPSTREAM_COMMIT
    assert manifest["rights"]["software_license"] == UPSTREAM_LICENSE
    assert manifest["rights"]["license_blob_sha"] == UPSTREAM_LICENSE_BLOB
    assert manifest["canonical_base"]["changed"] is False
    assert manifest["canonical_base"]["foreign_pretrained_weights"] is False
    assert manifest["runtime"]["execution_status"] in {"COMPLETED", "NOT_EXECUTED"}
    if manifest["runtime"]["execution_status"] == "NOT_EXECUTED":
        assert manifest["runtime"]["benchmark_status"] == "NOT_EXECUTED"
        assert manifest["runtime"]["parity_status"] == "NOT_EXECUTED"
    assert evidence["evidence_self_sha256"] == digest({k: v for k, v in evidence.items() if k != "evidence_self_sha256"})
    validate_task_state(manifest["project_task_fixture"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    validate(args.manifest, args.evidence)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
