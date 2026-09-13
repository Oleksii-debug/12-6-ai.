from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Callable
from pathlib import Path

RESERVE_PATH = Path("configs/evaluation/eval647_code_selection_reserve_v1.json")

_EXPECTED_OBJECTS = {
    "eval647-code-tenacity-wait-v1": {
        "repository": "jd/tenacity",
        "revision": "a2af454834c6bb5a1e39d67334031cdaf0f475b5",
        "path": "tenacity/wait.py",
        "source_family": "github:jd/tenacity",
        "expected_git_blob_sha1": "18fb6ea7b610f71f17cff7ea25de63177856dfbe",
        "expected_raw_bytes": 10438,
        "license_id": "Apache-2.0",
    },
    "eval647-code-more-itertools-recipes-v1": {
        "repository": "more-itertools/more-itertools",
        "revision": "64be96ceb2a6e836f76f069f4a96d2394d59fd0c",
        "path": "more_itertools/recipes.py",
        "source_family": "github:more-itertools/more-itertools",
        "expected_git_blob_sha1": "b984d86f2341b9fb74801d9b173f5e0fd00632f3",
        "expected_raw_bytes": 45752,
        "license_id": "MIT",
    },
}

_LICENSE_MARKERS = {
    "Apache-2.0": b"Apache License\n                           Version 2.0",
    "MIT": b"Permission is hereby granted, free of charge",
}


class Eval647ReserveError(RuntimeError):
    """Fail-closed error for the EVAL-CODE-RESERVE-V1 lineage."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Eval647ReserveError(message)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def raw_url(repository: str, revision: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repository}/{revision}/{path}"


def _default_fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "12-6-ai-eval647/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def load_reservation(repo_root: Path) -> dict:
    value = json.loads((repo_root / RESERVE_PATH).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "EVAL647 reservation must be a JSON object")
    return value


def validate_reservation(reservation: dict, *, require_sealed: bool) -> None:
    _require(
        reservation.get("schema_version") == "12-6.eval647-code-selection-reserve.v1",
        "EVAL647 schema drift",
    )
    _require(reservation.get("worker_id") == "SWARM-1640", "EVAL647 worker drift")
    _require(reservation.get("reservation_authority") == "github-issue:#647", "authority drift")
    _require(reservation.get("purpose") == "selection-validation", "purpose drift")
    _require(reservation.get("execution_profile") == "LOCAL_FREE", "execution profile drift")
    truth = reservation.get("truth_boundary", {})
    _require(truth.get("model_training_eligible") is False, "training boundary widened")
    _require(truth.get("tokenizer_fit_eligible") is False, "tokenizer boundary widened")
    _require(truth.get("final_test_eligible") is False, "final-test boundary widened")
    _require(truth.get("future_training_prohibited") is True, "future-training ban missing")

    objects = reservation.get("objects")
    _require(isinstance(objects, list), "objects must be a list")
    _require(len(objects) == len(_EXPECTED_OBJECTS), "object count drift")
    seen: set[str] = set()
    for obj in objects:
        _require(isinstance(obj, dict), "object must be a JSON object")
        record_id = obj.get("record_id")
        _require(record_id in _EXPECTED_OBJECTS, f"unexpected reserved object: {record_id}")
        _require(record_id not in seen, f"duplicate reserved object: {record_id}")
        seen.add(record_id)
        for field, expected in _EXPECTED_OBJECTS[record_id].items():
            _require(obj.get(field) == expected, f"{record_id} {field} drift")
        raw_sha = obj.get("expected_raw_sha256")
        if require_sealed:
            _require(
                isinstance(raw_sha, str)
                and len(raw_sha) == 64
                and all(char in "0123456789abcdef" for char in raw_sha),
                f"{record_id} raw SHA-256 is not sealed",
            )
        else:
            _require(raw_sha is None or isinstance(raw_sha, str), f"{record_id} raw SHA type")


def _verify_license(obj: dict, fetch: Callable[[str], bytes]) -> dict:
    license_raw = fetch(raw_url(obj["repository"], obj["revision"], "LICENSE"))
    marker = _LICENSE_MARKERS[obj["license_id"]]
    _require(marker in license_raw, f"{obj['record_id']} license marker mismatch")
    return {
        "path": "LICENSE",
        "raw_bytes": len(license_raw),
        "raw_sha256": sha256_bytes(license_raw),
        "git_blob_sha1": git_blob_sha1(license_raw),
    }


def _membership_record(obj: dict, raw_sha256: str) -> dict:
    return {
        "component_worker_id": "SWARM-1640-EVAL647-CODE-SELECTION-V1",
        "content_sha256": raw_sha256,
        "final_reporting_eligible": False,
        "final_test_eligible": False,
        "future_training_prohibited": True,
        "hyperparameter_selection_eligible": True,
        "language": "python",
        "license_id": obj["license_id"],
        "modality": "code",
        "model_selection_eligible": True,
        "purpose": "selection-validation",
        "record_id": obj["record_id"],
        "reservation_authority_ref": "github-issue:#647",
        "rights_authority_ref": "github-issue:#647",
        "selection_eligible": True,
        "selection_stratum": "code",
        "source_family": obj["source_family"],
        "source_git_blob_sha1": obj["expected_git_blob_sha1"],
        "source_id": f"github:{obj['repository']}:{obj['path']}",
        "source_path": obj["path"],
        "source_version": f"git:{obj['revision']}",
        "tokenizer_fit_eligible": False,
        "tokenizer_selection_eligible": True,
        "training_eligible": False,
        "utf8_bytes": obj["expected_raw_bytes"],
    }


def materialize(
    repo_root: Path,
    output_dir: Path,
    *,
    require_sealed: bool = True,
    fetch: Callable[[str], bytes] = _default_fetch,
) -> dict:
    reservation = load_reservation(repo_root)
    validate_reservation(reservation, require_sealed=require_sealed)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = output_dir / "payload"
    payload_dir.mkdir(parents=True, exist_ok=True)

    evidence_objects = []
    membership = []
    content_hashes: set[str] = set()
    families: set[str] = set()
    for obj in sorted(reservation["objects"], key=lambda item: item["record_id"]):
        raw = fetch(raw_url(obj["repository"], obj["revision"], obj["path"]))
        _require(len(raw) == obj["expected_raw_bytes"], f"{obj['record_id']} byte-count mismatch")
        _require(
            git_blob_sha1(raw) == obj["expected_git_blob_sha1"],
            f"{obj['record_id']} Git-blob mismatch",
        )
        raw.decode("utf-8", errors="strict")
        raw_sha = sha256_bytes(raw)
        sealed_sha = obj.get("expected_raw_sha256")
        if sealed_sha is not None:
            _require(raw_sha == sealed_sha, f"{obj['record_id']} raw SHA-256 mismatch")
        _require(raw_sha not in content_hashes, "duplicate reserved content hash")
        _require(obj["source_family"] not in families, "duplicate reserved source family")
        content_hashes.add(raw_sha)
        families.add(obj["source_family"])

        license_evidence = _verify_license(obj, fetch)
        (payload_dir / f"{obj['record_id']}.py").write_bytes(raw)
        membership.append(_membership_record(obj, raw_sha))
        evidence_objects.append(
            {
                "record_id": obj["record_id"],
                "repository": obj["repository"],
                "revision": obj["revision"],
                "path": obj["path"],
                "source_family": obj["source_family"],
                "license_id": obj["license_id"],
                "raw_bytes": len(raw),
                "raw_sha256": raw_sha,
                "git_blob_sha1": git_blob_sha1(raw),
                "license": license_evidence,
            }
        )

    membership_raw = "".join(canonical_json(row) + "\n" for row in membership).encode("utf-8")
    (output_dir / "membership.jsonl").write_bytes(membership_raw)
    evidence = {
        "schema_version": "12-6.eval647-code-selection-materialization.v1",
        "execution_profile": "LOCAL_FREE",
        "purpose": "selection-validation",
        "reservation_authority": "github-issue:#647",
        "documents": len(membership),
        "membership_sha256": sha256_bytes(membership_raw),
        "objects": evidence_objects,
        "training_eligible": False,
        "tokenizer_fit_eligible": False,
        "final_test_eligible": False,
        "future_training_prohibited": True,
        "payload_text_in_evidence": False,
        "status": "PASS_SOURCE_AUTHENTICATION",
    }
    evidence_raw = (canonical_json(evidence) + "\n").encode("utf-8")
    (output_dir / "materialization-evidence.json").write_bytes(evidence_raw)
    return evidence
