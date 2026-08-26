#!/usr/bin/env python3
"""Materialize EVAL-647 reserved code objects without persisting source payloads.

The tool fetches each object from raw.githubusercontent.com at an immutable
commit, verifies byte length and Git blob SHA-1, computes raw SHA-256, and
writes metadata-only evidence. It never writes the fetched source bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/evaluation/eval_code_reserve_v1.json"
USER_AGENT = "12-6-ai-eval647-reservation/1.0"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324 - Git identity, not security


def _raw_url(record: dict[str, Any]) -> str:
    return (
        "https://raw.githubusercontent.com/"
        f"{record['repository']}/{record['revision']}/{record['path']}"
    )


def _fetch(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            _require(getattr(response, "status", 200) == 200, f"HTTP status drift for {url}")
            return response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"immutable source fetch failed for {url}: {exc}") from exc


def materialize(manifest: dict[str, Any], *, timeout: int = 30) -> dict[str, Any]:
    reservation = manifest["reservation"]
    _require(manifest["execution_class"] == "LOCAL_FREE", "execution class drift")
    _require(manifest["purpose"] == "selection_validation_only", "purpose drift")
    _require(reservation["final_test"] is False, "final-test boundary widened")
    _require(reservation["final_test_payload_access_allowed"] is False, "final-test payload access widened")
    _require(reservation["final_test_outcome_access_allowed"] is False, "final-test outcome access widened")
    _require(reservation["training_allowed"] is False, "training accidentally allowed")
    _require(reservation["tokenizer_fit_allowed"] is False, "tokenizer fitting accidentally allowed")
    _require(reservation["permanent_future_training_exclusion"] is True, "future exclusion missing")

    objects = manifest["objects"]
    _require(len(objects) >= reservation["minimum_independent_families"], "insufficient reserved objects")
    families = {row["source_family"] for row in objects}
    _require(len(families) >= reservation["minimum_independent_families"], "insufficient independent families")

    sealed: list[dict[str, Any]] = []
    for row in objects:
        _require(row["raw_sha256"] is None, "contract raw SHA-256 must remain unset until live materialization")
        _require(row["training_allowed"] is False, f"training allowed for {row['repository']}")
        _require(row["tokenizer_fit_allowed"] is False, f"tokenizer fit allowed for {row['repository']}")
        _require(row["permanent_future_training_exclusion"] is True, f"future exclusion missing for {row['repository']}")

        payload = _fetch(_raw_url(row), timeout)
        _require(len(payload) == row["expected_raw_bytes"], f"raw byte-size drift for {row['repository']}:{row['path']}")
        observed_blob = _git_blob_sha1(payload)
        _require(observed_blob == row["git_blob_sha1"], f"Git blob identity drift for {row['repository']}:{row['path']}")

        sealed.append(
            {
                "source_family": row["source_family"],
                "repository": row["repository"],
                "revision": row["revision"],
                "path": row["path"],
                "git_blob_sha1": observed_blob,
                "raw_sha256": hashlib.sha256(payload).hexdigest(),
                "raw_bytes": len(payload),
                "license_spdx": row["license_spdx"],
                "evaluation_use": "selection_validation",
                "training_allowed": False,
                "tokenizer_fit_allowed": False,
                "permanent_future_training_exclusion": True,
            }
        )

    identity_payload = {
        "reservation_effective_at_utc": reservation["effective_at_utc"],
        "objects": sealed,
    }
    canonical = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {
        "schema_version": "12-6.eval-code-reserve-v1.materialization-evidence.v1",
        "worker_id": manifest["worker_id"],
        "issue": manifest["issue"],
        "execution_class": "LOCAL_FREE",
        "purpose": "selection_validation_only",
        "reservation_effective_at_utc": reservation["effective_at_utc"],
        "reserved_object_count": len(sealed),
        "independent_family_count": len(families),
        "objects": sealed,
        "object_set_identity_sha256": hashlib.sha256(canonical).hexdigest(),
        "raw_payload_persisted_in_repository": False,
        "selection_validation_records_authorized": 0,
        "remaining_gates": [
            "PROJECT_HISTORY_TOKENIZER_EXPOSURE_ZERO_PROVEN",
            "PROJECT_HISTORY_TRAINING_EXPOSURE_ZERO_PROVEN",
            "GLOBAL_CORPUS_EXACT_AND_NEAR_OVERLAP_ZERO_PROVEN",
            "FUTURE_TRAINING_EXCLUSION_CONSUMED_BY_CORPUS_PIPELINE",
            "PURPOSE_SPECIFIC_EVALUATION_AUTHORITY_TERMINAL",
        ],
        "terminal_status": "EXACT_RAW_OBJECTS_SEALED_PENDING_PROJECT_OVERLAP_AUDIT",
        "truth_boundary": {
            "final_test_touched": False,
            "model_training_authorized": False,
            "optimizer_updates_authorized": 0,
            "paid_compute_used": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evidence = materialize(manifest, timeout=args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": evidence["terminal_status"],
                "reserved_object_count": evidence["reserved_object_count"],
                "independent_family_count": evidence["independent_family_count"],
                "object_set_identity_sha256": evidence["object_set_identity_sha256"],
                "selection_validation_records_authorized": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
