"""Plan 2 Section 2: rights/provenance consumer of the existing DATA324 authority.

This is NOT a second rights adjudicator. A passing source-level rights check
does not authorize a corpus, tokenizer fit, evaluation or training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tools.plan2_source_inventory_v1 import build_inventory, verify_inventory
from tools.validate_data324_current_main_port import validate_snapshot

SCHEMA = "12-6.plan2-admissibility-receipt.v1"
SOURCE_ID = "uk.kubernetes.docs.what-is-kubernetes"
MANIFEST_PATH = "data/external/snapshots/data324-kubernetes-ua-v1/manifest.json"


class Plan2AdmissibilityError(ValueError):
    """An input cannot be admitted under the independently checked authority."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise Plan2AdmissibilityError(reason)


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(data: object) -> str:
    return _hash((json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode())


def _strict_relative(root: Path, relative: str) -> Path:
    _require(isinstance(relative, str) and relative and not relative.startswith("/"),
             "invalid relative path")
    _require("\\" not in relative and ".." not in Path(relative).parts, "path traversal")
    current = root.resolve()
    _require(not current.is_symlink(), "root symlink")
    for part in Path(relative).parts:
        _require(part not in ("", ".", ".."), "unsafe path")
        current = current / part
        _require(not current.is_symlink(), "symlink not authorized")
    _require(current.resolve().is_relative_to(root.resolve()), "outside root")
    _require(current.is_file(), "missing authority payload")
    return current


def attest_data324_source(root: Path, seed: Mapping[str, Any]) -> dict[str, Any]:
    """Read and revalidate incumbent rights, materialization and identity evidence."""
    _require(isinstance(seed, Mapping) and set(seed) == {"schema_version", "revision", "sources"},
             "seed fields changed")
    inv = verify_inventory(build_inventory(seed["revision"], seed["sources"]))
    _require(len(inv["sources"]) == 1, "unreviewed additional source in scope")
    row = next((s for s in inv["sources"] if s["source_id"] == SOURCE_ID), None)
    _require(row is not None and row["status"] == "candidate",
             "DATA324 must remain a source-level candidate")
    try:
        result = validate_snapshot(root)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        raise Plan2AdmissibilityError("incumbent DATA324 validator failed") from exc
    _require(result.get("decision") == "PASS_SOURCE_LEVEL_CURRENT_MAIN_PORT",
             "incumbent did not pass")
    manifest = json.loads(_strict_relative(root, MANIFEST_PATH).read_text(encoding="utf-8"))
    _require(type(manifest.get("objects")) is list and len(manifest["objects"]) == 1,
             "unexpected source membership")
    obj = manifest["objects"][0]
    _require(obj["source_id"] == row["source_id"], "source ID drift")
    _require(row["source_family"] == manifest["source_family"], "family drift")
    _require(row["provenance"]["upstream_revision"] == manifest["upstream_revision"],
             "upstream revision drift")
    _require(row["provenance"]["content_sha256"] == obj["normalized_sha256"],
             "normalized identity drift")
    _require(row["location"] == "repo://" + obj["normalized_snapshot_path"],
             "physical locator drift")
    raw = _strict_relative(root, obj["raw_snapshot_path"]).read_bytes()
    normalized = _strict_relative(root, obj["normalized_snapshot_path"]).read_bytes()
    _require(_hash(raw) == obj["raw_sha256"] and len(raw) == obj["raw_bytes"],
             "raw snapshot drift")
    _require(_hash(normalized) == obj["normalized_sha256"] and
             len(normalized) == obj["normalized_utf8_bytes"], "normalized snapshot drift")
    rights = manifest.get("rights", {})
    _require(manifest.get("license_id") == "CC-BY-4.0" and
             rights.get("model_training") == "ALLOWED_WITH_ATTRIBUTION_RETAINED_IN_PROVENANCE" and
             rights.get("evaluation") == "NOT_GRANTED_BY_DATA324" and
             manifest.get("evaluation") == "NOT_SEPARATELY_ADMITTED", "rights ambiguity")
    lic = manifest["license_evidence"]
    _require(_hash(_strict_relative(root, lic["materialized_path"]).read_bytes()) == lic["sha256"],
             "license evidence drift")
    _require(_strict_relative(root, manifest["attribution_path"]).stat().st_size > 0,
             "attribution evidence missing")
    receipt = {
        "schema_version": SCHEMA,
        "source_id": SOURCE_ID,
        "source_family": row["source_family"],
        "inventory_sha256": inv["inventory_sha256"],
        "authority": row["provenance"]["authority_path"],
        "rights_basis": {"license_id": manifest["license_id"],
                         "license_evidence_sha256": lic["sha256"],
                         "model_training": rights["model_training"],
                         "evaluation": rights["evaluation"]},
        "provenance": {"upstream_revision": manifest["upstream_revision"],
                       "source_path": obj["path"], "raw_sha256": obj["raw_sha256"],
                       "normalized_sha256": obj["normalized_sha256"]},
        "record": {"record_id": SOURCE_ID + ":0", "member_path": obj["normalized_snapshot_path"],
                   "payload_sha256": obj["normalized_sha256"],
                   "payload_bytes": obj["normalized_utf8_bytes"]},
        "source_level_rights_verified": True,
        "corpus_training_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


def verify_receipt(receipt: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Compare against an independently reacquired receipt; self-hash alone is never trust."""
    _require(isinstance(receipt, Mapping) and isinstance(expected, Mapping),
             "receipt must be a mapping")
    _require(receipt == expected and receipt.get("receipt_sha256") ==
             _digest({k: v for k, v in receipt.items() if k != "receipt_sha256"}),
             "receipt differs from independently verified evidence")


def require_training_materialization(receipt: Mapping[str, Any]) -> None:
    """Fail closed: rights alone cannot promote DATA324 into a training corpus."""
    _require(receipt.get("schema_version") == SCHEMA and
             receipt.get("source_id") == SOURCE_ID, "unknown provenance receipt")
    raise Plan2AdmissibilityError(
        "DATA324 remains a source-level candidate; separate corpus authority is absent"
    )
