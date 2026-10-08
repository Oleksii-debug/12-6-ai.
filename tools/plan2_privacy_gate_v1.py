"""Plan-2 Section 5: fail-closed privacy gate over the Section-4 candidate.

The existing privacy_filter_v3 and G06 execution authority are the detectors and
verification authority. This adapter only binds their decisions to physical
Section-3/4 records, excludes every non-ALLOW result, and publishes text-free
removal/rebuild evidence. This does NOT admit any training or tokenizer material.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from twelve_six.data import privacy_execution_authority as g06
from twelve_six.data.privacy_filter_v3 import assert_hash_safe_evidence
from tools import plan2_normalization_evidence_v1 as norm
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-privacy-candidate.v1"
POLICY_PATH = "configs/data/plan2_privacy_policy_v1.json"
POLICY = {
    "schema_version": "12-6.plan2-privacy-policy.v1",
    "scanner_authority": "twelve_six.data.privacy_execution_authority",
    "decision": "ALLOW_ONLY",
    "non_allow_handling": "EXCLUDE_ENTIRE_RECORD",
    "tombstone_handling": "EXCLUDE_ENTIRE_RECORD",
    "evidence": "TEXT_FREE_SHA256_AND_DETECTOR_COUNTS",
    "release_authority": "SOURCE_CANDIDATE_ONLY",
}


class Plan2PrivacyError(ValueError):
    """Incompatible, corrupt or unsafe Section-5 corpus candidate."""


def _fail(condition: bool, reason: str) -> None:
    if not condition:
        raise Plan2PrivacyError(reason)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        _fail(key not in result, "duplicate JSON authority key")
        result[key] = value
    return result


def _policy(root: Path) -> str:
    path = root / POLICY_PATH
    _fail(not path.is_symlink() and path.is_file(), "privacy policy unavailable")
    try:
        actual = json.loads(path.read_text(encoding="utf-8"),
                            object_pairs_hook=_pairs)
    except (ValueError, UnicodeError) as exc:
        raise Plan2PrivacyError("privacy policy is unreadable") from exc
    _fail(actual == POLICY, "privacy policy drift")
    return _sha(_canonical(actual))


def _validate_manifest(manifest: Mapping[str, Any], payload: bytes) -> None:
    _fail(isinstance(manifest, Mapping) and
          manifest.get("schema_version") == norm.MANIFEST_SCHEMA,
          "normalization schema drift")
    core = dict(manifest)
    digest = core.pop("manifest_sha256", None)
    _fail(digest == _sha(_canonical(core)), "normalization manifest self-hash drift")
    _fail(manifest.get("normalized_sha256") == _sha(payload) and
          manifest.get("normalized_bytes") == len(payload),
          "normalized payload hash or size drift")
    _fail(manifest.get("classification", {}).get("language") == "uk" and
          manifest.get("classification", {}).get("modality") == "text",
          "unsupported language or modality")
    _fail(manifest.get("training_corpus_authorized") is False and
          manifest.get("tokenizer_fit_authorized") is False and
          manifest.get("evaluation_authorized") is False,
          "upstream candidate improperly promoted")


def inspect_normalized(
    manifest: Mapping[str, Any],
    payload: bytes,
    *,
    policy_sha256: str,
    tombstones: Iterable[str] = (),
) -> dict[str, Any]:
    """Verify every S4 record then invoke the incumbent G06 privacy engine."""
    _validate_manifest(manifest, payload)
    _fail(isinstance(policy_sha256, str) and len(policy_sha256) == 64 and
          all(char in "0123456789abcdef" for char in policy_sha256),
          "invalid policy identity")
    source_id = manifest.get("source_id")
    _fail(isinstance(source_id, str) and bool(source_id),
          "source identity missing")
    offsets = manifest.get("records")
    _fail(isinstance(offsets, list) and bool(offsets) and
          len(offsets) == manifest.get("record_count"),
          "incomplete record index")
    _fail(isinstance(payload, bytes), "normalized bytes required")
    rows: list[dict[str, str]] = []
    inventory: list[dict[str, Any]] = []
    start = 0
    for index, entry in enumerate(offsets):
        _fail(isinstance(entry, Mapping) and
              set(entry) == {"index", "start_byte", "end_byte", "sha256"},
              "record index schema drift")
        end = entry["end_byte"]
        _fail(type(entry["index"]) is int and entry["index"] == index and
              type(entry["start_byte"]) is int and entry["start_byte"] == start and
              type(end) is int and start < end <= len(payload),
              "record boundary gap, overlap or overflow")
        chunk = payload[start:end]
        _fail(_sha(chunk) == entry["sha256"], "record digest drift")
        try:
            text = chunk.decode("utf-8", "strict")
        except UnicodeError as exc:
            raise Plan2PrivacyError("record is not strict UTF-8") from exc
        _fail(bool(text.strip()), "empty normalized record")
        record_id = f"{source_id}:r{index:08d}"
        rows.append({"id": record_id, "text": text, "mode": "uk"})
        inventory.append({
            "record_id": record_id, "source_id": source_id,
            "family": "plan2-normalized-candidate", "modality": "uk",
            "payload_sha256": _sha(chunk), "payload_bytes": len(chunk),
        })
        start = end
    _fail(start == len(payload), "trailing unindexed bytes")
    removed = tuple(tombstones)
    _fail(all(isinstance(v, str) for v in removed) and
          len(removed) == len(set(removed)) and
          set(removed).issubset({r["id"] for r in rows}),
          "invalid or unknown removal tombstone")
    expected = g06.input_rows_sha256_from_text_free_inventory(inventory)
    authority = g06.build_privacy_execution_authority(
        rows, expected_input_rows_sha256=expected)
    g06.verify_privacy_execution_authority(
        authority, rows, expected_input_rows_sha256=expected,
        expected_execution_identity_sha256=authority["execution_identity_sha256"])
    execution = {row["record_id"]: row for row in authority["records"]}
    clean, rejected = [], []
    for row in inventory:
        record_id = row["record_id"]
        decision = execution[record_id]["action"]
        if decision == "ALLOW" and record_id not in removed:
            clean.append({"record_id": record_id, "sha256": row["payload_sha256"]})
        else:
            rejected.append({
                "record_id": record_id,
                "reason": "REMOVAL" if record_id in removed else decision,
            })
    core = {
        "schema_version": SCHEMA,
        "policy_sha256": policy_sha256,
        "normalization_manifest_sha256": manifest["manifest_sha256"],
        "source_id": source_id,
        "g06_input_root_sha256": expected,
        "g06_execution_identity_sha256": authority["execution_identity_sha256"],
        "g06_privacy_binding": authority["privacy_binding"],
        "counts": authority["counts"],
        "detector_counts": authority["detector_counts"],
        "clean_record_hashes": clean,
        "excluded_records": rejected,
        "tombstones": sorted(removed),
        "input_record_count": len(rows),
        "candidate_record_count": len(clean),
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
        "privacy_coverage_limit": "Only configured high-confidence detectors",
    }
    result = {**core, "manifest_sha256": _sha(_canonical(core))}
    assert_hash_safe_evidence(result)
    return result


def stage_privacy(
    root: Path, destination: Path, *, tombstones: Iterable[str] = ()
) -> dict[str, Any]:
    """Immutable, restartable, no-clobber candidate publication."""
    _fail(not any(p.is_symlink() for p in (destination, *destination.parents)),
          "symlink destination or parent")
    root = root.resolve()
    policy_hash = _policy(root)
    try:
        normalization = norm.stage_normalization(root, destination / "normalization")
        payload = _read_destination(
            destination / "normalization" / "cohort" / "normalized.utf8")
    except (OSError, ValueError) as exc:
        raise Plan2PrivacyError("upstream normalization/physical cohort invalid") from exc
    result = inspect_normalized(
        normalization, payload, policy_sha256=policy_hash,
        tombstones=tombstones)
    artifact = destination / "privacy-manifest.json"
    expected = _canonical(result)
    if artifact.exists() or artifact.is_symlink():
        _fail(_read_destination(artifact) == expected,
              "immutable privacy output changed; rebuild at new destination")
    else:
        _atomic_write(destination, artifact, expected)
    _fail(_read_destination(artifact) == expected, "privacy readback mismatch")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan-2 privacy candidate gate")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--remove-record", action="append", default=[])
    args = parser.parse_args()
    result = stage_privacy(args.root, args.out_dir,
                           tombstones=args.remove_record)
    print(json.dumps({
        "status": "PASS_CANDIDATE_ONLY",
        "manifest_sha256": result["manifest_sha256"],
        "candidate_record_count": result["candidate_record_count"],
        "excluded_record_count": len(result["excluded_records"]),
        "training_corpus_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
