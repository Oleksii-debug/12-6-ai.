from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CONFIG_REL = Path("configs/evaluation/eval291_en_selection_validation_v1.json")
SELECTION_REL = Path("data/evaluation/eval291/selection-validation/en.jsonl")
AUTHORITY_REL = Path("evidence/eval291/en-selection-validation-v1-authority.json")

_ALLOWED_FINAL_TEST_KEYS = {
    "authority",
    "base_sha",
    "source_authority_path",
    "source_authority_git_blob_sha1",
    "source_authority_identity_sha256",
    "preserved_final_test_seed_git_blob_sha1",
    "admitted_source_ids",
    "payload_read_for_construction",
    "outcomes_read_for_construction",
    "outcomes_allowed_in_config",
}


class AuthorityError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _canonical(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _pretty(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _read_config(repo_root: Path) -> tuple[dict[str, Any], bytes]:
    data = (repo_root / CONFIG_REL).read_bytes()
    config = json.loads(data.decode("utf-8"))
    if config["schema_version"] != "12-6.eval291-en-selection-validation-config.v1":
        raise AuthorityError("unexpected EVAL-291 config schema")
    return config, data


def _assert_final_test_firewall(config: dict[str, Any]) -> None:
    boundary = config["final_test_boundary"]
    if set(boundary) != _ALLOWED_FINAL_TEST_KEYS:
        raise AuthorityError("final-test boundary contains unapproved fields")
    if boundary["payload_read_for_construction"] is not False:
        raise AuthorityError("final-test payload must not be read for selection construction")
    if boundary["outcomes_read_for_construction"] is not False:
        raise AuthorityError("final-test outcomes must not be read for selection construction")
    if boundary["outcomes_allowed_in_config"] is not False:
        raise AuthorityError("final-test outcome values are forbidden from EVAL-291 config")

    forbidden_path_fragments = ("recover174_real_holdout_seed", "/final-test/", "\\final-test\\")
    serialized = json.dumps(config["sources"], sort_keys=True)
    if any(fragment in serialized for fragment in forbidden_path_fragments):
        raise AuthorityError("selection source inputs reference final-test payload paths")

    final_ids = set(boundary["admitted_source_ids"])
    selected_ids = {source["source_family"] for source in config["sources"]}
    if final_ids & selected_ids:
        raise AuthorityError("selection source identity overlaps final-test source authority")


def _assert_purpose_firewall(config: dict[str, Any]) -> None:
    policy = config["construction_policy"]
    required = {
        "external_real_only": True,
        "deterministic_offline_rebuild": True,
        "final_test_payload_is_builder_input": False,
        "final_test_outcomes_inspected": False,
        "final_test_outcomes_may_influence_selection_construction": False,
        "training_bytes_may_be_reused": False,
        "tokenizer_fit_eligible": False,
        "training_eligible": False,
        "final_test_eligible": False,
        "selection_eligible": True,
    }
    for key, expected in required.items():
        if policy.get(key) is not expected:
            raise AuthorityError(f"purpose firewall violation: {key}")

    if config["purpose"] != "selection_validation" or config["language"] != "en":
        raise AuthorityError("authority must remain English selection-validation only")


def _verify_rights(repo_root: Path, source: dict[str, Any]) -> dict[str, Any]:
    rights = source["rights"]
    if rights["decision"] != "APPROVED_FOR_SELECTION_VALIDATION_RESERVATION":
        raise AuthorityError("source lacks selection-validation reservation")

    license_bytes = (repo_root / rights["local_license_path"]).read_bytes()
    if len(license_bytes) != rights["local_license_bytes"] or _sha256(license_bytes) != rights["local_license_sha256"]:
        raise AuthorityError("local license evidence mismatch")

    result: dict[str, Any] = {
        "license_id": rights["license_id"],
        "upstream_license_path": rights["upstream_license_path"],
        "upstream_license_git_blob_sha1": rights["upstream_license_git_blob_sha1"],
        "local_license_path": rights["local_license_path"],
        "local_license_sha256": rights["local_license_sha256"],
        "decision": rights["decision"],
    }
    if "local_notice_path" in rights:
        notice = (repo_root / rights["local_notice_path"]).read_bytes()
        if len(notice) != rights["local_notice_bytes"] or _sha256(notice) != rights["local_notice_sha256"]:
            raise AuthorityError("NOTICE evidence mismatch")
        if _git_blob_sha1(notice) != rights["upstream_notice_git_blob_sha1"]:
            raise AuthorityError("NOTICE is not the exact pinned upstream object")
        result.update(
            {
                "upstream_notice_path": rights["upstream_notice_path"],
                "upstream_notice_git_blob_sha1": rights["upstream_notice_git_blob_sha1"],
                "local_notice_path": rights["local_notice_path"],
                "local_notice_sha256": rights["local_notice_sha256"],
            }
        )
    else:
        if _git_blob_sha1(license_bytes) != rights["upstream_license_git_blob_sha1"]:
            raise AuthorityError("license evidence is not the exact pinned upstream object")
    return result


def _load_source(repo_root: Path, source: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    reservation = source["project_reservation"]
    if reservation != {
        "selection_validation": True,
        "training": False,
        "tokenizer_fit": False,
        "final_test": False,
        "training_object_from_terminal_admission": reservation["training_object_from_terminal_admission"],
    }:
        raise AuthorityError("invalid project reservation")
    training_object = reservation["training_object_from_terminal_admission"]
    if source["path"] == training_object["path"] or source["git_blob_sha1"] == training_object["git_blob_sha1"]:
        raise AuthorityError("selection-validation object reuses DATA-227 training object")

    snapshot_path = Path(source["snapshot_path"])
    if not snapshot_path.as_posix().startswith("data/evaluation/eval291/source-snapshots/"):
        raise AuthorityError("selection snapshot escaped reserved evaluation namespace")
    raw = (repo_root / snapshot_path).read_bytes()
    if len(raw) != source["raw_bytes"]:
        raise AuthorityError("source byte count mismatch")
    if _sha256(raw) != source["raw_sha256"]:
        raise AuthorityError("source SHA-256 mismatch")
    if _git_blob_sha1(raw) != source["git_blob_sha1"]:
        raise AuthorityError("source is not the exact pinned upstream Git object")
    raw.decode("utf-8", errors="strict")
    rights = _verify_rights(repo_root, source)
    return raw, rights


def materialize(repo_root: Path) -> tuple[bytes, bytes]:
    config, config_bytes = _read_config(repo_root)
    _assert_final_test_firewall(config)
    _assert_purpose_firewall(config)

    rows: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    families: set[str] = set()
    content_hashes: set[str] = set()
    for source in sorted(config["sources"], key=lambda item: item["document_id"]):
        raw, rights = _load_source(repo_root, source)
        text = raw.decode("utf-8")
        if source["raw_sha256"] in content_hashes:
            raise AuthorityError("duplicate selection content")
        content_hashes.add(source["raw_sha256"])
        families.add(source["source_family"])
        rows.append(
            {
                "content_sha256": source["raw_sha256"],
                "document_id": source["document_id"],
                "language": "en",
                "purpose": "selection_validation",
                "source_family": source["source_family"],
                "source_git_blob_sha1": source["git_blob_sha1"],
                "source_path": source["path"],
                "source_revision": source["revision"],
                "text": text,
            }
        )
        source_manifest.append(
            {
                "document_id": source["document_id"],
                "source_family": source["source_family"],
                "repository_url": source["repository_url"],
                "revision": source["revision"],
                "path": source["path"],
                "git_blob_sha1": source["git_blob_sha1"],
                "raw_bytes": source["raw_bytes"],
                "raw_sha256": source["raw_sha256"],
                "snapshot_path": source["snapshot_path"],
                "rights": rights,
                "project_reservation": source["project_reservation"],
            }
        )

    if len(families) < 2:
        raise AuthorityError("selection authority requires at least two terminal-admitted source families")

    selection_bytes = b"".join(_canonical(row) for row in rows)
    unsigned = {
        "schema_version": "12-6.eval291-en-selection-validation-authority.v1",
        "worker_id": config["worker_id"],
        "status": "IMMUTABLE_EXTERNAL_REAL_EN_SELECTION_VALIDATION",
        "execution_profile": config["execution_profile"],
        "language": "en",
        "purpose": "selection_validation",
        "documents": len(rows),
        "source_families": sorted(families),
        "config": {"path": CONFIG_REL.as_posix(), "sha256": _sha256(config_bytes)},
        "selection_artifact": {
            "path": SELECTION_REL.as_posix(),
            "bytes": len(selection_bytes),
            "sha256": _sha256(selection_bytes),
            "canonical_jsonl": True,
        },
        "sources": source_manifest,
        "terminal_source_family_admission": config["terminal_source_family_admission"],
        "final_test_boundary": config["final_test_boundary"],
        "firewalls": {
            "selection_bytes_are_training_eligible": False,
            "selection_bytes_are_tokenizer_fit_eligible": False,
            "selection_bytes_are_final_test_eligible": False,
            "final_test_payload_used_as_builder_input": False,
            "final_test_outcomes_inspected": False,
            "final_test_outcomes_influence_selection_construction": False,
            "data227_training_objects_reused": False,
            "offline_rebuild_requires_network": False,
        },
    }
    identity = _sha256(_canonical(unsigned))
    authority = {**unsigned, "authority_identity_sha256": identity}
    return selection_bytes, _pretty(authority)


def write(repo_root: Path) -> None:
    selection_bytes, authority_bytes = materialize(repo_root)
    (repo_root / SELECTION_REL).parent.mkdir(parents=True, exist_ok=True)
    (repo_root / AUTHORITY_REL).parent.mkdir(parents=True, exist_ok=True)
    (repo_root / SELECTION_REL).write_bytes(selection_bytes)
    (repo_root / AUTHORITY_REL).write_bytes(authority_bytes)


def verify(repo_root: Path) -> None:
    expected_selection, expected_authority = materialize(repo_root)
    actual_selection = (repo_root / SELECTION_REL).read_bytes()
    actual_authority = (repo_root / AUTHORITY_REL).read_bytes()
    if actual_selection != expected_selection:
        raise AuthorityError("committed selection JSONL is not a deterministic rebuild")
    if actual_authority != expected_authority:
        raise AuthorityError("committed authority manifest is not a deterministic rebuild")
    parsed = json.loads(actual_authority.decode("utf-8"))
    identity = parsed.pop("authority_identity_sha256")
    if _sha256(_canonical(parsed)) != identity:
        raise AuthorityError("authority self-identity mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "build":
        write(args.repo_root)
    else:
        verify(args.repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
