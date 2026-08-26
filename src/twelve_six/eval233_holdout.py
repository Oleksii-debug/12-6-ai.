"""EVAL-233 immutable real-source holdout v2 materializer.

RECOVER-174 is already reserved as final-test by DATA-232. Therefore EVAL-233
must not manufacture selection-validation data by splitting that authority.
The exact RECOVER-174 compressed seed is copied byte-for-byte into final-test;
selection-validation remains a distinct immutable but empty blocked set until
an independent selection authority exists. Code and decontamination also fail
closed at the observed DATA-227/DATA-230/DATA-232 boundary.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

WORKER_ID = "EVAL-233-REAL-HOLDOUT-V2"
SCHEMA = "12-6.eval233-real-holdout-v2.v2"
SET_SCHEMA = "12-6.eval233-real-holdout-set.v2"
RECOVER174_SEED_PATH = Path("data/evaluation/recover174_real_holdout_seed.jsonl.gz")
RECOVER174_AUTHORITY_PATH = Path("configs/evaluation/recover174_source_authority_v1.json")
RECOVER174_SEED_GIT_BLOB_SHA1 = "4bfbfbf29fa9538cabda6068efd3a1fd036a9479"
RECOVER174_AUTHORITY_GIT_BLOB_SHA1 = "3ba9f221a82468f971c17eda518cd6f1642fd311"
RECOVER174_AUTHORITY_ID = "c7211b3e1e6a4f22463d0e6174f0d6162c2452585704efad5564a35de8de609f"
NORMALIZATION_POLICY = "PRESERVE_RECOVER174_BYTES_NO_RENORMALIZATION"

DATA227_BRANCH = "data227/real-code-source-admission-v2-20260826"
DATA227_HEAD = "8ebdb2e132ed7bae5245e9d4c140752640ab9885"
DATA227_RIGHTS_POLICY_BLOB = "0ce5223a1cade10031899bf27348a1a65121d4c6"

DATA230_BRANCH = "data230/corpus-v03-external-real-20260826"
DATA230_OBSERVED_HEAD = "6d994e2aece6c44e28c1a2c344ac98b5a8fd5e08"
DATA230_OBSERVED_MESSAGE = "DATA-214 restore retained quality and privacy evidence"

DATA232_HEAD = "42eba0ae7a5ca903f2e03947d83abe8410e7cd80"
DATA232_BLOCKER_BLOB = "a125af8bf17cb0ef17ba6cd49618c610498e9dd0"
DATA232_CONFIG_BLOB = "993338ad73441ac4019f766bab21f763b1dd7947"
DATA232_RESERVED_BLOB = "4cb63faef0dda5037d706b3e51c4cdbff577137c"
DATA232_REPORT_ID = "1dc6e25ad795c790919e71e36a3a32bdf73736903452ea1c4234d2c208207b20"
DATA232_FINAL_TEST_ID = "86d51eb106524cd8e4d0f94d4ff6e2e3426c6321e0698279877dfc4d5fce3116"
DATA232_RESERVED_ID = "d169a9ca3bd561729227a222cdea47b8296b753bee0cd7f0ffea02621ac74b2a"


class Eval233Error(RuntimeError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def hash_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    # SHA-1 is required only to verify immutable Git object identity.
    return hashlib.sha1(header + value).hexdigest()  # noqa: S324


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Eval233Error(f"unable to read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise Eval233Error(f"expected JSON object: {path}")
    return value


def _verify_upstream_exact(
    repo_root: Path,
    *,
    expected_seed_git_blob_sha1: str = RECOVER174_SEED_GIT_BLOB_SHA1,
    expected_authority_git_blob_sha1: str = RECOVER174_AUTHORITY_GIT_BLOB_SHA1,
    expected_authority_identity: str = RECOVER174_AUTHORITY_ID,
) -> tuple[Path, dict[str, Any]]:
    seed_path = repo_root / RECOVER174_SEED_PATH
    authority_path = repo_root / RECOVER174_AUTHORITY_PATH
    if not seed_path.is_file() or not authority_path.is_file():
        raise Eval233Error("RECOVER-174 exact seed/authority inputs are missing")
    seed_blob = seed_path.read_bytes()
    authority_blob = authority_path.read_bytes()
    if git_blob_sha1(seed_blob) != expected_seed_git_blob_sha1:
        raise Eval233Error("RECOVER-174 seed Git blob identity changed")
    if git_blob_sha1(authority_blob) != expected_authority_git_blob_sha1:
        raise Eval233Error("RECOVER-174 authority Git blob identity changed")
    authority = _read_json(authority_path)
    if authority.get("authority_identity_sha256") != expected_authority_identity:
        raise Eval233Error("RECOVER-174 authority semantic identity changed")
    if authority.get("admitted_modalities") != ["ua", "en"]:
        raise Eval233Error("RECOVER-174 admitted modality contract changed")
    code = authority.get("blocked_modalities", {}).get("code", {})
    if code.get("status") != "BLOCKED_NO_EVALUATION_USE_AUTHORITY":
        raise Eval233Error("RECOVER-174 code blocker contract changed")
    return seed_path, authority


def _load_and_validate_rows(
    seed_path: Path, authority: dict[str, Any]
) -> list[tuple[bytes, dict[str, Any]]]:
    try:
        decompressed = gzip.decompress(seed_path.read_bytes())
    except (OSError, EOFError) as exc:
        raise Eval233Error("unable to decompress RECOVER-174 seed") from exc
    raw_lines = decompressed.splitlines(keepends=True)
    if not raw_lines:
        raise Eval233Error("RECOVER-174 seed is empty")
    rows: list[tuple[bytes, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    for raw in raw_lines:
        if not raw.endswith(b"\n"):
            raise Eval233Error("every RECOVER-174 seed row must end with newline")
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Eval233Error("invalid RECOVER-174 seed JSONL row") from exc
        if not isinstance(row, dict):
            raise Eval233Error("RECOVER-174 seed rows must be JSON objects")
        required = (
            "record_id",
            "modality",
            "source_id",
            "source_family",
            "source_version",
            "source_snapshot_sha256",
            "source_kind",
            "evaluation_use_authority_ref",
            "provenance_ref",
            "text",
        )
        missing = [field for field in required if field not in row]
        if missing:
            raise Eval233Error(f"RECOVER-174 record missing fields: {missing}")
        modality = str(row["modality"])
        if modality not in {"ua", "en"}:
            raise Eval233Error("baseline RECOVER-174 seed must remain UA/EN only")
        if row["source_kind"] != "EXTERNAL_REAL":
            raise Eval233Error("non-real source present in RECOVER-174 holdout")
        source_id = str(row["source_id"])
        source = authority.get("sources", {}).get(source_id)
        if not isinstance(source, dict):
            raise Eval233Error(f"source {source_id!r} missing from authority")
        if source.get("evaluation_status") != "APPROVED_FOR_HELDOUT_EVALUATION":
            raise Eval233Error(f"source {source_id!r} lacks heldout evaluation approval")
        if row["source_snapshot_sha256"] not in source.get(
            "admitted_source_snapshots_sha256", []
        ):
            raise Eval233Error(f"record snapshot for {source_id!r} not admitted")
        text = row["text"]
        if not isinstance(text, str) or not text:
            raise Eval233Error("heldout record text must be non-empty UTF-8 text")
        text_bytes = text.encode("utf-8")
        if "content_sha256" in row and row["content_sha256"] != sha256_bytes(text_bytes):
            raise Eval233Error("record content_sha256 mismatch")
        if "source_bytes" in row and row["source_bytes"] != len(text_bytes):
            raise Eval233Error("record source_bytes mismatch")
        record_id = str(row["record_id"])
        content_id = sha256_bytes(text_bytes)
        if record_id in seen_ids:
            raise Eval233Error("duplicate RECOVER-174 record_id")
        if content_id in seen_content:
            raise Eval233Error("duplicate exact content in RECOVER-174 seed")
        seen_ids.add(record_id)
        seen_content.add(content_id)
        rows.append((raw, row))
    return rows


def _record_binding(
    raw: bytes, row: dict[str, Any], authority: dict[str, Any]
) -> dict[str, Any]:
    source = authority["sources"][str(row["source_id"])]
    return {
        "record_id": str(row["record_id"]),
        "modality": str(row["modality"]),
        "source_id": str(row["source_id"]),
        "source_family": str(row["source_family"]),
        "source_version": str(row["source_version"]),
        "purpose": "final-test",
        "raw_source_sha256": list(source.get("raw_sha256", [])),
        "extracted_normalized_snapshot_sha256": str(row["source_snapshot_sha256"]),
        "upstream_source_identity_sha256": str(source.get("source_identity_sha256", "")),
        "evaluation_use_authority_ref": str(row["evaluation_use_authority_ref"]),
        "provenance_ref": str(row["provenance_ref"]),
        "normalization_policy": NORMALIZATION_POLICY,
        "content_sha256": sha256_bytes(str(row["text"]).encode("utf-8")),
        "source_jsonl_row_bytes_sha256": sha256_bytes(raw),
        "source_jsonl_row_bytes": len(raw),
    }


def _selection_manifest() -> dict[str, Any]:
    unsigned = {
        "schema_version": SET_SCHEMA,
        "worker_id": WORKER_ID,
        "purpose": "selection-validation",
        "status": "BLOCKED_NO_IMMUTABLE_SELECTION_VALIDATION_AUTHORITY",
        "modalities": [],
        "documents": 0,
        "files": {},
        "records": [],
        "selection_eligible": False,
        "tokenizer_fit_eligible": False,
        "hyperparameter_selection_eligible": False,
        "final_test_exposure_prohibited": False,
        "invented_from_final_test": False,
        "immutable": True,
    }
    manifest = dict(unsigned)
    manifest["set_identity_sha256"] = hash_json(unsigned)
    return manifest


def _final_manifest(
    seed_blob: bytes,
    rows: list[tuple[bytes, dict[str, Any]]],
    authority: dict[str, Any],
) -> dict[str, Any]:
    modality_counts = {"ua": 0, "en": 0}
    for _, row in rows:
        modality_counts[str(row["modality"])] += 1
    unsigned = {
        "schema_version": SET_SCHEMA,
        "worker_id": WORKER_ID,
        "purpose": "final-test",
        "status": "IMMUTABLE_RESERVED_FINAL_TEST",
        "modalities": ["ua", "en"],
        "documents": len(rows),
        "modality_documents": modality_counts,
        "files": {
            "recover174_seed": {
                "path": "final-test/recover174_real_holdout_seed.jsonl.gz",
                "bytes": len(seed_blob),
                "sha256": sha256_bytes(seed_blob),
                "git_blob_sha1": git_blob_sha1(seed_blob),
            }
        },
        "records": [_record_binding(raw, row, authority) for raw, row in rows],
        "selection_eligible": False,
        "tokenizer_fit_eligible": False,
        "hyperparameter_selection_eligible": False,
        "final_test_exposure_prohibited": True,
        "normalization_policy": NORMALIZATION_POLICY,
        "immutable": True,
    }
    manifest = dict(unsigned)
    manifest["set_identity_sha256"] = hash_json(unsigned)
    return manifest


def _root_manifest(
    source_sha: str,
    seed_blob: bytes,
    selection: dict[str, Any],
    final: dict[str, Any],
    authority: dict[str, Any],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": SCHEMA,
        "worker_id": WORKER_ID,
        "source_sha": source_sha,
        "status": "PARTIAL_FINAL_TEST_UA_EN_ONLY_SELECTION_CODE_DECONTAM_BLOCKED",
        "upstream": {
            "recover174_seed_path": str(RECOVER174_SEED_PATH),
            "recover174_seed_git_blob_sha1": RECOVER174_SEED_GIT_BLOB_SHA1,
            "recover174_seed_sha256": sha256_bytes(seed_blob),
            "recover174_authority_path": str(RECOVER174_AUTHORITY_PATH),
            "recover174_authority_git_blob_sha1": RECOVER174_AUTHORITY_GIT_BLOB_SHA1,
            "recover174_authority_identity_sha256": authority["authority_identity_sha256"],
            "recover174_reserved_role": "final_test",
        },
        "sets": {
            "selection-validation": {
                "set_identity_sha256": selection["set_identity_sha256"],
                "documents": selection["documents"],
                "status": selection["status"],
            },
            "final-test": {
                "set_identity_sha256": final["set_identity_sha256"],
                "documents": final["documents"],
                "status": final["status"],
            },
        },
        "code": {
            "documents": 0,
            "status": "BLOCKED_DATA227_TRAINING_ONLY_NO_EVALUATION_RESERVATION",
            "data227_branch": DATA227_BRANCH,
            "data227_head_sha": DATA227_HEAD,
            "data227_rights_policy_git_blob_sha1": DATA227_RIGHTS_POLICY_BLOB,
            "training_use_allowed": True,
            "evaluation_use_explicitly_authorized": False,
            "reserved_from_training": False,
            "training_rights_do_not_imply_evaluation_rights": True,
            "unauthorized_code_bytes_admitted": False,
        },
        "decontamination": {
            "status": "BLOCKED_DATA232_MISSING_TERMINAL_DATA230",
            "attempted": True,
            "scan_executed": False,
            "blocking_reason": "No terminal DATA-230 corpus identity/inventory is published.",
            "data232_head_sha": DATA232_HEAD,
            "data232_blocker_git_blob_sha1": DATA232_BLOCKER_BLOB,
            "data232_config_git_blob_sha1": DATA232_CONFIG_BLOB,
            "data232_reserved_authorities_git_blob_sha1": DATA232_RESERVED_BLOB,
            "data232_report_identity_sha256": DATA232_REPORT_ID,
            "data232_final_test_identity_sha256": DATA232_FINAL_TEST_ID,
            "data232_reserved_authorities_identity_sha256": DATA232_RESERVED_ID,
            "observed_data230_branch": DATA230_BRANCH,
            "observed_data230_head_sha": DATA230_OBSERVED_HEAD,
            "observed_data230_head_message": DATA230_OBSERVED_MESSAGE,
            "training_corpus_identity": None,
            "selection_validation_identity": None,
            "evaluation_release_allowed": False,
        },
        "truth_boundary": {
            "existing_ua_en_seed_blob_preserved_byte_for_byte": True,
            "existing_ua_en_records_rewritten": False,
            "final_test_reclassified_into_selection_validation": False,
            "selection_validation_invented": False,
            "code_authority_complete": False,
            "ua_en_code_complete": False,
            "data232_decontamination_complete": False,
            "final_test_may_be_used_for_selection": False,
            "final_test_may_be_used_for_tokenizer_fit": False,
            "final_test_may_be_used_for_hyperparameter_selection": False,
            "local_free_only": True,
        },
    }
    manifest = dict(unsigned)
    manifest["manifest_identity_sha256"] = hash_json(unsigned)
    return manifest


def _json_blob(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def _write_immutable(output_dir: Path, rendered: dict[Path, bytes]) -> None:
    if output_dir.exists():
        actual = {p.relative_to(output_dir) for p in output_dir.rglob("*") if p.is_file()}
        if actual != set(rendered):
            raise Eval233Error("immutable output inventory differs")
        for rel, blob in rendered.items():
            if (output_dir / rel).read_bytes() != blob:
                raise Eval233Error(f"immutable output bytes differ: {rel}")
        return
    output_dir.mkdir(parents=True, exist_ok=False)
    for rel, blob in rendered.items():
        path = output_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)


def build(
    repo_root: Path,
    output_dir: Path,
    *,
    source_sha: str,
    expected_seed_git_blob_sha1: str = RECOVER174_SEED_GIT_BLOB_SHA1,
    expected_authority_git_blob_sha1: str = RECOVER174_AUTHORITY_GIT_BLOB_SHA1,
    expected_authority_identity: str = RECOVER174_AUTHORITY_ID,
) -> dict[str, Any]:
    seed_path, authority = _verify_upstream_exact(
        repo_root,
        expected_seed_git_blob_sha1=expected_seed_git_blob_sha1,
        expected_authority_git_blob_sha1=expected_authority_git_blob_sha1,
        expected_authority_identity=expected_authority_identity,
    )
    seed_blob = seed_path.read_bytes()
    rows = _load_and_validate_rows(seed_path, authority)
    selection = _selection_manifest()
    final = _final_manifest(seed_blob, rows, authority)
    root = _root_manifest(source_sha, seed_blob, selection, final, authority)
    rendered = {
        Path("selection-validation/manifest.json"): _json_blob(selection),
        Path("final-test/recover174_real_holdout_seed.jsonl.gz"): seed_blob,
        Path("final-test/manifest.json"): _json_blob(final),
        Path("manifest.json"): _json_blob(root),
    }
    _write_immutable(output_dir, rendered)
    return verify(output_dir)


def verify(output_dir: Path) -> dict[str, Any]:
    root = _read_json(output_dir / "manifest.json")
    if root.get("schema_version") != SCHEMA or root.get("worker_id") != WORKER_ID:
        raise Eval233Error("EVAL-233 root manifest schema/worker mismatch")
    root_unsigned = dict(root)
    supplied_root = str(root_unsigned.pop("manifest_identity_sha256", ""))
    if hash_json(root_unsigned) != supplied_root:
        raise Eval233Error("EVAL-233 root manifest identity mismatch")
    selection = _read_json(output_dir / "selection-validation/manifest.json")
    final = _read_json(output_dir / "final-test/manifest.json")
    for purpose, manifest in (("selection-validation", selection), ("final-test", final)):
        unsigned = dict(manifest)
        supplied = str(unsigned.pop("set_identity_sha256", ""))
        if hash_json(unsigned) != supplied:
            raise Eval233Error(f"{purpose} set identity mismatch")
    if selection.get("documents") != 0 or selection.get("records") != []:
        raise Eval233Error("selection-validation must remain empty while authority is absent")
    if selection.get("invented_from_final_test") is not False:
        raise Eval233Error("final-test data was reclassified into selection-validation")
    if any(
        selection.get(field) is not False
        for field in (
            "selection_eligible",
            "tokenizer_fit_eligible",
            "hyperparameter_selection_eligible",
        )
    ):
        raise Eval233Error("blocked selection-validation exposed to selection")
    if any(
        final.get(field) is not False
        for field in (
            "selection_eligible",
            "tokenizer_fit_eligible",
            "hyperparameter_selection_eligible",
        )
    ):
        raise Eval233Error("final-test exposed to selection/tokenizer/hyperparameter use")
    if final.get("final_test_exposure_prohibited") is not True:
        raise Eval233Error("final-test exposure prohibition missing")
    final_path = output_dir / "final-test/recover174_real_holdout_seed.jsonl.gz"
    final_blob = final_path.read_bytes()
    file_meta = final.get("files", {}).get("recover174_seed", {})
    if sha256_bytes(final_blob) != file_meta.get("sha256"):
        raise Eval233Error("final-test exact seed SHA-256 mismatch")
    if git_blob_sha1(final_blob) != RECOVER174_SEED_GIT_BLOB_SHA1:
        raise Eval233Error("final-test is not the exact RECOVER-174 seed blob")
    if root.get("code", {}).get("documents") != 0:
        raise Eval233Error("unauthorized code documents admitted")
    if root.get("decontamination", {}).get("evaluation_release_allowed") is not False:
        raise Eval233Error("DATA-232 blocker must keep evaluation release closed")
    return root


def _cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_p = sub.add_parser("build")
    build_p.add_argument("--repo-root", type=Path, required=True)
    build_p.add_argument("--output-dir", type=Path, required=True)
    build_p.add_argument("--source-sha", required=True)
    verify_p = sub.add_parser("verify")
    verify_p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = (
        build(args.repo_root, args.output_dir, source_sha=args.source_sha)
        if args.command == "build"
        else verify(args.output_dir)
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
