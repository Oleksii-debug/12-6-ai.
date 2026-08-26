"""EVAL-233 immutable real-source holdout v2 materializer.

The implementation preserves RECOVER-174 UA/EN JSONL record bytes exactly,
separates selection-validation from final-test, and fails closed on code and
corpus decontamination until independent DATA-227/DATA-232 authorities exist.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

WORKER_ID = "EVAL-233-REAL-HOLDOUT-V2"
SCHEMA = "12-6.eval233-real-holdout-v2.v1"
SET_SCHEMA = "12-6.eval233-real-holdout-set.v1"
RECOVER174_SEED_PATH = Path("data/evaluation/recover174_real_holdout_seed.jsonl.gz")
RECOVER174_AUTHORITY_PATH = Path("configs/evaluation/recover174_source_authority_v1.json")
RECOVER174_SEED_GIT_BLOB_SHA1 = "4bfbfbf29fa9538cabda6068efd3a1fd036a9479"
RECOVER174_AUTHORITY_GIT_BLOB_SHA1 = "3ba9f221a82468f971c17eda518cd6f1642fd311"
RECOVER174_AUTHORITY_ID = "c7211b3e1e6a4f22463d0e6174f0d6162c2452585704efad5564a35de8de609f"
PARTITION_POLICY = "sha256(eval233-v2\\0modality\\0record_id); first half selection-validation"
NORMALIZATION_POLICY = "PRESERVE_RECOVER174_DECOMPRESSED_JSONL_ROW_BYTES_NO_RENORMALIZATION"
DATA227_BRANCH = "data227/real-code-source-admission-v2-20260826"
DATA232_WORKER = "DATA-232"


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
    # SHA-1 is required here because this verifies an immutable Git object identity.
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
    seed = repo_root / RECOVER174_SEED_PATH
    authority_path = repo_root / RECOVER174_AUTHORITY_PATH
    if not seed.is_file() or not authority_path.is_file():
        raise Eval233Error("RECOVER-174 exact seed/authority inputs are missing")
    seed_blob = seed.read_bytes()
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
    return seed, authority


def _load_seed_lines(seed_path: Path) -> list[tuple[bytes, dict[str, Any]]]:
    try:
        decompressed = gzip.decompress(seed_path.read_bytes())
    except (OSError, EOFError) as exc:
        raise Eval233Error("unable to decompress RECOVER-174 seed") from exc
    raw_lines = decompressed.splitlines(keepends=True)
    if not raw_lines:
        raise Eval233Error("RECOVER-174 seed is empty")
    parsed: list[tuple[bytes, dict[str, Any]]] = []
    for raw in raw_lines:
        if not raw.endswith(b"\n"):
            raise Eval233Error("every RECOVER-174 seed row must end with newline")
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Eval233Error("invalid RECOVER-174 seed JSONL row") from exc
        if not isinstance(row, dict):
            raise Eval233Error("RECOVER-174 seed rows must be JSON objects")
        parsed.append((raw, row))
    return parsed


def _validate_record(row: dict[str, Any], authority: dict[str, Any]) -> None:
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
        raise Eval233Error(f"source {source_id!r} missing from evaluation-use authority")
    if source.get("evaluation_status") != "APPROVED_FOR_HELDOUT_EVALUATION":
        raise Eval233Error(f"source {source_id!r} lacks heldout evaluation approval")
    admitted = source.get("admitted_source_snapshots_sha256", [])
    if row["source_snapshot_sha256"] not in admitted:
        raise Eval233Error(f"record snapshot for {source_id!r} not admitted by authority")
    text = row["text"]
    if not isinstance(text, str) or not text:
        raise Eval233Error("heldout record text must be non-empty UTF-8 text")
    text_bytes = text.encode("utf-8")
    if "content_sha256" in row and row["content_sha256"] != sha256_bytes(text_bytes):
        raise Eval233Error("record content_sha256 does not match exact text bytes")
    if "source_bytes" in row and row["source_bytes"] != len(text_bytes):
        raise Eval233Error("record source_bytes does not match exact text bytes")


def _partition(
    rows: list[tuple[bytes, dict[str, Any]]],
) -> dict[str, list[tuple[bytes, dict[str, Any]]]]:
    by_modality: dict[str, list[tuple[bytes, dict[str, Any]]]] = {"ua": [], "en": []}
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    for raw, row in rows:
        record_id = str(row["record_id"])
        if record_id in seen_ids:
            raise Eval233Error("duplicate RECOVER-174 record_id")
        seen_ids.add(record_id)
        content = sha256_bytes(str(row["text"]).encode("utf-8"))
        if content in seen_content:
            raise Eval233Error("duplicate exact content in RECOVER-174 seed")
        seen_content.add(content)
        by_modality[str(row["modality"])].append((raw, row))

    selection: list[tuple[bytes, dict[str, Any]]] = []
    final: list[tuple[bytes, dict[str, Any]]] = []
    for modality in ("ua", "en"):
        items = by_modality[modality]
        if len(items) < 2 or len(items) % 2:
            raise Eval233Error(f"{modality} record count must be even and at least 2")
        ordered = sorted(
            items,
            key=lambda item: hashlib.sha256(
                b"eval233-v2\0"
                + modality.encode("ascii")
                + b"\0"
                + str(item[1]["record_id"]).encode("utf-8")
            ).digest(),
        )
        cut = len(ordered) // 2
        selection.extend(ordered[:cut])
        final.extend(ordered[cut:])
    return {"selection-validation": selection, "final-test": final}


def _record_binding(
    raw: bytes,
    row: dict[str, Any],
    authority: dict[str, Any],
    purpose: str,
) -> dict[str, Any]:
    source = authority["sources"][str(row["source_id"])]
    return {
        "record_id": str(row["record_id"]),
        "modality": str(row["modality"]),
        "source_id": str(row["source_id"]),
        "source_family": str(row["source_family"]),
        "source_version": str(row["source_version"]),
        "purpose": purpose,
        "raw_source_sha256": list(source.get("raw_sha256", [])),
        "extracted_normalized_snapshot_sha256": str(row["source_snapshot_sha256"]),
        "upstream_source_identity_sha256": str(source.get("source_identity_sha256", "")),
        "evaluation_use_authority_ref": str(row["evaluation_use_authority_ref"]),
        "provenance_ref": str(row["provenance_ref"]),
        "normalization_policy": NORMALIZATION_POLICY,
        "content_sha256": sha256_bytes(str(row["text"]).encode("utf-8")),
        "jsonl_row_bytes_sha256": sha256_bytes(raw),
        "jsonl_row_bytes": len(raw),
    }


def _render_set(
    purpose: str,
    items: list[tuple[bytes, dict[str, Any]]],
    authority: dict[str, Any],
) -> tuple[dict[str, Any], dict[Path, bytes]]:
    by_modality = {"ua": [], "en": []}
    for raw, row in items:
        by_modality[str(row["modality"])].append((raw, row))
    rendered: dict[Path, bytes] = {}
    files: dict[str, Any] = {}
    bindings: list[dict[str, Any]] = []
    for modality in ("ua", "en"):
        subset = sorted(by_modality[modality], key=lambda item: str(item[1]["record_id"]))
        blob = b"".join(raw for raw, _ in subset)
        rel = Path(purpose) / f"{modality}.jsonl"
        rendered[rel] = blob
        files[modality] = {
            "path": str(rel),
            "documents": len(subset),
            "bytes": len(blob),
            "sha256": sha256_bytes(blob),
        }
        bindings.extend(_record_binding(raw, row, authority, purpose) for raw, row in subset)
    unsigned = {
        "schema_version": SET_SCHEMA,
        "worker_id": WORKER_ID,
        "purpose": purpose,
        "modalities": ["ua", "en"],
        "files": files,
        "records": bindings,
        "selection_eligible": purpose == "selection-validation",
        "tokenizer_fit_eligible": purpose == "selection-validation",
        "hyperparameter_selection_eligible": purpose == "selection-validation",
        "final_test_exposure_prohibited": purpose == "final-test",
        "normalization_policy": NORMALIZATION_POLICY,
    }
    manifest = dict(unsigned)
    manifest["set_identity_sha256"] = hash_json(unsigned)
    manifest_blob = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )
    rendered[Path(purpose) / "manifest.json"] = manifest_blob
    return manifest, rendered


def _root_manifest(
    source_sha: str,
    seed_path: Path,
    authority: dict[str, Any],
    set_manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    seed_bytes = seed_path.read_bytes()
    selection_ids = {r["record_id"] for r in set_manifests["selection-validation"]["records"]}
    final_ids = {r["record_id"] for r in set_manifests["final-test"]["records"]}
    if selection_ids & final_ids:
        raise Eval233Error("selection-validation and final-test record IDs overlap")
    selection_content = {
        r["content_sha256"] for r in set_manifests["selection-validation"]["records"]
    }
    final_content = {r["content_sha256"] for r in set_manifests["final-test"]["records"]}
    if selection_content & final_content:
        raise Eval233Error("selection-validation and final-test exact content overlaps")
    unsigned = {
        "schema_version": SCHEMA,
        "worker_id": WORKER_ID,
        "source_sha": source_sha,
        "status": "PARTIAL_UA_EN_CODE_AND_DECONTAMINATION_BLOCKED",
        "upstream": {
            "recover174_seed_path": str(RECOVER174_SEED_PATH),
            "recover174_seed_git_blob_sha1": RECOVER174_SEED_GIT_BLOB_SHA1,
            "recover174_seed_sha256": sha256_bytes(seed_bytes),
            "recover174_authority_path": str(RECOVER174_AUTHORITY_PATH),
            "recover174_authority_git_blob_sha1": RECOVER174_AUTHORITY_GIT_BLOB_SHA1,
            "recover174_authority_identity_sha256": authority["authority_identity_sha256"],
        },
        "partition": {
            "policy": PARTITION_POLICY,
            "policy_identity_sha256": sha256_bytes(PARTITION_POLICY.encode("utf-8")),
            "sets": {
                purpose: {
                    "set_identity_sha256": manifest["set_identity_sha256"],
                    "documents": len(manifest["records"]),
                    "selection_eligible": manifest["selection_eligible"],
                    "tokenizer_fit_eligible": manifest["tokenizer_fit_eligible"],
                    "hyperparameter_selection_eligible": manifest[
                        "hyperparameter_selection_eligible"
                    ],
                }
                for purpose, manifest in set_manifests.items()
            },
            "record_id_overlap": 0,
            "content_sha256_overlap": 0,
        },
        "code": {
            "documents": 0,
            "status": "BLOCKED_NO_TERMINAL_DATA227_EVALUATION_USE_AUTHORITY",
            "data227_branch_observed": DATA227_BRANCH,
            "training_rights_do_not_imply_evaluation_rights": True,
            "unauthorized_code_bytes_admitted": False,
        },
        "decontamination": {
            "status": "BLOCKED_NO_PUBLISHED_DATA232_AUTHORITY",
            "required_worker": DATA232_WORKER,
            "current_corpus_candidate_identity_sha256": None,
            "data232_report_identity_sha256": None,
            "evaluation_release_allowed": False,
        },
        "truth_boundary": {
            "existing_ua_en_record_bytes_preserved": True,
            "code_authority_complete": False,
            "ua_en_code_complete": False,
            "data232_decontamination_complete": False,
            "selection_validation_may_be_used_for_selection": True,
            "final_test_may_be_used_for_selection": False,
            "final_test_may_be_used_for_tokenizer_fit": False,
            "final_test_may_be_used_for_hyperparameter_selection": False,
            "local_free_only": True,
        },
    }
    manifest = dict(unsigned)
    manifest["manifest_identity_sha256"] = hash_json(unsigned)
    return manifest


def _write_immutable(output_dir: Path, rendered: dict[Path, bytes]) -> None:
    if output_dir.exists():
        actual_paths = {
            p.relative_to(output_dir)
            for p in output_dir.rglob("*")
            if p.is_file()
        }
        if actual_paths != set(rendered):
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
    rows = _load_seed_lines(seed_path)
    for _, row in rows:
        _validate_record(row, authority)
    partition = _partition(rows)
    rendered: dict[Path, bytes] = {}
    set_manifests: dict[str, dict[str, Any]] = {}
    for purpose in ("selection-validation", "final-test"):
        set_manifest, set_rendered = _render_set(purpose, partition[purpose], authority)
        set_manifests[purpose] = set_manifest
        rendered.update(set_rendered)
    root_manifest = _root_manifest(source_sha, seed_path, authority, set_manifests)
    rendered[Path("manifest.json")] = (
        json.dumps(root_manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )
    _write_immutable(output_dir, rendered)
    return verify(output_dir)


def verify(output_dir: Path) -> dict[str, Any]:
    root = _read_json(output_dir / "manifest.json")
    if root.get("schema_version") != SCHEMA or root.get("worker_id") != WORKER_ID:
        raise Eval233Error("EVAL-233 root manifest schema/worker mismatch")
    supplied = str(root.get("manifest_identity_sha256", ""))
    unsigned = dict(root)
    unsigned.pop("manifest_identity_sha256", None)
    if hash_json(unsigned) != supplied:
        raise Eval233Error("EVAL-233 root manifest identity mismatch")
    if root.get("code", {}).get("documents") != 0:
        raise Eval233Error("unauthorized code documents admitted")
    if root.get("decontamination", {}).get("evaluation_release_allowed") is not False:
        raise Eval233Error("DATA-232 absence must keep evaluation release blocked")
    set_manifests: dict[str, dict[str, Any]] = {}
    for purpose in ("selection-validation", "final-test"):
        manifest = _read_json(output_dir / purpose / "manifest.json")
        supplied_set = str(manifest.get("set_identity_sha256", ""))
        unsigned_set = dict(manifest)
        unsigned_set.pop("set_identity_sha256", None)
        if hash_json(unsigned_set) != supplied_set:
            raise Eval233Error(f"{purpose} set identity mismatch")
        for modality in ("ua", "en"):
            meta = manifest.get("files", {}).get(modality)
            if not isinstance(meta, dict):
                raise Eval233Error(f"missing {purpose}/{modality} metadata")
            path = output_dir / str(meta["path"])
            blob = path.read_bytes()
            if sha256_bytes(blob) != meta["sha256"] or len(blob) != meta["bytes"]:
                raise Eval233Error(f"{purpose}/{modality} immutable file mismatch")
        if purpose == "final-test":
            if any(
                manifest.get(field) is not False
                for field in (
                    "selection_eligible",
                    "tokenizer_fit_eligible",
                    "hyperparameter_selection_eligible",
                )
            ):
                raise Eval233Error("final-test exposed to selection/tokenizer/hyperparameter use")
            if manifest.get("final_test_exposure_prohibited") is not True:
                raise Eval233Error("final-test exposure prohibition missing")
        set_manifests[purpose] = manifest
    s_ids = {r["record_id"] for r in set_manifests["selection-validation"]["records"]}
    f_ids = {r["record_id"] for r in set_manifests["final-test"]["records"]}
    if s_ids & f_ids:
        raise Eval233Error("cross-set record overlap")
    s_content = {r["content_sha256"] for r in set_manifests["selection-validation"]["records"]}
    f_content = {r["content_sha256"] for r in set_manifests["final-test"]["records"]}
    if s_content & f_content:
        raise Eval233Error("cross-set exact-content overlap")
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
    if args.command == "build":
        manifest = build(args.repo_root, args.output_dir, source_sha=args.source_sha)
    else:
        manifest = verify(args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
