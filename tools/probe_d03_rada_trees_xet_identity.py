#!/usr/bin/env python3
"""Pin exact Xet identities for Rada_Trees archives without downloading archive bodies."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping

CONFIG_SCHEMA = "12-6.d03-rada-trees-xet-identity-probe.v1"
REPORT_SCHEMA = "12-6.d03-rada-trees-xet-identity-report.v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_TREE_API_TEMPLATE = "https://huggingface.co/api/datasets/{repo_id}/tree/{revision}"
EXPECTED_RESOLVE_TEMPLATE = "https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"


class ProbeError(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProbeError(f"JSON root must be object: {path}")
    return value


def _validate_config(config: Mapping[str, Any], repo_root: Path) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ProbeError("unsupported config schema")
    if config.get("execution_profile") != "LOCAL_FREE":
        raise ProbeError("execution profile must remain LOCAL_FREE")
    boundary = config.get("claim_boundary")
    if not isinstance(boundary, Mapping):
        raise ProbeError("missing claim boundary")
    forbidden_truths = (
        "archive_bytes_downloaded",
        "archive_members_inventoried",
        "archive_content_sha256_claimed",
        "normalized_capacity_claimed",
        "evaluation_authorized",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "paid_compute_used",
    )
    if any(boundary.get(key) is not False for key in forbidden_truths):
        raise ProbeError("probe overclaims downstream state")
    if boundary.get("training_authorized_bytes") != 0 or boundary.get("optimizer_updates") != 0:
        raise ProbeError("probe must authorize zero training exposure")

    parent = config.get("parent_probe")
    if not isinstance(parent, Mapping):
        raise ProbeError("missing parent probe binding")
    parent_path = repo_root / str(parent.get("path", ""))
    raw = parent_path.read_bytes()
    if _git_blob_sha1(raw) != parent.get("git_blob_sha1"):
        raise ProbeError("parent probe Git blob drift")
    parent_json = json.loads(raw.decode("utf-8"))
    if parent_json.get("claim_boundary", {}).get("safe_result") != parent.get("required_safe_result"):
        raise ProbeError("parent probe safe-result drift")
    source = parent_json.get("source", {})
    upstream = config.get("upstream", {})
    if source.get("dataset") != upstream.get("repo_id") or source.get("observed_head_sha") != upstream.get("revision"):
        raise ProbeError("upstream identity no longer matches parent probe")
    if parent_json.get("rights", {}).get("dataset_card_license") != upstream.get("expected_license_discovery"):
        raise ProbeError("license discovery boundary drift")

    if upstream.get("repo_type") != "dataset":
        raise ProbeError("upstream repo type must remain dataset")
    if upstream.get("tree_api_template") != EXPECTED_TREE_API_TEMPLATE:
        raise ProbeError("Hub tree API template drift")
    if upstream.get("resolve_template") != EXPECTED_RESOLVE_TEMPLATE:
        raise ProbeError("Hub resolve template drift")

    revision = str(upstream.get("revision", ""))
    if not HEX40.fullmatch(revision):
        raise ProbeError("upstream revision must be exact lowercase 40-hex commit")

    identity_requirements = config.get("identity_requirements")
    if not isinstance(identity_requirements, Mapping):
        raise ProbeError("missing identity requirements")
    required_true = (
        "exact_revision_required",
        "git_blob_oid_required",
        "xet_hash_required",
        "tree_size_required",
        "resolve_header_xet_crosscheck_required",
    )
    if any(identity_requirements.get(key) is not True for key in required_true):
        raise ProbeError("identity requirement weakened")
    if identity_requirements.get("archive_bytes_downloaded_by_this_probe") is not False:
        raise ProbeError("identity probe may not download archive bodies")

    targets = config.get("archive_targets")
    if not isinstance(targets, list) or len(targets) != 2:
        raise ProbeError("expected exactly two archive targets")
    paths: list[str] = []
    for target in targets:
        if not isinstance(target, Mapping):
            raise ProbeError("archive target must be object")
        path = str(target.get("path", ""))
        if not path or path.startswith("/") or ".." in Path(path).parts:
            raise ProbeError("unsafe archive path")
        if target.get("training_capacity_credit_bytes") != 0:
            raise ProbeError("identity probe may not grant archive capacity")
        lower, upper = int(target.get("minimum_bytes", 0)), int(target.get("maximum_bytes", 0))
        if lower <= 0 or upper <= lower:
            raise ProbeError("invalid archive size guard")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ProbeError("duplicate archive target")


def _fetch_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "12-6-ai-rada-trees-xet-probe/1"})
    with urllib.request.urlopen(req, timeout=30) as response:
        if response.status != 200:
            raise ProbeError(f"unexpected HTTP status {response.status} for {url}")
        return json.loads(response.read().decode("utf-8"))


def _tree_entries(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    upstream = config["upstream"]
    url = str(upstream["tree_api_template"]).format(
        repo_id=urllib.parse.quote(str(upstream["repo_id"]), safe="/"),
        revision=str(upstream["revision"]),
    )
    value = _fetch_json(url)
    if not isinstance(value, list):
        raise ProbeError("Hub tree API did not return a list")
    return [item for item in value if isinstance(item, dict)]


def _resolve_headers(url: str) -> Mapping[str, str]:
    """Read Xet identity from the documented GET/302 resolve handshake.

    Hugging Face documents X-Xet-Hash on a GET request to the resolve URL and
    requires clients not to follow the redirect when using that identity.  A
    normal 200/206 response is rejected because accepting it could silently
    convert this metadata-only probe into an archive-body transfer.
    """

    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"User-Agent": "12-6-ai-rada-trees-xet-probe/1"},
    )
    try:
        response = opener.open(request, timeout=30)
    except urllib.error.HTTPError as exc:
        if exc.code not in (301, 302, 303, 307, 308):
            raise ProbeError(f"resolve metadata HTTP error {exc.code}") from exc
        return {k.lower(): v for k, v in exc.headers.items()}
    else:
        response.close()
        raise ProbeError(
            "resolve endpoint did not return a blocked redirect; refusing possible archive body"
        )


def _select_tree_record(entries: list[dict[str, Any]], path: str) -> dict[str, Any]:
    matches = [item for item in entries if item.get("path") == path]
    if len(matches) != 1:
        raise ProbeError(f"expected one exact Hub tree record for {path}, got {len(matches)}")
    record = matches[0]
    if record.get("type") != "file":
        raise ProbeError(f"tree entry is not a file: {path}")
    return record


def _normalize_file_record(
    config: Mapping[str, Any], target: Mapping[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    path = str(target["path"])
    oid = str(record.get("oid", ""))
    xet_hash = str(record.get("xetHash", ""))
    size = int(record.get("size", -1))
    if not HEX40.fullmatch(oid):
        raise ProbeError(f"missing/invalid Git blob OID for {path}")
    if not HEX64.fullmatch(xet_hash):
        raise ProbeError(f"missing/invalid Xet hash for {path}")
    if not (int(target["minimum_bytes"]) <= size <= int(target["maximum_bytes"])):
        raise ProbeError(f"archive size outside guarded discovery range: {path}={size}")

    upstream = config["upstream"]
    resolve_url = str(upstream["resolve_template"]).format(
        repo_id=urllib.parse.quote(str(upstream["repo_id"]), safe="/"),
        revision=str(upstream["revision"]),
        path=urllib.parse.quote(path, safe="/"),
    )
    headers = _resolve_headers(resolve_url)
    header_xet = headers.get("x-xet-hash")
    if header_xet != xet_hash:
        raise ProbeError(f"tree/resolve Xet identity mismatch for {path}")
    header_commit = headers.get("x-repo-commit")
    if header_commit is not None and header_commit != upstream["revision"]:
        raise ProbeError(f"resolve commit drift for {path}")
    linked_size = headers.get("x-linked-size")
    if linked_size is not None and int(linked_size) != size:
        raise ProbeError(f"tree/resolve size mismatch for {path}")

    return {
        "path": path,
        "role": str(target["role"]),
        "git_blob_oid": oid,
        "xet_hash": xet_hash,
        "size_bytes": size,
        "resolve_http_contract": "GET_NO_REDIRECT",
        "resolve_repo_commit": header_commit or str(upstream["revision"]),
        "resolve_xet_hash": header_xet,
        "resolve_linked_size_bytes": int(linked_size) if linked_size is not None else None,
        "archive_body_downloaded": False,
        "training_capacity_credit_bytes": 0,
    }


def probe(config: Mapping[str, Any], repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root)
    _validate_config(config, root)
    entries = _tree_entries(config)
    files = [
        _normalize_file_record(config, target, _select_tree_record(entries, str(target["path"])))
        for target in config["archive_targets"]
    ]
    total = sum(int(item["size_bytes"]) for item in files)
    core = {
        "schema_version": REPORT_SCHEMA,
        "worker_id": str(config["worker_id"]),
        "execution_profile": "LOCAL_FREE",
        "dataset": str(config["upstream"]["repo_id"]),
        "exact_revision": str(config["upstream"]["revision"]),
        "files": files,
        "archive_count": len(files),
        "total_archive_bytes": total,
        "claim_boundary": {
            "exact_archive_git_and_xet_identity_pinned": True,
            "archive_bodies_downloaded": False,
            "archive_content_sha256_claimed": False,
            "archive_members_inventoried": False,
            "normalized_capacity_claimed": False,
            "training_authorized_bytes": 0,
            "evaluation_authorized": False,
            "tokenizer_fit_authorized": False,
            "model_training_executed": False,
            "paid_compute_used": False,
            "next_gate": "DOWNLOAD_PRIMARY_ARCHIVE_VERIFY_CONTENT_SHA256_AND_INVENTORY_MEMBERS",
        },
    }
    return {**core, "report_sha256": _sha256(_canonical_bytes(core))}


def verify_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ProbeError("unsupported report schema")
    core = dict(report)
    expected = core.pop("report_sha256", None)
    if expected != _sha256(_canonical_bytes(core)):
        raise ProbeError("report self-hash mismatch")
    if report.get("archive_count") != 2:
        raise ProbeError("expected exactly two archive identities")
    files = report.get("files")
    if not isinstance(files, list) or len(files) != 2:
        raise ProbeError("invalid report file list")
    for item in files:
        if not isinstance(item, Mapping):
            raise ProbeError("invalid report file record")
        if not HEX40.fullmatch(str(item.get("git_blob_oid", ""))):
            raise ProbeError("invalid report Git blob OID")
        if not HEX64.fullmatch(str(item.get("xet_hash", ""))):
            raise ProbeError("invalid report Xet hash")
        if item.get("xet_hash") != item.get("resolve_xet_hash"):
            raise ProbeError("report Xet crosscheck failed")
        if item.get("resolve_http_contract") != "GET_NO_REDIRECT":
            raise ProbeError("report resolve HTTP contract drift")
        if item.get("archive_body_downloaded") is not False or item.get("training_capacity_credit_bytes") != 0:
            raise ProbeError("report overclaims archive use")
    boundary = report.get("claim_boundary", {})
    if boundary.get("archive_bodies_downloaded") is not False or boundary.get("training_authorized_bytes") != 0:
        raise ProbeError("report training/download boundary violated")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    config = _load_json(Path(args.config))
    report = probe(config, args.repo_root)
    verify_report(report)
    Path(args.report).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
