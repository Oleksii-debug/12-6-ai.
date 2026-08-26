#!/usr/bin/env python3
"""Fail-closed immutable-object discovery for the Rada_Trees primary archive.

This tool deliberately does not download the archive body. It pins the exact
dataset revision, reads file metadata from the Hugging Face tree API, and makes
a no-redirect resolve request so an X-Xet-Hash can be observed without
following the large-file redirect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/data/d03_rada_trees_object_identity_v1.json"

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_PATH = re.compile(r"^[A-Za-z0-9._/-]+$")


class DiscoveryError(RuntimeError):
    """Raised when immutable object discovery cannot be proven fail-closed."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _require_exact_revision(value: str) -> str:
    if not HEX40.fullmatch(value):
        raise DiscoveryError("revision must be an exact lowercase 40-hex commit")
    return value


def _require_path(value: str) -> str:
    if not value or value.startswith("/") or ".." in value.split("/") or not SAFE_PATH.fullmatch(value):
        raise DiscoveryError(f"unsafe repository path: {value!r}")
    return value


def tree_api_url(repo_id: str, revision: str) -> str:
    _require_exact_revision(revision)
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    quoted_revision = urllib.parse.quote(revision, safe="")
    return (
        f"https://huggingface.co/api/datasets/{quoted_repo}/tree/{quoted_revision}"
        "?recursive=true&expand=true"
    )


def resolve_url(repo_id: str, revision: str, path: str) -> str:
    _require_exact_revision(revision)
    _require_path(path)
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_path = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    return f"https://huggingface.co/datasets/{quoted_repo}/resolve/{quoted_revision}/{quoted_path}"


def _load_json_url(url: str, *, timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "12-6-ai-rada-trees-object-discovery/1"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _no_redirect_headers(url: str, *, timeout: float) -> tuple[int, Mapping[str, str]]:
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "12-6-ai-rada-trees-object-discovery/1"},
        method="GET",
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            return int(response.status), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        # urllib surfaces an intentionally un-followed redirect as HTTPError.
        if exc.code in {301, 302, 303, 307, 308}:
            return int(exc.code), dict(exc.headers.items())
        raise DiscoveryError(f"resolve request failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise DiscoveryError(f"resolve request failed: {exc.reason}") from exc


def _lower_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lower(): str(value).strip() for key, value in headers.items()}


def _extract_object(entry: Mapping[str, Any], *, expected_path: str) -> dict[str, Any]:
    if entry.get("type") != "file":
        raise DiscoveryError(f"{expected_path} is not a file")
    if entry.get("path") != expected_path:
        raise DiscoveryError("tree entry path drift")
    size = entry.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise DiscoveryError("archive size must be a positive integer")
    blob_id = entry.get("oid") or entry.get("blobId") or entry.get("blob_id")
    if not isinstance(blob_id, str) or not HEX40.fullmatch(blob_id):
        raise DiscoveryError("exact Git blob id is missing or malformed")

    lfs = entry.get("lfs")
    lfs_oid = None
    lfs_size = None
    if lfs is not None:
        if not isinstance(lfs, Mapping):
            raise DiscoveryError("lfs metadata must be an object")
        raw_oid = lfs.get("oid")
        if raw_oid is not None:
            if not isinstance(raw_oid, str):
                raise DiscoveryError("lfs oid must be a string")
            raw_oid = raw_oid.removeprefix("sha256:")
            if not HEX64.fullmatch(raw_oid):
                raise DiscoveryError("lfs oid must be a sha256 identity")
            lfs_oid = raw_oid
        raw_size = lfs.get("size")
        if raw_size is not None:
            if not isinstance(raw_size, int) or isinstance(raw_size, bool) or raw_size <= 0:
                raise DiscoveryError("lfs size must be a positive integer")
            lfs_size = raw_size
            if lfs_size != size:
                raise DiscoveryError("tree size and lfs size disagree")

    xet_hash = entry.get("xetHash") or entry.get("xet_hash")
    if xet_hash is not None:
        if not isinstance(xet_hash, str) or not HEX64.fullmatch(xet_hash):
            raise DiscoveryError("xet hash must be a lowercase 64-hex identity")

    if xet_hash is None and lfs_oid is None:
        raise DiscoveryError("archive has neither Xet nor LFS immutable object identity")

    return {
        "path": expected_path,
        "size_bytes": size,
        "git_blob_id": blob_id,
        "lfs_sha256": lfs_oid,
        "xet_hash": xet_hash,
    }


def parse_tree_payload(payload: Any, *, primary_path: str, secondary_path: str) -> dict[str, Any]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise DiscoveryError("tree API payload must be a list")
    files: dict[str, Mapping[str, Any]] = {}
    wanted = {primary_path, secondary_path}
    for raw in payload:
        if not isinstance(raw, Mapping):
            raise DiscoveryError("tree API entry must be an object")
        path = raw.get("path")
        if path not in wanted:
            continue
        if path in files:
            raise DiscoveryError(f"duplicate tree entry for {path}")
        files[str(path)] = raw
    missing = wanted - files.keys()
    if missing:
        raise DiscoveryError(f"required archive entries missing: {sorted(missing)}")
    return {
        "primary": _extract_object(files[primary_path], expected_path=primary_path),
        "secondary": _extract_object(files[secondary_path], expected_path=secondary_path),
    }


def _validate_parent(config: Mapping[str, Any]) -> None:
    parent = config["parent_probe"]
    parent_path = ROOT / str(parent["config_path"])
    if not parent_path.is_file():
        raise DiscoveryError(f"parent probe config missing: {parent_path}")
    data = parent_path.read_bytes()
    actual = _git_blob_sha1(data)
    if actual != parent["config_git_blob_sha1"]:
        raise DiscoveryError(
            f"parent probe config blob drift: expected {parent['config_git_blob_sha1']} got {actual}"
        )
    parsed = json.loads(data)
    if parsed["source"]["observed_head_sha"] != config["source"]["revision"]:
        raise DiscoveryError("parent probe revision does not match object-discovery revision")
    if parsed["source"]["dataset"] != config["source"]["repo_id"]:
        raise DiscoveryError("parent probe dataset does not match object-discovery dataset")
    if parsed["claim_boundary"]["training_authorized_bytes"] != 0:
        raise DiscoveryError("parent probe unexpectedly grants training capacity")


def validate_config(config: Mapping[str, Any]) -> None:
    if config["schema_version"] != "12-6.d03-rada-trees-object-identity.v1":
        raise DiscoveryError("unexpected schema version")
    if config["execution_profile"] != "LOCAL_FREE":
        raise DiscoveryError("only LOCAL_FREE execution is allowed")
    source = config["source"]
    if source["repo_type"] != "dataset" or source["repo_id"] != "uacorpus/Rada_Trees":
        raise DiscoveryError("unexpected source repository")
    _require_exact_revision(str(source["revision"]))
    _require_path(str(source["primary_archive_path"]))
    _require_path(str(source["secondary_archive_path"]))
    if source["primary_archive_path"] == source["secondary_archive_path"]:
        raise DiscoveryError("primary and secondary archives must be distinct")
    contract = config["discovery_contract"]
    required_true = (
        "tree_api_must_use_exact_revision",
        "resolve_request_must_use_exact_revision",
        "require_positive_size",
        "require_git_blob_id",
        "require_xet_or_lfs_object_identity",
        "cross_check_tree_and_resolve_xet_hash_when_both_present",
    )
    if not all(contract.get(key) is True for key in required_true):
        raise DiscoveryError("required fail-closed discovery controls were weakened")
    if contract["resolve_redirects_followed"] is not False:
        raise DiscoveryError("resolve redirects must not be followed")
    if contract["archive_body_downloaded"] is not False:
        raise DiscoveryError("metadata discovery must not download the archive body")
    if contract["secondary_archive_training_credit_bytes"] != 0:
        raise DiscoveryError("secondary archive must remain zero-credit")
    boundary = config["claim_boundary"]
    if boundary["training_authorized_bytes"] != 0 or boundary["optimizer_updates"] != 0:
        raise DiscoveryError("metadata discovery cannot authorize training")
    for key in (
        "archive_downloaded",
        "archive_content_sha256_verified",
        "archive_members_inventoried",
        "normalized_capacity_claimed",
        "training_exposure_authorized",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "paid_compute_used",
    ):
        if boundary[key] is not False:
            raise DiscoveryError(f"claim boundary must remain false: {key}")


def build_report(
    config: Mapping[str, Any],
    tree_payload: Any,
    *,
    resolve_status: int,
    resolve_headers: Mapping[str, str],
) -> dict[str, Any]:
    validate_config(config)
    source = config["source"]
    parsed = parse_tree_payload(
        tree_payload,
        primary_path=str(source["primary_archive_path"]),
        secondary_path=str(source["secondary_archive_path"]),
    )
    headers = _lower_headers(resolve_headers)
    resolve_xet_hash = headers.get("x-xet-hash")
    if resolve_xet_hash is not None and not HEX64.fullmatch(resolve_xet_hash):
        raise DiscoveryError("resolve X-Xet-Hash is malformed")
    tree_xet_hash = parsed["primary"]["xet_hash"]
    if tree_xet_hash is not None and resolve_xet_hash is not None and tree_xet_hash != resolve_xet_hash:
        raise DiscoveryError("tree Xet hash and resolve X-Xet-Hash disagree")
    if tree_xet_hash is None and parsed["primary"]["lfs_sha256"] is None and resolve_xet_hash is None:
        raise DiscoveryError("primary archive immutable object identity is absent")

    report: dict[str, Any] = {
        "schema_version": "12-6.d03-rada-trees-object-identity-report.v1",
        "source": {
            "repo_id": source["repo_id"],
            "revision": source["revision"],
            "tree_api_url": tree_api_url(str(source["repo_id"]), str(source["revision"])),
            "resolve_url": resolve_url(
                str(source["repo_id"]),
                str(source["revision"]),
                str(source["primary_archive_path"]),
            ),
        },
        "primary_archive": parsed["primary"],
        "secondary_archive": {
            **parsed["secondary"],
            "training_capacity_credit_bytes": 0,
            "status": "HOLD_ANNOTATION_OR_DERIVATIVE_PENDING_MEMBER_LINEAGE_AUDIT",
        },
        "resolve_observation": {
            "status_code": resolve_status,
            "redirect_followed": False,
            "xet_hash": resolve_xet_hash,
            "location_present": "location" in headers,
        },
        "identity_decision": "IMMUTABLE_OBJECT_METADATA_PINNED_DOWNLOAD_BODY_NOT_EXECUTED",
        "claim_boundary": dict(config["claim_boundary"]),
    }
    report["report_identity_sha256"] = hashlib.sha256(
        (json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    ).hexdigest()
    validate_report(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report["schema_version"] != "12-6.d03-rada-trees-object-identity-report.v1":
        raise DiscoveryError("unexpected report schema")
    source = report["source"]
    _require_exact_revision(str(source["revision"]))
    primary = report["primary_archive"]
    if not isinstance(primary["size_bytes"], int) or primary["size_bytes"] <= 0:
        raise DiscoveryError("invalid primary archive size")
    if not HEX40.fullmatch(str(primary["git_blob_id"])):
        raise DiscoveryError("invalid primary Git blob identity")
    xet_hash = primary.get("xet_hash") or report["resolve_observation"].get("xet_hash")
    lfs_sha256 = primary.get("lfs_sha256")
    if xet_hash is None and lfs_sha256 is None:
        raise DiscoveryError("report lacks an immutable primary object identity")
    if xet_hash is not None and not HEX64.fullmatch(str(xet_hash)):
        raise DiscoveryError("invalid primary Xet hash")
    if lfs_sha256 is not None and not HEX64.fullmatch(str(lfs_sha256)):
        raise DiscoveryError("invalid primary LFS sha256")
    if report["resolve_observation"]["redirect_followed"] is not False:
        raise DiscoveryError("report cannot claim followed large-file redirect")
    boundary = report["claim_boundary"]
    if boundary["archive_downloaded"] is not False:
        raise DiscoveryError("metadata report cannot claim archive download")
    if boundary["archive_content_sha256_verified"] is not False:
        raise DiscoveryError("metadata report cannot claim content SHA verification")
    if boundary["training_authorized_bytes"] != 0:
        raise DiscoveryError("metadata report cannot grant training bytes")
    if boundary["model_training_executed"] is not False or boundary["paid_compute_used"] is not False:
        raise DiscoveryError("metadata report cannot claim training or paid compute")

    claimed_identity = report.get("report_identity_sha256")
    if not isinstance(claimed_identity, str) or not HEX64.fullmatch(claimed_identity):
        raise DiscoveryError("report identity is missing")
    copy = dict(report)
    copy.pop("report_identity_sha256", None)
    expected = hashlib.sha256(
        (json.dumps(copy, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    ).hexdigest()
    if claimed_identity != expected:
        raise DiscoveryError("report self-hash mismatch")


def discover(config_path: Path, report_path: Path, *, timeout: float) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    _validate_parent(config)
    source = config["source"]
    tree = _load_json_url(
        tree_api_url(str(source["repo_id"]), str(source["revision"])),
        timeout=timeout,
    )
    status, headers = _no_redirect_headers(
        resolve_url(
            str(source["repo_id"]),
            str(source["revision"]),
            str(source["primary_archive_path"]),
        ),
        timeout=timeout,
    )
    report = build_report(config, tree, resolve_status=status, resolve_headers=headers)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    discover_parser = sub.add_parser("discover")
    discover_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    discover_parser.add_argument("--report", type=Path, required=True)
    discover_parser.add_argument("--timeout", type=float, default=20.0)

    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--report", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "discover":
        discover(args.config, args.report, timeout=args.timeout)
    else:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        validate_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
