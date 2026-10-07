#!/usr/bin/env python3
"""Pin Hugging Face Xet object identities for the exact Rada_Trees revision.

This is a metadata-only, fail-closed acquisition step. It deliberately does not
download either archive and cannot grant corpus or training capacity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import validate_d03_rada_trees_acquisition_probe as parent_probe

REPO_ID = "uacorpus/Rada_Trees"
REVISION = "1b994a5804dcda122721e8d33a03fd172cf8d867"
EXPECTED_PATHS = ("Rada_Trees.7z", "rada_xtag_texts.7z")
DEFAULT_OUTPUT = ROOT / "evidence/d03-rada-trees/hf-object-identity-v1.json"
USER_AGENT = "12-6-ai-rada-trees-object-probe/1.0"

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ProbeError(RuntimeError):
    """Fail-closed metadata probe error."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _tree_url() -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in REPO_ID.split("/"))
    rev = urllib.parse.quote(REVISION, safe="")
    return (
        f"https://huggingface.co/api/datasets/{repo}/tree/{rev}"
        "?recursive=false&expand=false"
    )


def _resolve_url(path: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in REPO_ID.split("/"))
    rev = urllib.parse.quote(REVISION, safe="")
    encoded_path = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{encoded_path}"


def _http_json(url: str, timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise ProbeError(f"unexpected HTTP status {response.status} for {url}")
        return json.load(response)


def fetch_tree(timeout: float = 30.0) -> list[dict[str, Any]]:
    payload = _http_json(_tree_url(), timeout)
    if not isinstance(payload, list):
        raise ProbeError("Hugging Face tree response is not a JSON list")
    return payload


def fetch_resolve_xet_hash(path: str, timeout: float = 30.0) -> str:
    """Read X-Xet-Hash without following the large-file download redirect."""
    request = urllib.request.Request(
        _resolve_url(path),
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
        method="GET",
    )
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code not in {301, 302, 303, 307, 308}:
            raise ProbeError(f"unexpected resolve status {exc.code} for {path}") from exc
        headers = exc.headers
    else:
        try:
            # A large Xet-backed file should not be streamed here. A direct 200 is
            # rejected rather than risk silently downloading the archive.
            raise ProbeError(f"resolve endpoint did not redirect for {path}")
        finally:
            response.close()

    value = headers.get("X-Xet-Hash")
    if value is None or _HEX64.fullmatch(value) is None:
        raise ProbeError(f"missing or malformed X-Xet-Hash for {path}")
    return value


def _normalize_tree_entry(entry: dict[str, Any]) -> dict[str, Any]:
    path = entry.get("path")
    if path not in EXPECTED_PATHS:
        raise ProbeError(f"unexpected selected path: {path!r}")
    if entry.get("type") != "file":
        raise ProbeError(f"{path}: expected type=file")

    size = entry.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ProbeError(f"{path}: missing positive exact byte size")

    oid = entry.get("oid", entry.get("blobId", entry.get("blob_id")))
    if not isinstance(oid, str) or _HEX40.fullmatch(oid) is None:
        raise ProbeError(f"{path}: missing 40-hex Git blob OID")

    xet_hash = entry.get("xetHash", entry.get("xet_hash"))
    if not isinstance(xet_hash, str) or _HEX64.fullmatch(xet_hash) is None:
        raise ProbeError(f"{path}: missing 64-hex Xet object identity")

    lfs = entry.get("lfs")
    if lfs is not None and not isinstance(lfs, dict):
        raise ProbeError(f"{path}: malformed LFS metadata")

    return {
        "path": path,
        "size_bytes": size,
        "git_blob_oid": oid,
        "xet_hash": xet_hash,
    }


def select_archive_entries(tree: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_raw = [
        entry
        for entry in tree
        if isinstance(entry, dict) and entry.get("path") in EXPECTED_PATHS
    ]
    if len(selected_raw) != len(EXPECTED_PATHS):
        raise ProbeError(
            f"expected exactly {len(EXPECTED_PATHS)} archive entries, got {len(selected_raw)}"
        )
    normalized = [_normalize_tree_entry(entry) for entry in selected_raw]
    by_path = {item["path"]: item for item in normalized}
    if len(by_path) != len(EXPECTED_PATHS) or set(by_path) != set(EXPECTED_PATHS):
        raise ProbeError("archive path inventory is missing or duplicated")
    return [by_path[path] for path in EXPECTED_PATHS]


def build_snapshot(
    tree: list[dict[str, Any]],
    resolve_xet_hashes: dict[str, str],
) -> dict[str, Any]:
    parent_value = parent_probe.load_and_validate()
    if parent_value["source"]["dataset"] != REPO_ID:
        raise ProbeError("parent probe dataset drift")
    if parent_value["source"]["observed_head_sha"] != REVISION:
        raise ProbeError("parent probe revision drift")
    if parent_value["claim_boundary"]["training_authorized_bytes"] != 0:
        raise ProbeError("parent probe unexpectedly authorizes training bytes")

    files = select_archive_entries(tree)
    for item in files:
        observed = resolve_xet_hashes.get(item["path"])
        if observed != item["xet_hash"]:
            raise ProbeError(
                f"{item['path']}: resolve X-Xet-Hash does not match tree Xet identity"
            )

    payload: dict[str, Any] = {
        "schema_version": "12-6.d03-rada-trees-hf-object-identity.v1",
        "execution_profile": "LOCAL_FREE_METADATA_ONLY",
        "source": {
            "repo_id": REPO_ID,
            "repo_type": "dataset",
            "revision": REVISION,
            "tree_endpoint": _tree_url(),
        },
        "files": files,
        "verification": {
            "tree_revision_is_immutable_40hex": True,
            "git_blob_oids_bound": True,
            "xet_hashes_bound": True,
            "resolve_header_xet_hashes_match_tree": True,
        },
        "claim_boundary": {
            "archives_downloaded": False,
            "archive_content_sha256_verified": False,
            "archive_members_inventoried": False,
            "normalized_capacity_claimed": False,
            "training_authorized_bytes": 0,
            "training_exposure_authorized": False,
            "tokenizer_fit_authorized": False,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "paid_compute_used": False,
            "safe_result": "EXACT_HF_OBJECT_IDENTITIES_PINNED_DOWNLOAD_AND_MEMBER_AUDIT_REQUIRED",
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    payload["snapshot_identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def validate_snapshot(snapshot: dict[str, Any]) -> None:
    identity = snapshot.get("snapshot_identity_sha256")
    if not isinstance(identity, str) or _HEX64.fullmatch(identity) is None:
        raise ProbeError("missing snapshot identity")
    copy = dict(snapshot)
    del copy["snapshot_identity_sha256"]
    encoded = json.dumps(
        copy, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    if hashlib.sha256(encoded).hexdigest() != identity:
        raise ProbeError("snapshot identity mismatch")

    if snapshot["source"]["repo_id"] != REPO_ID or snapshot["source"]["revision"] != REVISION:
        raise ProbeError("snapshot source identity drift")
    if [item["path"] for item in snapshot["files"]] != list(EXPECTED_PATHS):
        raise ProbeError("snapshot archive order/path drift")
    if snapshot["claim_boundary"]["training_authorized_bytes"] != 0:
        raise ProbeError("metadata snapshot cannot authorize training")
    if snapshot["claim_boundary"]["archives_downloaded"] is not False:
        raise ProbeError("metadata snapshot cannot claim archive download")
    if snapshot["claim_boundary"]["archive_content_sha256_verified"] is not False:
        raise ProbeError("metadata snapshot cannot claim content SHA-256 verification")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tree-json",
        type=Path,
        help="Offline tree JSON fixture. Use with --resolve-json for deterministic replay.",
    )
    parser.add_argument(
        "--resolve-json",
        type=Path,
        help="Offline JSON object mapping archive path to X-Xet-Hash.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tree = (
        json.loads(args.tree_json.read_text(encoding="utf-8"))
        if args.tree_json is not None
        else fetch_tree(args.timeout)
    )
    if not isinstance(tree, list):
        raise ProbeError("tree fixture must be a JSON list")

    if args.resolve_json is not None:
        resolve_hashes = json.loads(args.resolve_json.read_text(encoding="utf-8"))
        if not isinstance(resolve_hashes, dict):
            raise ProbeError("resolve fixture must be a JSON object")
    else:
        resolve_hashes = {
            path: fetch_resolve_xet_hash(path, args.timeout) for path in EXPECTED_PATHS
        }

    snapshot = build_snapshot(tree, resolve_hashes)
    validate_snapshot(snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("D03_RADA_TREES_HF_OBJECT_PIN=PASS")
    print("SNAPSHOT_IDENTITY_SHA256=" + snapshot["snapshot_identity_sha256"])
    print("TRAINING_AUTHORIZED_BYTES=0")
    print("NEXT=DOWNLOAD_PRIMARY_ARCHIVE_VERIFY_CONTENT_SHA256_AND_INVENTORY_MEMBERS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
