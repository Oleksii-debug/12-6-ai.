#!/usr/bin/env python3
"""Acquire the exact Rada_Trees primary archive from its pinned HF object snapshot.

This LOCAL_FREE acquisition layer deliberately grants zero training capacity. It
binds the Hugging Face resolve handoff to the immutable Xet object identity,
streams exactly the expected byte count, and requires the downloaded SHA-256 to
match the independently pinned content identity before atomic promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, BinaryIO, Mapping

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import run_d03_rada_trees_inventory_from_hf_snapshot as bridge

USER_AGENT = "12-6-ai-rada-trees-exact-acquisition/1"
CHUNK_BYTES = 1024 * 1024
MAX_ARCHIVE_BYTES = 1_000_000_000
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RESOLVE_URL = (
    "https://huggingface.co/datasets/"
    f"{bridge.REPO_ID}/resolve/{bridge.REVISION}/{bridge.PRIMARY_ARCHIVE}"
)
_ALLOWED_STORAGE_SUFFIXES = (
    ".huggingface.co",
    ".hf.co",
    ".xethub.hf.co",
)


class AcquisitionError(RuntimeError):
    """Fail-closed exact-source acquisition error."""


def _provider_https_url(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.fragment:
        return False
    host = (parsed.hostname or "").lower()
    if host == "huggingface.co":
        return True
    return any(host.endswith(suffix) for suffix in _ALLOWED_STORAGE_SUFFIXES)


def _normalized_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lower(): str(value).strip() for key, value in headers.items()}


def _pinned_archive_identity() -> dict[str, str]:
    config = bridge.inventory.load_config()
    archive = config["primary_archive"]
    content_sha256 = archive.get("exact_content_sha256")
    xet_hash = archive.get("content_xet_hash")
    if not isinstance(content_sha256, str) or HEX64.fullmatch(content_sha256) is None:
        raise AcquisitionError("inventory config has no valid pinned archive SHA-256")
    if content_sha256 != bridge.inventory.PINNED_CONTENT_SHA256:
        raise AcquisitionError("inventory config archive SHA-256 authority drift")
    if xet_hash != bridge.inventory.PINNED_XET_HASH:
        raise AcquisitionError("inventory config Xet authority drift")
    return {"content_sha256": content_sha256, "xet_hash": str(xet_hash)}


def validate_first_redirect(
    handoff: Mapping[str, Any],
    source_url: str,
    code: int,
    headers: Mapping[str, str],
    target_url: str,
) -> dict[str, Any]:
    """Validate the exact HF -> Xet/LFS transfer handoff before following it."""
    if source_url != RESOLVE_URL:
        raise AcquisitionError("resolve source URL drift")
    if code != 302:
        raise AcquisitionError(f"expected exact HTTP 302 resolve handoff, got {code}")
    if not _provider_https_url(target_url):
        raise AcquisitionError("resolve redirect target is outside approved HF storage origins")

    normalized = _normalized_headers(headers)
    if normalized.get("x-xet-hash") != handoff["xet_hash"]:
        raise AcquisitionError("resolve X-Xet-Hash detached from parent snapshot")

    repo_commit = normalized.get("x-repo-commit")
    if repo_commit is not None and repo_commit != bridge.REVISION:
        raise AcquisitionError("resolve X-Repo-Commit drift")

    linked_size = normalized.get("x-linked-size")
    if linked_size is not None:
        try:
            parsed_size = int(linked_size)
        except ValueError as exc:
            raise AcquisitionError("resolve X-Linked-Size is malformed") from exc
        if parsed_size != handoff["size_bytes"]:
            raise AcquisitionError("resolve X-Linked-Size detached from parent snapshot")

    return {
        "status": code,
        "target_host": urllib.parse.urlsplit(target_url).hostname,
        "xet_hash": handoff["xet_hash"],
        "repo_commit": repo_commit,
        "linked_size_bytes": int(linked_size) if linked_size is not None else None,
    }


class _SnapshotBoundRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, handoff: Mapping[str, Any]) -> None:
        super().__init__()
        self.handoff = handoff
        self.first_redirect: dict[str, Any] | None = None
        self.redirect_count = 0

    def redirect_request(  # type: ignore[no-untyped-def]
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl,
    ):
        self.redirect_count += 1
        if self.redirect_count == 1:
            self.first_redirect = validate_first_redirect(
                self.handoff,
                req.full_url,
                code,
                headers,
                newurl,
            )
        elif not _provider_https_url(newurl):
            raise AcquisitionError("secondary redirect left approved HF storage origins")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _stream_exact_bytes(
    response: BinaryIO,
    output: BinaryIO,
    expected_size: int,
    expected_sha256: str,
) -> tuple[int, str]:
    if expected_size <= 0 or expected_size > MAX_ARCHIVE_BYTES:
        raise AcquisitionError("snapshot archive byte size is outside acquisition bound")
    if HEX64.fullmatch(expected_sha256) is None:
        raise AcquisitionError("pinned archive SHA-256 is malformed")

    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = response.read(CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > expected_size:
            raise AcquisitionError("download exceeded exact snapshot byte size")
        digest.update(chunk)
        output.write(chunk)

    if total != expected_size:
        raise AcquisitionError(
            f"download byte-size mismatch expected={expected_size} actual={total}"
        )
    observed_sha256 = digest.hexdigest()
    if observed_sha256 != expected_sha256:
        raise AcquisitionError("download SHA-256 mismatch against pinned source authority")
    return total, observed_sha256


def acquire(
    snapshot: dict[str, Any],
    destination: Path,
    *,
    timeout: float = 120.0,
) -> dict[str, Any]:
    handoff = bridge.validate_hf_object_snapshot(snapshot)
    pinned = _pinned_archive_identity()
    if handoff["xet_hash"] != pinned["xet_hash"]:
        raise AcquisitionError("parent snapshot Xet identity detached from pinned source authority")
    if handoff["size_bytes"] > MAX_ARCHIVE_BYTES:
        raise AcquisitionError("parent snapshot exceeds acquisition byte bound")
    if destination.name != bridge.PRIMARY_ARCHIVE:
        raise AcquisitionError(f"destination filename must be {bridge.PRIMARY_ARCHIVE}")
    if destination.exists() or destination.is_symlink():
        raise AcquisitionError("refusing to overwrite an existing archive destination")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise AcquisitionError("stale partial archive exists; remove it before retry")

    redirect = _SnapshotBoundRedirect(handoff)
    opener = urllib.request.build_opener(redirect)
    request = urllib.request.Request(
        RESOLVE_URL,
        method="GET",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
        },
    )

    try:
        with opener.open(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            if status != 200:
                raise AcquisitionError(f"unexpected final archive HTTP status: {status}")
            if redirect.first_redirect is None:
                raise AcquisitionError("archive response arrived without validated HF redirect")

            final_url = response.geturl()
            if not _provider_https_url(final_url):
                raise AcquisitionError("final archive response is outside approved HF origins")

            content_encoding = response.headers.get("Content-Encoding")
            if content_encoding not in (None, "", "identity"):
                raise AcquisitionError("archive response applied content encoding")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    parsed_length = int(content_length)
                except ValueError as exc:
                    raise AcquisitionError("archive Content-Length is malformed") from exc
                if parsed_length != handoff["size_bytes"]:
                    raise AcquisitionError("archive Content-Length detached from snapshot size")

            with temporary.open("xb") as output:
                size_bytes, sha256 = _stream_exact_bytes(
                    response,
                    output,
                    handoff["size_bytes"],
                    pinned["content_sha256"],
                )
                output.flush()
                os.fsync(output.fileno())
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise AcquisitionError(f"exact archive acquisition failed: {exc}") from exc
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    os.replace(temporary, destination)
    report: dict[str, Any] = {
        "schema_version": "12-6.d03-rada-trees-exact-acquisition.v1",
        "state": "EXACT_HF_TRANSPORT_AND_PINNED_CONTENT_IDENTITY_ACQUIRED_ZERO_CREDIT",
        "dataset": bridge.REPO_ID,
        "exact_revision": bridge.REVISION,
        "object_snapshot_identity_sha256": handoff["snapshot_identity_sha256"],
        "resolve_url": RESOLVE_URL,
        "first_redirect": redirect.first_redirect,
        "archive": {
            "path": bridge.PRIMARY_ARCHIVE,
            "size_bytes": size_bytes,
            "git_blob_oid": handoff["git_blob_oid"],
            "xet_hash": handoff["xet_hash"],
            "expected_content_sha256": pinned["content_sha256"],
            "content_sha256": sha256,
            "content_sha256_matches_pinned_authority": True,
            "transport_bound_to_exact_resolve": True,
        },
        "claim_boundary": {
            "archive_downloaded": True,
            "archive_sha256_verified_against_pinned_authority": True,
            "archive_member_inventory_complete": False,
            "plain_text_classification_complete": False,
            "period_provenance_stratification_complete": False,
            "language_quality_privacy_complete": False,
            "global_lineage_dedup_complete": False,
            "evaluation_decontamination_complete": False,
            "training_authorized_bytes": 0,
            "training_exposure_authorized": False,
            "tokenizer_fit_authorized": False,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "paid_compute_used": False,
        },
        "next_gate": "RUN_HF_SNAPSHOT_INVENTORY_BRIDGE_WITH_PINNED_CONTENT_SHA256",
    }
    stable = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    report["report_identity_sha256"] = hashlib.sha256(stable).hexdigest()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-snapshot", type=Path, required=True)
    parser.add_argument("--archive-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = bridge.load_snapshot(args.object_snapshot)
    try:
        report = acquire(snapshot, args.archive_output, timeout=args.timeout)
    except (AcquisitionError, bridge.HandoffError) as exc:
        print(f"BLOCKED: {exc}")
        return 2

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = args.report_output.with_suffix(args.report_output.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, args.report_output)
    print("D03_RADA_TREES_EXACT_ACQUISITION=ACQUIRED_ZERO_CREDIT")
    print("CONTENT_SHA256=" + report["archive"]["content_sha256"])
    print("TRAINING_AUTHORIZED_BYTES=0")
    print("NEXT=RUN_HF_SNAPSHOT_INVENTORY_BRIDGE_WITH_PINNED_CONTENT_SHA256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
