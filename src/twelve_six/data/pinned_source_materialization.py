"""Fail-closed materialization of immutable source payload bytes.

This module bridges terminal source authorities to downstream corpus tooling without
implicitly trusting moving branches, mutable URLs, or unverified local files. It is
strictly a LOCAL_FREE data-engineering primitive; it does not normalize text or run
model training.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

CONFIG_SCHEMA = "12-6.pinned-source-payload-materialization.v1"
MANIFEST_SCHEMA = "12-6.pinned-source-payload-manifest.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
ALLOWED_PROVIDERS = {"git_object", "https_exact"}


class PinnedSourceMaterializationError(RuntimeError):
    """Raised when immutable source materialization cannot be proven exactly."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validated_repo_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise PinnedSourceMaterializationError("git_path must be a nonempty string")
    if "\\" in value or "\x00" in value or ":" in value:
        raise PinnedSourceMaterializationError(f"unsafe git_path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PinnedSourceMaterializationError(f"unsafe git_path: {value!r}")
    return value


def _validated_https_url(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise PinnedSourceMaterializationError("https_exact provider requires HTTPS URL")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise PinnedSourceMaterializationError("https_exact provider requires public HTTPS URL")
    return value


def _read_git_object(repo_root: Path, commit: str, git_path: str) -> bytes:
    if not COMMIT_RE.fullmatch(commit):
        raise PinnedSourceMaterializationError(
            "git_object provider requires an exact lowercase 40-hex git_commit"
        )
    path = _validated_repo_path(git_path)
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{commit}:{path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise PinnedSourceMaterializationError(
            f"pinned git object unavailable: {commit}:{path}: {detail}"
        )
    return bytes(result.stdout)


def _download_https_exact(url: str, expected_bytes: int) -> tuple[bytes, str]:
    _validated_https_url(url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "12-6-ai-pinned-payload-materializer/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            final_url = response.geturl()
            if urlparse(final_url).scheme != "https":
                raise PinnedSourceMaterializationError(
                    f"HTTPS source redirected to non-HTTPS URL: {final_url}"
                )
            payload = response.read(expected_bytes + 1)
    except PinnedSourceMaterializationError:
        raise
    except Exception as exc:  # urllib exposes several transport-specific exception types
        raise PinnedSourceMaterializationError(f"HTTPS acquisition failed: {url}: {exc}") from exc
    return payload, final_url


def _validate_config(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise PinnedSourceMaterializationError("unsupported materialization config schema")
    if config.get("local_free_only") is not True:
        raise PinnedSourceMaterializationError("materialization config must bind local_free_only=true")
    if config.get("model_training_executed") is not False:
        raise PinnedSourceMaterializationError(
            "materialization config must bind model_training_executed=false"
        )
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise PinnedSourceMaterializationError("sources must be a nonempty list")

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for raw in sources:
        if not isinstance(raw, Mapping):
            raise PinnedSourceMaterializationError("every source row must be an object")
        row = dict(raw)
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not SOURCE_ID_RE.fullmatch(source_id):
            raise PinnedSourceMaterializationError(f"invalid source_id: {source_id!r}")
        folded = source_id.casefold()
        if folded in seen:
            raise PinnedSourceMaterializationError(f"duplicate source_id: {source_id}")
        seen.add(folded)

        provider = row.get("provider")
        if provider not in ALLOWED_PROVIDERS:
            raise PinnedSourceMaterializationError(
                f"{source_id}: provider must be one of {sorted(ALLOWED_PROVIDERS)}"
            )
        expected_bytes = row.get("expected_raw_bytes")
        expected_sha = row.get("expected_raw_sha256")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes <= 0
        ):
            raise PinnedSourceMaterializationError(
                f"{source_id}: expected_raw_bytes must be a positive integer"
            )
        if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(expected_sha):
            raise PinnedSourceMaterializationError(f"{source_id}: invalid expected_raw_sha256")

        for key in ("source_family", "stratum", "authority_id", "authority_identity_sha256"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise PinnedSourceMaterializationError(f"{source_id}: missing {key}")
        if row["stratum"] not in {"ua", "en", "code"}:
            raise PinnedSourceMaterializationError(f"{source_id}: unsupported stratum")
        if not SHA256_RE.fullmatch(row["authority_identity_sha256"]):
            raise PinnedSourceMaterializationError(
                f"{source_id}: invalid authority_identity_sha256"
            )

        if provider == "git_object":
            commit = row.get("git_commit")
            if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
                raise PinnedSourceMaterializationError(
                    f"{source_id}: git_commit must be exact lowercase 40-hex"
                )
            _validated_repo_path(row.get("git_path"))
        else:
            _validated_https_url(row.get("url"))

        normalized_bytes = row.get("authority_normalized_bytes")
        normalized_sha = row.get("authority_normalized_sha256")
        if (normalized_bytes is None) != (normalized_sha is None):
            raise PinnedSourceMaterializationError(
                f"{source_id}: authority normalized bytes/hash must be supplied together"
            )
        if normalized_bytes is not None:
            if (
                not isinstance(normalized_bytes, int)
                or isinstance(normalized_bytes, bool)
                or normalized_bytes <= 0
            ):
                raise PinnedSourceMaterializationError(
                    f"{source_id}: invalid authority_normalized_bytes"
                )
            if not isinstance(normalized_sha, str) or not SHA256_RE.fullmatch(normalized_sha):
                raise PinnedSourceMaterializationError(
                    f"{source_id}: invalid authority_normalized_sha256"
                )
        rows.append(row)
    return rows


def materialize_pinned_sources(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
    output_dir: Path,
    https_downloader: Callable[[str, int], tuple[bytes, str]] | None = None,
) -> dict[str, Any]:
    """Materialize exact raw bytes and atomically publish an identity manifest.

    The caller is responsible for making pinned git commits reachable in the local
    repository before using ``git_object``. HTTPS payloads are accepted only when
    both byte count and SHA-256 match their terminal authority.
    """

    rows = _validate_config(config)
    downloader = https_downloader or _download_https_exact
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise PinnedSourceMaterializationError(
            f"output_dir already exists; refusing stale/partial reuse: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )

    try:
        manifest_rows: list[dict[str, Any]] = []
        for row in sorted(rows, key=lambda item: item["source_id"]):
            source_id = row["source_id"]
            provider = row["provider"]
            if provider == "git_object":
                payload = _read_git_object(repo_root, row["git_commit"], row["git_path"])
                source_locator = f"git:{row['git_commit']}:{row['git_path']}"
            else:
                payload, final_url = downloader(row["url"], row["expected_raw_bytes"])
                if urlparse(final_url).scheme != "https":
                    raise PinnedSourceMaterializationError(
                        f"{source_id}: downloader resolved to non-HTTPS URL"
                    )
                source_locator = row["url"]

            actual_bytes = len(payload)
            actual_sha = _sha256(payload)
            if actual_bytes != row["expected_raw_bytes"]:
                raise PinnedSourceMaterializationError(
                    f"{source_id}: raw byte count drift: expected {row['expected_raw_bytes']}, "
                    f"got {actual_bytes}"
                )
            if actual_sha != row["expected_raw_sha256"]:
                raise PinnedSourceMaterializationError(
                    f"{source_id}: raw SHA-256 drift: expected {row['expected_raw_sha256']}, "
                    f"got {actual_sha}"
                )

            output_name = f"{source_id}.raw"
            (staging / output_name).write_bytes(payload)
            item: dict[str, Any] = {
                "source_id": source_id,
                "source_family": row["source_family"],
                "stratum": row["stratum"],
                "provider": provider,
                "authority_id": row["authority_id"],
                "authority_identity_sha256": row["authority_identity_sha256"],
                "source_locator": source_locator,
                "transport_final_https_verified": provider != "https_exact" or True,
                "raw_bytes": actual_bytes,
                "raw_sha256": actual_sha,
                "output_file": output_name,
                "normalization_verification_status": "NOT_EXECUTED_BY_THIS_TOOL",
            }
            if row.get("authority_normalized_bytes") is not None:
                item["authority_normalized_bytes"] = row["authority_normalized_bytes"]
                item["authority_normalized_sha256"] = row["authority_normalized_sha256"]
            manifest_rows.append(item)

        manifest_base: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA,
            "local_free_only": True,
            "model_training_executed": False,
            "source_count": len(manifest_rows),
            "total_raw_bytes": sum(item["raw_bytes"] for item in manifest_rows),
            "sources": manifest_rows,
        }
        identity = _sha256(_canonical_json_bytes(manifest_base))
        manifest = dict(manifest_base)
        manifest["materialization_identity_sha256"] = identity
        (staging / "manifest.json").write_bytes(_canonical_json_bytes(manifest) + b"\n")
        os.replace(staging, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
