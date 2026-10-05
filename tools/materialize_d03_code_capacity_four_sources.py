#!/usr/bin/env python3
"""Materialize four bounded code-source authorities into one zero-credit ephemeral candidate."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SRC = ROOT / "src"
for location in (str(TOOLS), str(SRC)):
    if location not in sys.path:
        sys.path.insert(0, location)

import materialize_data_bulk_code1_permissive_python_bundle as code1
import qualify_next100_048_pydantic as pydantic_qual
from twelve_six import pandas_source_authority, typer_source_authority
from twelve_six.data.permissive_repo_source_authority import (
    load_and_validate_source_authority,
)

SCHEMA = "12-6.d03-code-capacity-four-source-materialization.v1"
RECEIPT_SCHEMA = "12-6.d03-code-capacity-four-source-materialization-receipt.v1"
PYDANTIC_CONFIG = Path("configs/data/next100_048_pydantic_code_rights_v1.json")
SCIPY_CONFIG = Path("configs/data/scipy_v118_source_authority_v1.json")
PANDAS_CONFIG = Path("configs/data/next100_050_pandas_source_authority_v2.json")
TYPER_CONFIG = Path("configs/data/next100_052_typer_source_authority_v2.json")
EXPECTED_OBJECTS = 8
EXPECTED_RAW_BYTES = 336_947
MAX_JSON_BYTES = 1_048_576
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 20_000
MAX_INT_DIGITS = 64

Fetch = Callable[[str, int], bytes]
_REAL_PYDANTIC_DOWNLOAD = pydantic_qual.download


class FourSourceMaterializationError(RuntimeError):
    """Fail-closed error for source-authority composition or physical materialization."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FourSourceMaterializationError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FourSourceMaterializationError(f"duplicate JSON object member: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise FourSourceMaterializationError(f"non-finite JSON constant: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise FourSourceMaterializationError("JSON float is not finite")
    significand = value.split("e", 1)[0].split("E", 1)[0]
    if parsed == 0.0 and any(ch in "123456789" for ch in significand):
        raise FourSourceMaterializationError("nonzero JSON float underflowed to zero")
    return parsed


def _bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > MAX_INT_DIGITS:
        raise FourSourceMaterializationError("JSON integer exceeds digit limit")
    return int(value)


def _strict_object(raw: bytes, *, context: str) -> dict[str, Any]:
    _require(len(raw) <= MAX_JSON_BYTES, f"{context} exceeds byte limit")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_pairs,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
            parse_int=_bounded_int,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise FourSourceMaterializationError(f"{context} is invalid JSON") from exc
    _require(type(value) is dict, f"{context} root must be object")
    pending: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        _require(nodes <= MAX_JSON_NODES, f"{context} exceeds node limit")
        _require(depth <= MAX_JSON_DEPTH, f"{context} exceeds depth limit")
        if isinstance(current, dict):
            for key, child in current.items():
                key.encode("utf-8")
                pending.append((child, depth + 1))
        elif isinstance(current, list):
            pending.extend((child, depth + 1) for child in current)
        elif isinstance(current, str):
            current.encode("utf-8")
    return value


def _read_strict(path: Path, *, context: str) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_JSON_BYTES + 1)
    except OSError as exc:
        raise FourSourceMaterializationError(f"cannot read {context}") from exc
    return _strict_object(raw, context=context)


def _git_head(repo_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FourSourceMaterializationError("cannot inspect execution Git HEAD") from exc
    _require(proc.returncode == 0, "cannot inspect execution Git HEAD")
    head = proc.stdout.strip()
    _require(
        len(head) == 40 and all(ch in "0123456789abcdef" for ch in head),
        "execution Git HEAD is not exact lowercase 40-hex",
    )
    return head


def _fetch(url: str, max_bytes: int) -> bytes:
    try:
        return _REAL_PYDANTIC_DOWNLOAD(url, max_bytes=max_bytes)
    except Exception as exc:
        raise FourSourceMaterializationError("bounded source download failed") from exc


def _verify_code_payload(
    raw: bytes,
    *,
    path: str,
    expected_bytes: int,
    expected_blob: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    _require(type(expected_bytes) is int and expected_bytes > 0, "expected byte count invalid")
    _require(len(raw) == expected_bytes, f"source byte count drift: {path}")
    _require(_git_blob_sha1(raw) == expected_blob, f"source Git blob drift: {path}")
    digest = _sha256(raw)
    if expected_sha256 is not None:
        _require(digest == expected_sha256, f"source SHA-256 drift: {path}")
    _require(b"\x00" not in raw, f"NUL byte in source: {path}")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise FourSourceMaterializationError(f"source is not strict UTF-8: {path}") from exc
    _require("\ufffd" not in text, f"replacement character in source: {path}")
    _require(
        not any(0xD800 <= ord(ch) <= 0xDFFF for ch in text),
        f"surrogate code point in source: {path}",
    )
    hits = code1._credential_hits(raw)
    _require(not hits, f"credential pattern in source: {path}")
    marker = code1._generated_marker(raw)
    _require(marker is None, f"generated-material marker in source: {path}")
    try:
        ast.parse(text, filename=path)
    except SyntaxError as exc:
        raise FourSourceMaterializationError(f"Python AST parse failed: {path}") from exc
    return {"text": text, "raw_sha256": digest}


def _verify_license(
    raw: bytes,
    *,
    path: str,
    expected_blob: str | None,
    expected_sha256: str | None,
    license_id: str,
) -> dict[str, Any]:
    if expected_blob is not None:
        _require(_git_blob_sha1(raw) == expected_blob, f"license Git blob drift: {path}")
    digest = _sha256(raw)
    if expected_sha256 is not None:
        _require(digest == expected_sha256, f"license SHA-256 drift: {path}")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise FourSourceMaterializationError(f"license is not strict UTF-8: {path}") from exc
    normalized = " ".join(text.split())
    if license_id == "MIT":
        _require(
            "Permission is hereby granted" in normalized
            or "deal in the Software without restriction" in normalized,
            f"MIT grant missing: {path}",
        )
    elif license_id == "BSD-3-Clause":
        _require(
            "Redistribution and use in source and binary forms" in normalized,
            f"BSD grant missing: {path}",
        )
    else:
        raise FourSourceMaterializationError(f"unsupported license id: {license_id}")
    return {
        "license_id": license_id,
        "path": path,
        "git_blob_sha1": _git_blob_sha1(raw),
        "sha256": digest,
        "utf8_bytes": len(raw),
    }


def _record(
    *,
    authority_source_id: str | None,
    source_family: str,
    repository: str,
    commit: str,
    path: str,
    blob: str,
    raw: bytes,
    expected_bytes: int,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    verified = _verify_code_payload(
        raw,
        path=path,
        expected_bytes=expected_bytes,
        expected_blob=blob,
        expected_sha256=expected_sha256,
    )
    object_id = f"{source_family}@{commit}:{path}"
    return {
        "schema_version": SCHEMA,
        "object_id": object_id,
        "authority_source_id": authority_source_id,
        "source_family": source_family,
        "repository": repository,
        "commit": commit,
        "path": path,
        "git_blob_sha1": blob,
        "raw_sha256": verified["raw_sha256"],
        "utf8_bytes": len(raw),
        "modality": "code",
        "normalization": "STRICT_UTF8_IDENTITY_PRESERVE_V1",
        "text": verified["text"],
    }


def _capture_pydantic(
    repo_root: Path,
    *,
    execution_head: str,
    fetch: Fetch,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy_path = repo_root / PYDANTIC_CONFIG
    policy = _read_strict(policy_path, context="Pydantic source authority")
    decisions = policy.get("decisions")
    _require(type(decisions) is list and len(decisions) == 4, "Pydantic decision set drift")
    selected_urls = {row.get("raw_url") for row in decisions if type(row) is dict}
    _require(len(selected_urls) == 4 and None not in selected_urls, "Pydantic raw URL set drift")
    captured: dict[str, bytes] = {}
    original = pydantic_qual.download

    def capture(url: str, max_bytes: int = 300_000) -> bytes:
        raw = fetch(url, max_bytes)
        if url in selected_urls:
            captured[url] = raw
        return raw

    pydantic_qual.download = capture
    try:
        evidence = pydantic_qual.qualify(
            repo_root=repo_root,
            policy_path=PYDANTIC_CONFIG,
            source_sha=execution_head,
        )
    except Exception as exc:
        raise FourSourceMaterializationError("Pydantic live qualification failed") from exc
    finally:
        pydantic_qual.download = original

    _require(evidence.get("status") == "ADMIT", "Pydantic live qualification did not admit")
    _require(
        evidence.get("source_family_accounting", {}).get("selected_authored_capacity_bytes")
        == 235_204,
        "Pydantic qualified capacity drift",
    )
    objects_by_id = {
        row["source_id"]: row
        for row in evidence.get("objects", [])
        if type(row) is dict and type(row.get("source_id")) is str
    }
    result: list[dict[str, Any]] = []
    for decision in decisions:
        _require(type(decision) is dict, "Pydantic decision must be object")
        raw_url = decision["raw_url"]
        _require(raw_url in captured, "Pydantic qualifier did not materialize selected object")
        raw = captured[raw_url]
        evidence_row = objects_by_id.get(decision["source_id"])
        _require(type(evidence_row) is dict, "Pydantic evidence object missing")
        _require(
            evidence_row.get("raw_sha256") == _sha256(raw),
            "Pydantic captured payload differs from qualified evidence",
        )
        result.append(
            _record(
                authority_source_id=decision["source_id"],
                source_family=decision["source_family"],
                repository="pydantic/pydantic",
                commit=decision["commit"],
                path=decision["path"],
                blob=decision["blob_sha1"],
                raw=raw,
                expected_bytes=decision["size_bytes"],
                expected_sha256=evidence_row["raw_sha256"],
            )
        )
    return result, evidence["license"]


def _materialize_scipy(
    repo_root: Path,
    *,
    fetch: Fetch,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = load_and_validate_source_authority(repo_root / SCIPY_CONFIG)
    _require(summary["candidate_raw_bytes"] == 78_307, "SciPy authority capacity drift")
    config = _read_strict(repo_root / SCIPY_CONFIG, context="SciPy source authority")
    upstream = config["upstream"]
    commit = upstream["commit_sha"]
    rows: list[dict[str, Any]] = []
    for entry in config["allowlist"]:
        raw = fetch(entry["raw_url"], entry["raw_bytes"])
        rows.append(
            _record(
                authority_source_id=None,
                source_family=config["source_family"],
                repository=upstream["repository"],
                commit=commit,
                path=entry["path"],
                blob=entry["git_blob_sha1"],
                raw=raw,
                expected_bytes=entry["raw_bytes"],
            )
        )
    base = f"https://raw.githubusercontent.com/{upstream['repository']}/{commit}/"
    root_raw = fetch(base + config["license"]["root_path"], 100_000)
    root_license = _verify_license(
        root_raw,
        path=config["license"]["root_path"],
        expected_blob=None,
        expected_sha256=None,
        license_id=config["license"]["root_spdx"],
    )
    bundled_raw = fetch(base + config["license"]["bundled_license_path"], 300_000)
    _require(bool(bundled_raw), "SciPy bundled-license file is empty")
    try:
        bundled_raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise FourSourceMaterializationError("SciPy bundled-license file is not UTF-8") from exc
    return rows, {
        **root_license,
        "bundled_license_path": config["license"]["bundled_license_path"],
        "bundled_license_git_blob_sha1": _git_blob_sha1(bundled_raw),
        "bundled_license_sha256": _sha256(bundled_raw),
        "whole_repository_credit_forbidden": True,
    }


def _materialize_pandas(
    repo_root: Path,
    *,
    fetch: Fetch,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    errors = pandas_source_authority.validate_pandas_source_authority_files(repo_root)
    _require(errors == [], "Pandas source authority invalid: " + ",".join(errors[:5]))
    config = _read_strict(repo_root / PANDAS_CONFIG, context="Pandas source authority")
    source = config["bounded_source"]
    raw_url = (
        f"https://raw.githubusercontent.com/{source['repository']}/{source['commit']}/"
        f"{source['path']}"
    )
    raw = fetch(raw_url, source["size_bytes"])
    row = _record(
        authority_source_id=source["source_id"],
        source_family=source["source_family"],
        repository=source["repository"],
        commit=source["commit"],
        path=source["path"],
        blob=source["git_blob_sha1"],
        raw=raw,
        expected_bytes=source["size_bytes"],
        expected_sha256=source["raw_sha256"],
    )
    license_info = config["license"]
    license_url = (
        f"https://raw.githubusercontent.com/{source['repository']}/{source['commit']}/"
        f"{license_info['path']}"
    )
    license_raw = fetch(license_url, 100_000)
    license_receipt = _verify_license(
        license_raw,
        path=license_info["path"],
        expected_blob=license_info["git_blob_sha1"],
        expected_sha256=license_info["sha256"],
        license_id=license_info["license_id"],
    )
    return [row], license_receipt


def _materialize_typer(
    repo_root: Path,
    *,
    fetch: Fetch,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    errors = typer_source_authority.validate_typer_source_authority_files(repo_root)
    _require(errors == [], "Typer source authority invalid: " + ",".join(errors[:5]))
    config = _read_strict(repo_root / TYPER_CONFIG, context="Typer source authority")
    source = config["bounded_source"]
    raw_url = (
        f"https://raw.githubusercontent.com/{source['repository']}/{source['commit']}/"
        f"{source['path']}"
    )
    raw = fetch(raw_url, source["size_bytes"])
    row = _record(
        authority_source_id=source["source_id"],
        source_family=source["source_family"],
        repository=source["repository"],
        commit=source["commit"],
        path=source["path"],
        blob=source["git_blob_sha1"],
        raw=raw,
        expected_bytes=source["size_bytes"],
        expected_sha256=source["raw_sha256"],
    )
    license_info = config["license"]
    license_url = (
        f"https://raw.githubusercontent.com/{source['repository']}/{source['commit']}/"
        f"{license_info['path']}"
    )
    license_raw = fetch(license_url, license_info["size_bytes"])
    license_receipt = _verify_license(
        license_raw,
        path=license_info["path"],
        expected_blob=license_info["git_blob_sha1"],
        expected_sha256=license_info["sha256"],
        license_id=license_info["license_id"],
    )
    return [row], license_receipt


def _render_candidate(records: list[dict[str, Any]]) -> bytes:
    ordered = sorted(records, key=lambda row: row["object_id"])
    _require(len(ordered) == EXPECTED_OBJECTS, "materialized object count drift")
    object_ids = [row["object_id"] for row in ordered]
    _require(len(set(object_ids)) == len(object_ids), "duplicate materialized object id")
    raw_hashes = [row["raw_sha256"] for row in ordered]
    _require(len(set(raw_hashes)) == len(raw_hashes), "exact duplicate materialized source bytes")
    total = sum(row["utf8_bytes"] for row in ordered)
    _require(total == EXPECTED_RAW_BYTES, "materialized raw byte total drift")
    return b"".join(_canonical(row) + b"\n" for row in ordered)


def _receipt(
    records: list[dict[str, Any]],
    licenses: dict[str, dict[str, Any]],
    *,
    candidate: bytes,
    execution_head: str,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: row["object_id"])
    family_rows: list[dict[str, Any]] = []
    for family in sorted({row["source_family"] for row in ordered}):
        selected = [row for row in ordered if row["source_family"] == family]
        family_rows.append(
            {
                "source_family": family,
                "object_count": len(selected),
                "raw_bytes": sum(row["utf8_bytes"] for row in selected),
                "object_identity_sha256": _sha256(
                    _canonical(
                        [
                            {
                                "object_id": row["object_id"],
                                "raw_sha256": row["raw_sha256"],
                                "git_blob_sha1": row["git_blob_sha1"],
                                "utf8_bytes": row["utf8_bytes"],
                            }
                            for row in selected
                        ]
                    )
                ),
            }
        )
    core: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": execution_head,
        "materialized_object_count": len(ordered),
        "materialized_raw_bytes": sum(row["utf8_bytes"] for row in ordered),
        "candidate_sha256": _sha256(candidate),
        "families": family_rows,
        "licenses": licenses,
        "content_boundary": {
            "candidate_contains_raw_source_text": True,
            "candidate_must_remain_ephemeral_until_downstream_authority": True,
            "receipt_contains_raw_source_text": False,
        },
        "truth_boundary": {
            "canonical_capacity_credit_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
        },
        "required_downstream_gates": [
            "INCUMBENT_WHOLE_GRAPH_GLOBAL_CROSS_SOURCE_DEDUP",
            "FRESH_RESERVED_EVALUATION_DECONTAMINATION",
            "POST_COMPOSITION_QUALITY_PRIVACY",
            "WHOLE_FAMILY_BALANCE_AND_CAPS",
            "CLUSTER_SAFE_SPLIT",
            "DETERMINISTIC_PACK_AND_TWO_CLEAN_BUILDS",
            "POSITIVE_EXACT_POSTPACK_UNIQUE_LOSS_LEDGER",
        ],
    }
    identity = _sha256(_canonical(core))
    return {**core, "receipt_identity_sha256": identity}


def materialize(
    repo_root: Path,
    *,
    execution_head: str,
    fetch: Fetch = _fetch,
) -> tuple[bytes, dict[str, Any]]:
    _require(repo_root.is_dir(), "repository root missing")
    pydantic_rows, pydantic_license = _capture_pydantic(
        repo_root, execution_head=execution_head, fetch=fetch
    )
    scipy_rows, scipy_license = _materialize_scipy(repo_root, fetch=fetch)
    pandas_rows, pandas_license = _materialize_pandas(repo_root, fetch=fetch)
    typer_rows, typer_license = _materialize_typer(repo_root, fetch=fetch)
    records = pydantic_rows + scipy_rows + pandas_rows + typer_rows
    candidate = _render_candidate(records)
    receipt = _receipt(
        records,
        {
            "github:pydantic/pydantic": pydantic_license,
            "code.scipy.project": scipy_license,
            "github:pandas-dev/pandas": pandas_license,
            "github:fastapi/typer": typer_license,
        },
        candidate=candidate,
        execution_head=execution_head,
    )
    _require(receipt["materialized_object_count"] == EXPECTED_OBJECTS, "receipt object count drift")
    _require(receipt["materialized_raw_bytes"] == EXPECTED_RAW_BYTES, "receipt byte total drift")
    return candidate, receipt


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
    except FileExistsError as exc:
        raise FourSourceMaterializationError(f"refusing to overwrite output: {path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--expected-execution-head", required=True)
    parser.add_argument("--candidate-out", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    observed_head = _git_head(repo_root)
    if observed_head != args.expected_execution_head:
        parser.error("execution HEAD differs from explicitly selected exact head")
    if args.candidate_out.resolve(strict=False) == args.receipt_out.resolve(strict=False):
        parser.error("candidate and receipt outputs must be distinct")

    try:
        candidate, receipt = materialize(
            repo_root,
            execution_head=observed_head,
        )
        _write_create_only(args.candidate_out, candidate)
        _write_create_only(args.receipt_out, _canonical(receipt) + b"\n")
    except (OSError, ValueError, FourSourceMaterializationError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "status": "MATERIALIZED_ZERO_CREDIT",
                "objects": receipt["materialized_object_count"],
                "raw_bytes": receipt["materialized_raw_bytes"],
                "candidate_sha256": receipt["candidate_sha256"],
                "receipt_identity_sha256": receipt["receipt_identity_sha256"],
                "canonical_capacity_credit_bytes": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
