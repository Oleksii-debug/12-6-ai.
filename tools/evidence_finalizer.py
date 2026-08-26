#!/usr/bin/env python3
"""Fail-closed experiment evidence finalization for GitHub Actions.

This tool is intentionally stdlib-only. It can retain bootstrap and focused-test
metadata even when the locked ML runtime never installs. Checkpoint validation is
delegated to the repository's existing D05 or DCP verifier in a caller-selected
Python interpreter; this module never invents a checkpoint format or validity bit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "12-6.failure-evidence-finalization.v1"
BOOTSTRAP_SCHEMA = "12-6.failure-evidence-bootstrap.v1"
PHASE_SCHEMA = "12-6.failure-evidence-phase-state.v1"
TEST_SCHEMA = "12-6.failure-evidence-focused-tests.v1"
AUTHORITY = "LOCAL_FREE_DIAGNOSTIC_EVIDENCE_NOT_CHECKPOINT_PROMOTION"

FORBIDDEN_PARTS = frozenset(
    {
        "corpus-a",
        "corpus-b",
        "raw-corpus",
        "private-corpus",
        "private_data",
        "raw_data",
        "secrets",
    }
)

SAFE_EXACT_NAMES = frozenset(
    {
        "bootstrap.json",
        "focused-tests.xml",
        "focused-test-status.json",
        "ladder-truth.json",
        "ladder-report.json",
        "corpus-manifest.json",
        "run-manifest.json",
        "phase1.json",
        "report.json",
        "report.preverify.json",
        "s2-1m-executable-preflight.json",
        "scale141-report.json",
        "scale141-phase1.json",
    }
)
SAFE_PREFIXES = (
    "machine-",
    "machine_",
    "bootstrap-",
    "focused-test-",
    "locked-environment-",
)
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9]{20,}\b"),
)
D05_MARKERS = frozenset(
    {
        "manifest.json",
        "MANIFEST.sha256",
        "weights.safetensors",
        "state.safetensors",
        "state.json",
    }
)
DCP_MARKERS = frozenset({"scale-manifest.json", "scale-manifest.sha256", "COMMITTED"})


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _self_hashed(payload: dict[str, Any], key: str = "report_sha256") -> dict[str, Any]:
    value = dict(payload)
    value[key] = _hash_json(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": stat.st_size,
        "sha256": _sha256_file(path),
    }


def _tree_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink forbidden in retained evidence: {path}")
        if path.is_file():
            rows.append(_record(path, relative_to=root))
    return rows


def _safe_relative(path: Path, root: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"evidence path escapes workspace: {path}") from exc
    if any(part in FORBIDDEN_PARTS for part in relative.parts):
        raise ValueError(f"raw/private corpus path is forbidden: {relative.as_posix()}")
    if path.is_symlink():
        raise ValueError(f"symlink evidence is forbidden: {relative.as_posix()}")
    return relative


def _scan_secrets(path: Path) -> None:
    if path.suffix.lower() not in {".json", ".jsonl", ".xml", ".md", ".csv"}:
        return
    data = path.read_bytes()
    for pattern in SECRET_PATTERNS:
        if pattern.search(data):
            raise ValueError(f"secret-like content rejected from evidence: {path}")


def _copy_verified_file(source: Path, destination: Path) -> dict[str, Any]:
    _scan_secrets(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    before = {"bytes": source.stat().st_size, "sha256": _sha256_file(source)}
    after = {"bytes": destination.stat().st_size, "sha256": _sha256_file(destination)}
    if before != after:
        raise ValueError(f"copy integrity mismatch: {source}")
    return before


def write_bootstrap(output: Path, source_sha: str, label: str) -> dict[str, Any]:
    runner_keys = (
        "GITHUB_ACTIONS",
        "GITHUB_EVENT_NAME",
        "GITHUB_JOB",
        "GITHUB_REPOSITORY",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_RUN_ID",
        "GITHUB_WORKFLOW",
        "RUNNER_ARCH",
        "RUNNER_OS",
    )
    payload = _self_hashed(
        {
            "schema": BOOTSTRAP_SCHEMA,
            "authority": AUTHORITY,
            "created_at_utc": _utc_now(),
            "source_sha": source_sha,
            "label": label,
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable_name": Path(sys.executable).name,
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "runner": {key: os.environ[key] for key in runner_keys if key in os.environ},
            "privacy": {
                "environment_allowlist_only": True,
                "secret_values_recorded": False,
                "raw_corpus_text_recorded": False,
            },
        }
    )
    _write_json(output, payload)
    return payload


def mark_phase(state_path: Path, source_sha: str, phase: str) -> dict[str, Any]:
    history: list[dict[str, str]] = []
    if state_path.exists():
        prior = json.loads(state_path.read_text(encoding="utf-8"))
        if prior.get("source_sha") != source_sha:
            raise ValueError("phase-state source SHA changed")
        prior_history = prior.get("history", [])
        if isinstance(prior_history, list):
            history = [dict(item) for item in prior_history if isinstance(item, dict)]
    history.append({"phase": phase, "entered_at_utc": _utc_now()})
    payload = _self_hashed(
        {
            "schema": PHASE_SCHEMA,
            "source_sha": source_sha,
            "current_phase": phase,
            "history": history,
        },
        key="state_sha256",
    )
    _write_json(state_path, payload)
    return payload


def write_test_status(output: Path, source_sha: str, exit_code: int, label: str) -> dict[str, Any]:
    payload = _self_hashed(
        {
            "schema": TEST_SCHEMA,
            "authority": AUTHORITY,
            "source_sha": source_sha,
            "label": label,
            "exit_code": int(exit_code),
            "status": "PASS" if int(exit_code) == 0 else "FAIL",
            "recorded_at_utc": _utc_now(),
        }
    )
    _write_json(output, payload)
    return payload


def _checkpoint_kind(path: Path) -> str | None:
    if not path.is_dir() or path.is_symlink():
        return None
    names = {item.name for item in path.iterdir()}
    if D05_MARKERS.issubset(names):
        return "d05"
    if DCP_MARKERS.issubset(names):
        return "dcp"
    return None


def _verify_checkpoint(
    path: Path, *, kind: str, verifier_python: Path, repo_root: Path
) -> dict[str, Any]:
    if not verifier_python.is_file():
        raise ValueError(f"locked verifier interpreter unavailable: {verifier_python}")
    if kind == "d05":
        code = (
            "import json,sys; "
            "from twelve_six.checkpoint import verify_checkpoint; "
            "m=verify_checkpoint(sys.argv[1]); "
            "print(json.dumps({'kind':'d05','identity':m['checkpoint_id']},sort_keys=True))"
        )
    elif kind == "dcp":
        code = (
            "import json,sys; "
            "from twelve_six.distributed.dcp_checkpoint import verify_scale_checkpoint; "
            "m=verify_scale_checkpoint(sys.argv[1]); "
            "print(json.dumps({'kind':'dcp','identity':m['aggregate_checkpoint_sha256']},sort_keys=True))"
        )
    else:
        raise ValueError(f"unsupported checkpoint verifier kind: {kind}")
    env = dict(os.environ)
    src = str((repo_root / "src").resolve())
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(
        [str(verifier_python), "-c", code, str(path.resolve())],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown verifier error"
        raise ValueError(f"{kind} verifier rejected checkpoint: {detail}")
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ValueError(f"{kind} verifier emitted invalid result") from exc
    if payload.get("kind") != kind or not isinstance(payload.get("identity"), str):
        raise ValueError(f"{kind} verifier result missing identity")
    return payload


def _metadata_candidate(path: Path) -> bool:
    name = path.name
    if name in SAFE_EXACT_NAMES:
        return True
    return name.endswith(".json") and name.startswith(SAFE_PREFIXES)


def _discover_checkpoint_dirs(workspace: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in workspace.rglob("*"):
        if not path.is_dir() or path.is_symlink():
            continue
        if path.name == "checkpoint" or path.name.startswith("checkpoint-"):
            candidates.append(path)
    candidates.sort(key=lambda item: item.as_posix())
    return candidates


def finalize_workspace(
    *,
    workspace: Path,
    artifact_dir: Path,
    phase_file: Path,
    source_sha: str,
    verifier_python: Path,
    repo_root: Path,
    job_status: str,
    retention_days: int,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    repo_root = repo_root.resolve()
    artifact_dir = artifact_dir.resolve()
    if not workspace.is_dir():
        workspace.mkdir(parents=True, exist_ok=True)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    payload_root = artifact_dir / "payload"
    payload_root.mkdir(parents=True, exist_ok=True)

    phase = "UNKNOWN_BEFORE_PHASE_TRACKING"
    phase_history: list[dict[str, Any]] = []
    if phase_file.exists():
        state = json.loads(phase_file.read_text(encoding="utf-8"))
        if state.get("source_sha") != source_sha:
            raise ValueError("phase-state source SHA mismatch during finalization")
        phase = str(state.get("current_phase", phase))
        if isinstance(state.get("history"), list):
            phase_history = list(state["history"])

    retained_metadata: list[dict[str, Any]] = []
    rejected_metadata: list[dict[str, Any]] = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or path.is_symlink() or not _metadata_candidate(path):
            continue
        try:
            relative = _safe_relative(path, workspace)
            details = _copy_verified_file(path, payload_root / relative)
            retained_metadata.append({"path": relative.as_posix(), **details})
        except Exception as exc:  # diagnostic rejection must not hide other evidence
            rejected_metadata.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})

    checkpoint_rows: list[dict[str, Any]] = []
    valid_checkpoint_parents: set[Path] = set()
    for checkpoint in _discover_checkpoint_dirs(workspace):
        relative = _safe_relative(checkpoint, workspace)
        kind = _checkpoint_kind(checkpoint)
        row: dict[str, Any] = {
            "path": relative.as_posix(),
            "kind": kind or "unknown_or_incomplete",
            "valid": False,
            "retained": False,
        }
        if kind is None:
            row["reason"] = "missing exact D05 inventory or DCP committed control plane"
            checkpoint_rows.append(row)
            continue
        try:
            source_verified = _verify_checkpoint(
                checkpoint,
                kind=kind,
                verifier_python=verifier_python,
                repo_root=repo_root,
            )
            source_inventory = _tree_inventory(checkpoint)
            destination = payload_root / relative
            shutil.copytree(checkpoint, destination)
            staged_verified = _verify_checkpoint(
                destination,
                kind=kind,
                verifier_python=verifier_python,
                repo_root=repo_root,
            )
            staged_inventory = _tree_inventory(destination)
            if source_verified != staged_verified:
                raise ValueError("checkpoint identity changed across artifact staging")
            if source_inventory != staged_inventory:
                raise ValueError("checkpoint hash/size inventory changed across artifact staging")
            row.update(
                {
                    "valid": True,
                    "retained": True,
                    "identity": source_verified["identity"],
                    "file_count": len(source_inventory),
                    "bytes": sum(int(item["bytes"]) for item in source_inventory),
                    "inventory_sha256": _hash_json(source_inventory),
                }
            )
            valid_checkpoint_parents.add(checkpoint.parent.resolve())
        except Exception as exc:
            destination = payload_root / relative
            if destination.exists():
                shutil.rmtree(destination, ignore_errors=True)
            row["reason"] = f"{type(exc).__name__}: {exc}"
        checkpoint_rows.append(row)

    training_rows: list[dict[str, Any]] = []
    for curve in sorted(workspace.rglob("train-curve.jsonl")):
        relative = _safe_relative(curve, workspace)
        parent = curve.parent.resolve()
        row: dict[str, Any] = {"path": relative.as_posix(), "retained": False}
        if parent not in valid_checkpoint_parents:
            row["reason"] = "no D05/DCP-verified committed checkpoint in the same run directory"
            training_rows.append(row)
            continue
        try:
            details = _copy_verified_file(curve, payload_root / relative)
            row.update({"retained": True, **details})
        except Exception as exc:
            row["reason"] = f"{type(exc).__name__}: {exc}"
        training_rows.append(row)

    artifact_inventory = _tree_inventory(payload_root)
    valid_count = sum(1 for row in checkpoint_rows if row["valid"] and row["retained"])
    report = _self_hashed(
        {
            "schema": SCHEMA,
            "authority": AUTHORITY,
            "source_sha": source_sha,
            "created_at_utc": _utc_now(),
            "job_status": job_status,
            "termination_phase": phase,
            "phase_history": phase_history,
            "retention_days": int(retention_days),
            "checkpoint_contract": {
                "formats_invented": False,
                "d05_verifier": "twelve_six.checkpoint.verify_checkpoint",
                "dcp_verifier": "twelve_six.distributed.dcp_checkpoint.verify_scale_checkpoint",
                "valid_checkpoint_count": valid_count,
                "checkpoints": checkpoint_rows,
            },
            "training_evidence": training_rows,
            "metadata": {
                "retained": retained_metadata,
                "rejected": rejected_metadata,
            },
            "privacy": {
                "allowlisted_metadata_only": True,
                "raw_private_corpus_paths_forbidden": sorted(FORBIDDEN_PARTS),
                "secret_like_text_scan": True,
                "raw_corpus_text_uploaded": False,
                "secrets_uploaded": False,
            },
            "artifact": {
                "payload_file_count": len(artifact_inventory),
                "payload_bytes": sum(int(item["bytes"]) for item in artifact_inventory),
                "payload_inventory_sha256": _hash_json(artifact_inventory),
                "files": artifact_inventory,
            },
            "interpretation": (
                "COMPLETED_EVIDENCE_BUNDLE"
                if job_status == "success"
                else "FAILURE_DIAGNOSTICS_ONLY_NO_COMPLETION_CLAIM"
            ),
        }
    )
    _write_json(artifact_dir / "finalization-report.json", report)
    report_hash = _sha256_file(artifact_dir / "finalization-report.json")
    report_size = (artifact_dir / "finalization-report.json").stat().st_size
    (artifact_dir / "finalization-report.sha256").write_text(
        f"{report_hash}  finalization-report.json\n", encoding="ascii"
    )
    # Read back both report artifacts before returning success.
    if _sha256_file(artifact_dir / "finalization-report.json") != report_hash:
        raise ValueError("finalization report changed after write")
    if (artifact_dir / "finalization-report.json").stat().st_size != report_size:
        raise ValueError("finalization report size changed after write")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bootstrap = sub.add_parser("bootstrap")
    bootstrap.add_argument("--output", type=Path, required=True)
    bootstrap.add_argument("--source-sha", required=True)
    bootstrap.add_argument("--label", required=True)

    phase = sub.add_parser("mark-phase")
    phase.add_argument("--state", type=Path, required=True)
    phase.add_argument("--source-sha", required=True)
    phase.add_argument("--phase", required=True)

    tests = sub.add_parser("test-status")
    tests.add_argument("--output", type=Path, required=True)
    tests.add_argument("--source-sha", required=True)
    tests.add_argument("--exit-code", type=int, required=True)
    tests.add_argument("--label", required=True)

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--workspace", type=Path, required=True)
    finalize.add_argument("--artifact-dir", type=Path, required=True)
    finalize.add_argument("--phase-file", type=Path, required=True)
    finalize.add_argument("--source-sha", required=True)
    finalize.add_argument("--verifier-python", type=Path, required=True)
    finalize.add_argument("--repo-root", type=Path, default=Path("."))
    finalize.add_argument("--job-status", required=True)
    finalize.add_argument("--retention-days", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "bootstrap":
        write_bootstrap(args.output, args.source_sha, args.label)
        return 0
    if args.command == "mark-phase":
        mark_phase(args.state, args.source_sha, args.phase)
        return 0
    if args.command == "test-status":
        write_test_status(args.output, args.source_sha, args.exit_code, args.label)
        return 0
    if args.command == "finalize":
        finalize_workspace(
            workspace=args.workspace,
            artifact_dir=args.artifact_dir,
            phase_file=args.phase_file,
            source_sha=args.source_sha,
            verifier_python=args.verifier_python,
            repo_root=args.repo_root,
            job_status=args.job_status,
            retention_days=args.retention_days,
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
