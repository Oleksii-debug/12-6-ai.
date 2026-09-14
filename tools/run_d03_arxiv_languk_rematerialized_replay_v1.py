#!/usr/bin/env python3
"""Rematerialize exact ArXiv+LangUK candidates twice and replay audited PR #1800 V9."""
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from twelve_six.data.arxiv_languk_rematerialized_replay_v1 import (
    ARXIV,
    LANGUK,
    PARENT_INTAKE_BLOB_SHA1,
    PARENT_RUNNER_BLOB_SHA1,
    HistoricalMaterializerSpec,
    RematerializationError,
    build_receipt,
    canonical_json_bytes,
    git_blob_sha1,
    sha256_bytes,
    verify_candidate,
    verify_git_blob,
)

INCUMBENT_RUNNER = ROOT / "tools/run_d03_arxiv_languk_postadmission_global_dedup_v2.py"
INCUMBENT_INTAKE = ROOT / "src/twelve_six/data/post_admission_dedup_intake_v2.py"
EXPECTED_PYARROW = "17.0.0"


def _run(
    command: list[str],
    *,
    cwd: Path,
    capture: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=capture,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        rendered = " ".join(command)
        raise RematerializationError(f"command failed: {rendered}") from exc


def _git(repo_root: Path, *args: str, capture: bool = True) -> bytes:
    result = _run(["git", "-C", str(repo_root), *args], cwd=repo_root, capture=capture)
    return result.stdout if capture else b""


def _ensure_commit(repo_root: Path, commit: str) -> None:
    try:
        _git(repo_root, "cat-file", "-e", f"{commit}^{{commit}}")
        return
    except RematerializationError:
        pass
    _git(
        repo_root,
        "fetch",
        "--no-tags",
        "--depth=1",
        "origin",
        commit,
        capture=False,
    )
    _git(repo_root, "cat-file", "-e", f"{commit}^{{commit}}")


def _verify_historical_program(repo_root: Path, spec: HistoricalMaterializerSpec) -> None:
    _ensure_commit(repo_root, spec.execution_commit)
    tool_raw = _git(repo_root, "show", f"{spec.execution_commit}:{spec.tool_path}")
    config_raw = _git(repo_root, "show", f"{spec.execution_commit}:{spec.config_path}")
    verify_git_blob(tool_raw, spec.tool_blob_sha1, label=f"{spec.key} historical tool")
    verify_git_blob(config_raw, spec.config_blob_sha1, label=f"{spec.key} historical config")


def _extract_commit(repo_root: Path, commit: str, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    archive_raw = _git(repo_root, "archive", "--format=tar", commit)
    root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:") as archive:
        members = archive.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root):
                raise RematerializationError("historical archive escaped workspace")
            if member.issym() or member.islnk():
                raise RematerializationError("historical archive links are forbidden")
        archive.extractall(root, members=members)


def _require_pyarrow() -> None:
    try:
        actual = metadata.version("pyarrow")
    except metadata.PackageNotFoundError as exc:
        raise RematerializationError(
            f"LangUK rematerialization requires pyarrow=={EXPECTED_PYARROW}"
        ) from exc
    if actual != EXPECTED_PYARROW:
        raise RematerializationError(
            f"LangUK rematerialization requires pyarrow=={EXPECTED_PYARROW}; got {actual}"
        )


def _verify_incumbent_checkout() -> None:
    verify_git_blob(
        INCUMBENT_RUNNER.read_bytes(),
        PARENT_RUNNER_BLOB_SHA1,
        label="PR1800 incumbent runner",
    )
    verify_git_blob(
        INCUMBENT_INTAKE.read_bytes(),
        PARENT_INTAKE_BLOB_SHA1,
        label="PR1800 incumbent intake",
    )


def _run_historical_materializer(
    *,
    spec: HistoricalMaterializerSpec,
    historical_root: Path,
    pass_root: Path,
) -> Path:
    source = pass_root / spec.source_filename
    candidate = pass_root / f"{spec.key}-candidate.jsonl"
    report = pass_root / f"{spec.key}-materialization-report.json"
    pass_root.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(historical_root / spec.tool_path),
    ]
    if spec.key == "arxiv":
        command.extend(["--config", str(historical_root / spec.config_path)])
    command.extend(
        [
            spec.source_arg,
            str(source),
            "--output-jsonl",
            str(candidate),
            "--report",
            str(report),
            "--download",
        ]
    )
    _run(command, cwd=historical_root)
    verify_candidate(spec, candidate.read_bytes())
    return candidate


def _replay_command(
    args: argparse.Namespace,
    *,
    arxiv_candidate: Path,
    languk_candidate: Path,
    report: Path,
    survivors: Path,
) -> list[str]:
    return [
        sys.executable,
        str(INCUMBENT_RUNNER),
        "--v7-root",
        str(args.v7_root),
        "--bulk-workspace",
        str(args.bulk_workspace),
        "--v8-config",
        str(args.v8_config),
        "--data526-config",
        str(args.data526_config),
        "--v8-report",
        str(args.v8_report),
        "--v8-survivors",
        str(args.v8_survivors),
        "--data526-evidence",
        str(args.data526_evidence),
        "--data526-record-inventory",
        str(args.data526_record_inventory),
        "--rada-language-report",
        str(args.rada_language_report),
        "--rada-quality-privacy-jsonl",
        str(args.rada_quality_privacy_jsonl),
        "--rada-quality-privacy-report",
        str(args.rada_quality_privacy_report),
        "--expected-rada-report-sha256",
        args.expected_rada_report_sha256,
        "--arxiv-authority",
        str(args.arxiv_authority),
        "--arxiv-candidate",
        str(arxiv_candidate),
        "--languk-authority",
        str(args.languk_authority),
        "--languk-candidate",
        str(languk_candidate),
        "--output-report",
        str(report),
        "--output-survivors",
        str(survivors),
    ]


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RematerializationError(f"cannot read {label}") from exc
    if type(value) is not dict:
        raise RematerializationError(f"{label} must be a JSON object")
    return value


def _pass_result(
    *,
    arxiv_candidate: Path,
    languk_candidate: Path,
    report: Path,
    survivors: Path,
) -> dict[str, str]:
    arxiv_raw = arxiv_candidate.read_bytes()
    languk_raw = languk_candidate.read_bytes()
    verify_candidate(ARXIV, arxiv_raw)
    verify_candidate(LANGUK, languk_raw)

    report_raw = report.read_bytes()
    survivor_raw = survivors.read_bytes()
    report_value = _load_json_object(report, label="replay report")
    survivor_value = _load_json_object(survivors, label="survivor authority")

    report_identity = report_value.get("report_sha256")
    survivor_identity = survivor_value.get("survivor_authority_sha256")
    if not isinstance(report_identity, str):
        raise RematerializationError("replay report identity missing")
    if not isinstance(survivor_identity, str):
        raise RematerializationError("survivor authority identity missing")

    return {
        "arxiv_candidate_sha256": sha256_bytes(arxiv_raw),
        "languk_candidate_sha256": sha256_bytes(languk_raw),
        "report_file_sha256": sha256_bytes(report_raw),
        "survivor_file_sha256": sha256_bytes(survivor_raw),
        "report_identity_sha256": report_identity,
        "survivor_authority_sha256": survivor_identity,
    }


def _remove_ephemeral_payloads(pass_root: Path) -> None:
    for spec in (ARXIV, LANGUK):
        for path in (
            pass_root / spec.source_filename,
            pass_root / f"{spec.key}-candidate.jsonl",
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--v7-root", type=Path, required=True)
    parser.add_argument("--bulk-workspace", type=Path, required=True)
    parser.add_argument(
        "--v8-config",
        type=Path,
        default=ROOT / "configs/data/next100_065f_global_dedup_v8.json",
    )
    parser.add_argument(
        "--data526-config",
        type=Path,
        default=ROOT / "configs/data/data526_v8_record_composition_v1.json",
    )
    parser.add_argument("--v8-report", type=Path, required=True)
    parser.add_argument("--v8-survivors", type=Path, required=True)
    parser.add_argument(
        "--data526-evidence",
        type=Path,
        default=ROOT / "evidence/data526/v8/materialization_evidence.json",
    )
    parser.add_argument(
        "--data526-record-inventory",
        type=Path,
        default=ROOT / "evidence/data526/v8/record_inventory.json",
    )
    parser.add_argument(
        "--rada-language-report",
        type=Path,
        default=ROOT / "evidence/d03-rada-trees/secondary-plaintext-language-gate-v1.json",
    )
    parser.add_argument("--rada-quality-privacy-jsonl", type=Path, required=True)
    parser.add_argument("--rada-quality-privacy-report", type=Path, required=True)
    parser.add_argument("--expected-rada-report-sha256", required=True)
    parser.add_argument(
        "--arxiv-authority",
        type=Path,
        default=ROOT
        / "configs/data/d03_common_pile_arxiv_abstracts_source_admission_v1.json",
    )
    parser.add_argument(
        "--languk-authority",
        type=Path,
        default=ROOT
        / "configs/data/d03_languk_supreme_court_postexecution_rights_v1.json",
    )
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-survivors", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        _require_pyarrow()
        _verify_incumbent_checkout()
        workspace = args.workspace.resolve()
        historical = workspace / "historical"
        for spec in (ARXIV, LANGUK):
            _verify_historical_program(ROOT, spec)
            _extract_commit(ROOT, spec.execution_commit, historical / spec.key)

        results: list[dict[str, Any]] = []
        for pass_no in (1, 2):
            pass_root = workspace / f"pass-{pass_no}"
            if pass_root.exists():
                shutil.rmtree(pass_root)
            pass_root.mkdir(parents=True)
            try:
                arxiv_candidate = _run_historical_materializer(
                    spec=ARXIV,
                    historical_root=historical / ARXIV.key,
                    pass_root=pass_root,
                )
                languk_candidate = _run_historical_materializer(
                    spec=LANGUK,
                    historical_root=historical / LANGUK.key,
                    pass_root=pass_root,
                )
                report = pass_root / "postadmission-report.json"
                survivors = pass_root / "postadmission-survivors.json"
                _run(
                    _replay_command(
                        args,
                        arxiv_candidate=arxiv_candidate,
                        languk_candidate=languk_candidate,
                        report=report,
                        survivors=survivors,
                    ),
                    cwd=ROOT,
                )
                results.append(
                    _pass_result(
                        arxiv_candidate=arxiv_candidate,
                        languk_candidate=languk_candidate,
                        report=report,
                        survivors=survivors,
                    )
                )
            finally:
                _remove_ephemeral_payloads(pass_root)

        receipt = build_receipt(
            pass_results=results,
            incumbent_runner_blob_sha1=git_blob_sha1(INCUMBENT_RUNNER.read_bytes()),
            incumbent_intake_blob_sha1=git_blob_sha1(INCUMBENT_INTAKE.read_bytes()),
        )
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_survivors.parent.mkdir(parents=True, exist_ok=True)
        args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / "pass-1/postadmission-report.json", args.output_report)
        shutil.copyfile(
            workspace / "pass-1/postadmission-survivors.json",
            args.output_survivors,
        )
        args.output_receipt.write_bytes(canonical_json_bytes(receipt))
    except (RematerializationError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        return 2

    print("D03_ARXIV_LANGUK_REMATERIALIZED_V9_REPLAY=PASS_ZERO_CREDIT")
    print("TWO_CLEAN_REPLAY_PASSES=true")
    print("CANONICAL_CAPACITY_CREDITED=0")
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TOKENIZER_FIT_AUTHORIZED=false")
    print("OPTIMIZER_UPDATES_EXECUTED_ON_REAL_TARGETS=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
