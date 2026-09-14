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
    PARENT_PR1800_HEAD,
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

INCUMBENT_RUNNER_REL = Path("tools/run_d03_arxiv_languk_postadmission_global_dedup_v2.py")
INCUMBENT_INTAKE_REL = Path("src/twelve_six/data/post_admission_dedup_intake_v2.py")
WRAPPER_RUNNER_REL = Path("tools/run_d03_arxiv_languk_rematerialized_replay_v1.py")
WRAPPER_HELPER_REL = Path("src/twelve_six/data/arxiv_languk_rematerialized_replay_v1.py")
REPAIRED_RECEIPT_SCHEMA = "12-6.d03-arxiv-languk-rematerialized-v9-replay.v2"
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


def _git_text(repo_root: Path, *args: str) -> str:
    try:
        return _git(repo_root, *args).decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise RematerializationError("git returned non-UTF-8 authority data") from exc


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


def _require_lower_hex(value: str, *, length: int, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != length
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RematerializationError(f"{label} malformed")


def _verify_wrapper_checkout(repo_root: Path) -> dict[str, str]:
    """Bind the exact tracked wrapper/helper bytes that construct the durable receipt."""
    source_head = _git_text(repo_root, "rev-parse", "HEAD")
    _require_lower_hex(source_head, length=40, label="wrapper source head")

    authority: dict[str, str] = {"source_head_sha": source_head}
    for key, relative in (
        ("runner", WRAPPER_RUNNER_REL),
        ("helper", WRAPPER_HELPER_REL),
    ):
        path = repo_root / relative
        if path.is_symlink() or not path.is_file():
            raise RematerializationError(f"wrapper {key} must be a regular file")
        tracked = _git_text(
            repo_root,
            "ls-files",
            "--error-unmatch",
            "--",
            relative.as_posix(),
        )
        if tracked != relative.as_posix():
            raise RematerializationError(f"wrapper {key} is not tracked exactly")
        expected_blob = _git_text(repo_root, "rev-parse", f"HEAD:{relative.as_posix()}")
        observed_blob = git_blob_sha1(path.read_bytes())
        if observed_blob != expected_blob:
            raise RematerializationError(f"wrapper {key} working-tree blob drift")
        authority[f"{key}_path"] = relative.as_posix()
        authority[f"{key}_blob_sha1"] = observed_blob
    return authority


def _prepare_parent_execution_tree(repo_root: Path, workspace: Path) -> Path:
    """Extract and verify the exact audited PR1800 tree used for all replay code/imports."""
    _ensure_commit(repo_root, PARENT_PR1800_HEAD)
    parent_root = workspace / "parent-pr1800"
    _extract_commit(repo_root, PARENT_PR1800_HEAD, parent_root)
    runner = parent_root / INCUMBENT_RUNNER_REL
    intake = parent_root / INCUMBENT_INTAKE_REL
    if runner.is_symlink() or not runner.is_file():
        raise RematerializationError("PR1800 incumbent runner missing from exact parent tree")
    if intake.is_symlink() or not intake.is_file():
        raise RematerializationError("PR1800 incumbent intake missing from exact parent tree")
    verify_git_blob(
        runner.read_bytes(),
        PARENT_RUNNER_BLOB_SHA1,
        label="PR1800 incumbent runner",
    )
    verify_git_blob(
        intake.read_bytes(),
        PARENT_INTAKE_BLOB_SHA1,
        label="PR1800 incumbent intake",
    )
    return parent_root


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


def _repo_rooted(path: Path) -> Path:
    """Preserve the incumbent wrapper's repo-root-relative child-argument semantics."""
    return (path if path.is_absolute() else ROOT / path).resolve()


def _replay_command(
    args: argparse.Namespace,
    *,
    parent_root: Path,
    arxiv_candidate: Path,
    languk_candidate: Path,
    report: Path,
    survivors: Path,
) -> list[str]:
    return [
        sys.executable,
        "-I",
        str(parent_root / INCUMBENT_RUNNER_REL),
        "--v7-root",
        str(_repo_rooted(args.v7_root)),
        "--bulk-workspace",
        str(_repo_rooted(args.bulk_workspace)),
        "--v8-config",
        str(_repo_rooted(args.v8_config)),
        "--data526-config",
        str(_repo_rooted(args.data526_config)),
        "--v8-report",
        str(_repo_rooted(args.v8_report)),
        "--v8-survivors",
        str(_repo_rooted(args.v8_survivors)),
        "--data526-evidence",
        str(_repo_rooted(args.data526_evidence)),
        "--data526-record-inventory",
        str(_repo_rooted(args.data526_record_inventory)),
        "--rada-language-report",
        str(_repo_rooted(args.rada_language_report)),
        "--rada-quality-privacy-jsonl",
        str(_repo_rooted(args.rada_quality_privacy_jsonl)),
        "--rada-quality-privacy-report",
        str(_repo_rooted(args.rada_quality_privacy_report)),
        "--expected-rada-report-sha256",
        args.expected_rada_report_sha256,
        "--arxiv-authority",
        str(_repo_rooted(args.arxiv_authority)),
        "--arxiv-candidate",
        str(arxiv_candidate.resolve()),
        "--languk-authority",
        str(_repo_rooted(args.languk_authority)),
        "--languk-candidate",
        str(languk_candidate.resolve()),
        "--output-report",
        str(report.resolve()),
        "--output-survivors",
        str(survivors.resolve()),
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


def _verify_outer_output_targets(args: argparse.Namespace) -> None:
    outputs = (
        ("outer report", args.output_report),
        ("outer survivors", args.output_survivors),
        ("outer receipt", args.output_receipt),
    )
    resolved: set[Path] = set()
    for label, path in outputs:
        if path.exists() or path.is_symlink():
            raise RematerializationError(f"refusing to overwrite {label}: {path}")
        target = path.resolve()
        if target in resolved:
            raise RematerializationError("outer output targets must be distinct")
        resolved.add(target)


def _write_new_bytes(path: Path, raw: bytes, *, label: str) -> None:
    """Publish authority bytes with exclusive-create semantics; never clobber or follow links."""
    if path.exists() or path.is_symlink():
        raise RematerializationError(f"refusing to overwrite {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
    except FileExistsError as exc:
        raise RematerializationError(f"refusing to overwrite {label}: {path}") from exc


def _finalize_receipt(
    receipt: dict[str, Any],
    *,
    wrapper_execution_authority: dict[str, str],
) -> dict[str, Any]:
    required_wrapper_keys = {
        "source_head_sha",
        "runner_path",
        "runner_blob_sha1",
        "helper_path",
        "helper_blob_sha1",
    }
    if set(wrapper_execution_authority) != required_wrapper_keys:
        raise RematerializationError("wrapper execution authority keyset drift")
    _require_lower_hex(
        wrapper_execution_authority["source_head_sha"],
        length=40,
        label="wrapper source head",
    )
    for key in ("runner_blob_sha1", "helper_blob_sha1"):
        _require_lower_hex(
            wrapper_execution_authority[key],
            length=40,
            label=f"wrapper {key}",
        )
    if wrapper_execution_authority["runner_path"] != WRAPPER_RUNNER_REL.as_posix():
        raise RematerializationError("wrapper runner path drift")
    if wrapper_execution_authority["helper_path"] != WRAPPER_HELPER_REL.as_posix():
        raise RematerializationError("wrapper helper path drift")

    result = dict(receipt)
    result["schema_version"] = REPAIRED_RECEIPT_SCHEMA
    parent = dict(result.get("parent_authority", {}))
    if parent.get("exact_head_sha") != PARENT_PR1800_HEAD:
        raise RematerializationError("receipt parent exact head drift")
    parent["execution_tree_mode"] = "EXTRACTED_EXACT_GIT_TREE"
    parent["isolated_python_mode"] = True
    result["parent_authority"] = parent
    result["wrapper_execution_authority"] = dict(wrapper_execution_authority)
    return result


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
        _verify_outer_output_targets(args)
        _require_pyarrow()
        wrapper_authority = _verify_wrapper_checkout(ROOT)
        workspace = args.workspace.resolve()
        parent_root = _prepare_parent_execution_tree(ROOT, workspace)
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
                        parent_root=parent_root,
                        arxiv_candidate=arxiv_candidate,
                        languk_candidate=languk_candidate,
                        report=report,
                        survivors=survivors,
                    ),
                    cwd=parent_root,
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
            incumbent_runner_blob_sha1=git_blob_sha1(
                (parent_root / INCUMBENT_RUNNER_REL).read_bytes()
            ),
            incumbent_intake_blob_sha1=git_blob_sha1(
                (parent_root / INCUMBENT_INTAKE_REL).read_bytes()
            ),
        )
        receipt = _finalize_receipt(
            receipt,
            wrapper_execution_authority=wrapper_authority,
        )
        _write_new_bytes(
            args.output_report,
            (workspace / "pass-1/postadmission-report.json").read_bytes(),
            label="outer report",
        )
        _write_new_bytes(
            args.output_survivors,
            (workspace / "pass-1/postadmission-survivors.json").read_bytes(),
            label="outer survivors",
        )
        _write_new_bytes(
            args.output_receipt,
            canonical_json_bytes(receipt),
            label="outer receipt",
        )
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
