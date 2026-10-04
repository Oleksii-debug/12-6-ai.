#!/usr/bin/env python3
"""Rematerialize ArXiv+LangUK twice and replay them on the current clean D03 graph."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from contextlib import contextmanager
from importlib import metadata
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_HELPER_PATH = SRC / "twelve_six/data/arxiv_languk_rematerialized_replay_v1.py"
_HELPER_SPEC = importlib.util.spec_from_file_location("_pr1851_replay_helper", _HELPER_PATH)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:
    raise RuntimeError(f"cannot load PR1851 replay helper: {_HELPER_PATH}")
_HELPER = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules[_HELPER_SPEC.name] = _HELPER
try:
    _HELPER_SPEC.loader.exec_module(_HELPER)
except BaseException:
    sys.modules.pop(_HELPER_SPEC.name, None)
    raise

ARXIV = _HELPER.ARXIV
LANGUK = _HELPER.LANGUK
PARENT_PR1800_HEAD = _HELPER.PARENT_PR1800_HEAD
HistoricalMaterializerSpec = _HELPER.HistoricalMaterializerSpec
RematerializationError = _HELPER.RematerializationError
build_receipt = _HELPER.build_receipt
canonical_json_bytes = _HELPER.canonical_json_bytes
git_blob_sha1 = _HELPER.git_blob_sha1
sha256_bytes = _HELPER.sha256_bytes
verify_candidate = _HELPER.verify_candidate
verify_git_blob = _HELPER.verify_git_blob

INCUMBENT_RUNNER_REL = Path("tools/run_d03_arxiv_languk_postadmission_global_dedup_v2.py")
INCUMBENT_INTAKE_REL = Path("src/twelve_six/data/post_admission_dedup_intake_v2.py")
WRAPPER_RUNNER_REL = Path("tools/run_d03_arxiv_languk_rematerialized_replay_v1.py")
WRAPPER_HELPER_REL = Path("src/twelve_six/data/arxiv_languk_rematerialized_replay_v1.py")
CLEAN_SUCCESSOR_REL = Path("tools/run_d03_nomis_free_clean_successor_v1.py")
SURVIVOR_TOOL_REL = Path("tools/derive_next100_065f_v8_survivors.py")
INDEXED_EXECUTOR_REL = Path("src/twelve_six/data/incumbent_dedup_indexed_execution.py")
CURRENT_INTAKE_REL = Path("src/twelve_six/data/post_admission_dedup_intake_v2.py")

EXPECTED_CLEAN_SUCCESSOR_BLOB = "bcc5c40c54f0a93f42acfd584f800048bbf49e7e"
EXPECTED_SURVIVOR_TOOL_BLOB = "ae91d60e5d62466c69394abb1c4b27d2e49f40e3"
EXPECTED_INDEXED_EXECUTOR_BLOB = "f75336008839198b6d46bea4954f120e2d81613c"
EXPECTED_CURRENT_INTAKE_BLOB = "322f1441326ca17447581be8ebf2d98c2385bd38"
CURRENT_CLEAN_BASE_PR = 2107
CURRENT_INDEXED_EXECUTOR_PR = 1459
CURRENT_MAIN_AT_CONVERGENCE = "7b3df41c10a826183fab0b04ae85a90cdf0ce351"
EXPECTED_BASE_PRE_DEDUP_SOURCE_COUNT = 263
EXPECTED_BASE_PRE_DEDUP_BYTES = 6_093_965
EXPECTED_EXTENSION_SOURCE_COUNT = 1_280
EXPECTED_EXTENSION_BYTES = 3_949_184
EXPECTED_COMBINED_PRE_DEDUP_SOURCE_COUNT = 1_543
EXPECTED_COMBINED_PRE_DEDUP_BYTES = 10_043_149
EXPECTED_V7_HEAD = "d3333ec1b4a508df232a5aefccd6686adda745fb"
EXPECTED_V7_TREE = "f6bb58379e9e249583480c246b844b673be38b4c"

REPAIRED_RECEIPT_SCHEMA = "12-6.d03-arxiv-languk-current-clean-indexed-replay.v3"
CURRENT_REPORT_SCHEMA = "12-6.d03-arxiv-languk-current-clean-global-dedup.v1"
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
    if destination.exists() or destination.is_symlink():
        raise RematerializationError(f"refusing to overwrite historical checkout: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
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


def _load_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    """Decode exactly the bytes being hashed; reject ambiguous JSON."""

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RematerializationError(f"{label} duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> None:
        raise RematerializationError(f"{label} nonfinite JSON: {token}")

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_pairs,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RematerializationError(f"cannot decode {label}") from exc
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
    report_value = _load_json_object(report_raw, label="replay report")
    survivor_value = _load_json_object(survivor_raw, label="survivor authority")
    if report_raw != canonical_json_bytes(report_value):
        raise RematerializationError("on-disk replay report is noncanonical")
    if survivor_raw != canonical_json_bytes(survivor_value):
        raise RematerializationError("on-disk survivor authority is noncanonical")

    report_identity = report_value.get("report_sha256")
    survivor_identity = survivor_value.get("survivor_authority_sha256")
    report_body = dict(report_value)
    report_body.pop("report_sha256", None)
    if type(report_identity) is not str or report_identity != sha256_bytes(
        canonical_json_bytes(report_body)
    ):
        raise RematerializationError("on-disk replay report self-hash mismatch")
    if type(survivor_identity) is not str:
        raise RematerializationError("survivor authority identity missing")
    try:
        survivor_tool = _load_module(
            "_pr1851_disk_survivor_verifier", ROOT / SURVIVOR_TOOL_REL
        )
        survivor_tool.verify_survivor_authority(report_value, survivor_value)
    except Exception as exc:
        raise RematerializationError("on-disk survivor authority invalid") from exc

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


def _verify_workspace_targets(workspace: Path) -> None:
    """Fail before acquisition if a prior attempt or unrelated evidence is present."""
    if workspace.is_symlink() or (workspace.exists() and not workspace.is_dir()):
        raise RematerializationError("workspace must be a real directory or absent")
    for name in ("historical", "pass-1", "pass-2"):
        target = workspace / name
        if target.exists() or target.is_symlink():
            raise RematerializationError(f"refusing to reuse workspace evidence: {target}")


def _stage_new_bytes(path: Path, raw: bytes, *, label: str) -> Path:
    """Durably stage all bytes beside the final name; never create a partial final."""
    if path.exists() or path.is_symlink():
        raise RematerializationError(f"refusing to overwrite {label}: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        )
    except OSError as exc:
        raise RematerializationError(f"cannot stage {label}: {path}") from exc
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            if handle.write(raw) != len(raw):
                raise OSError(f"incomplete staged write for {label}")
            handle.flush()
            os.fsync(handle.fileno())
        if sha256_bytes(staged.read_bytes()) != sha256_bytes(raw):
            raise OSError(f"staged {label} digest mismatch")
        return staged
    except BaseException as failure:
        try:
            staged.unlink(missing_ok=True)
        except OSError as cleanup_error:
            raise RematerializationError(
                "STAGING_CLEANUP_INCOMPLETE: "
                f"{label} write/verification failed ({type(failure).__name__}), "
                f"staged cleanup failed ({type(cleanup_error).__name__}); "
                f"unpublished stage: {staged}; manual reconciliation required"
            ) from failure
        raise


def _link_staged_new_bytes(staged: Path, path: Path, *, label: str) -> None:
    """Atomic no-clobber publication, including a late destination race."""
    try:
        os.link(staged, path)
    except FileExistsError as exc:
        raise RematerializationError(f"refusing to overwrite {label}: {path}") from exc
    except OSError as exc:
        raise RematerializationError(f"cannot publish {label} atomically: {path}") from exc



def _matches_staged_identity(
    path: Path, expected: bytes, identity: tuple[int, int],
) -> bool:
    """Require the original inode and exact bytes, not merely a matching pathname."""
    try:
        if path.is_symlink():
            return False
        stat = path.stat(follow_symlinks=False)
        if (stat.st_dev, stat.st_ino) != identity:
            return False
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            return (
                (opened.st_dev, opened.st_ino) == identity
                and handle.read(len(expected) + 1) == expected
            )
    except OSError:
        return False


def _link_verified_new_bytes(
    staged: Path, path: Path, raw: bytes, *, label: str,
) -> None:
    """Hold the original inode open across no-replace publication and rollback.

    Trusted stable output directories are required. Keeping the descriptor
    open prevents deletion/recreation of the staged pathname from reusing the
    original inode while deciding whether it is safe to remove a failed link.
    """
    try:
        source = staged.open("rb")
    except OSError as exc:
        raise RematerializationError(
            f"staged {label} identity unavailable before publication"
        ) from exc
    with source:
        info = os.fstat(source.fileno())
        identity = (info.st_dev, info.st_ino)
        if (
            not info.st_ino
            or source.read(len(raw) + 1) != raw
            or not _matches_staged_identity(staged, raw, identity)
        ):
            raise RematerializationError(f"staged {label} changed before publication")
        _link_staged_new_bytes(staged, path, label=label)
        if (
            _matches_staged_identity(staged, raw, identity)
            and _matches_staged_identity(path, raw, identity)
        ):
            return
        try:
            published = path.stat(follow_symlinks=False)
        except OSError:
            published = None
        if (
            published is not None
            and not path.is_symlink()
            and (published.st_dev, published.st_ino) == identity
        ):
            try:
                path.unlink()
            except OSError as exc:
                raise RematerializationError(
                    f"invalid published {label}; rollback failed; "
                    "manual reconciliation required"
                ) from exc
        raise RematerializationError(
            f"published {label} failed exact byte/inode verification; "
            "manual reconciliation required"
        )

def _write_new_bytes(path: Path, raw: bytes, *, label: str) -> None:
    """Create one complete output, without following or replacing existing names."""
    staged = _stage_new_bytes(path, raw, label=label)
    published_and_verified = False
    try:
        _link_verified_new_bytes(staged, path, raw, label=label)
        published_and_verified = True
    finally:
        try:
            staged.unlink(missing_ok=True)
        except OSError as exc:
            if published_and_verified:
                raise RematerializationError(
                    f"{label} was published and byte-verified, but staged cleanup "
                    f"is pending: {staged}; inspect the final output before retry"
                ) from exc
            raise RematerializationError(
                f"{label} publication failed and staged cleanup is pending: "
                f"{staged}; manual reconciliation required"
            ) from exc


def _capture_verified_publication_bytes(
    pass_root: Path, pass_result: dict[str, str],
) -> tuple[bytes, bytes]:
    """Capture immutable bytes; reject changes since pass-level verification."""
    report_raw = (pass_root / "current-clean-report.json").read_bytes()
    survivors_raw = (pass_root / "current-clean-survivors.json").read_bytes()
    for label, raw, key in (
        ("replay report", report_raw, "report_file_sha256"),
        ("survivor authority", survivors_raw, "survivor_file_sha256"),
    ):
        expected = pass_result.get(key)
        if type(expected) is not str or sha256_bytes(raw) != expected:
            raise RematerializationError(
                f"{label} changed after pass verification; refusing publication"
            )
    return report_raw, survivors_raw



def inspect_outer_publication_recovery(
    args: argparse.Namespace,
    *,
    pass_root: Path,
    pass_result: dict[str, str],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Independently classify a prior publication without modifying any files.

    The verified pass-1 source bytes and caller's two-pass receipt are the
    external authorities. The journal alone is never trusted as a source of
    evidence, permission, or content identity.
    """
    report_raw, survivors_raw = _capture_verified_publication_bytes(
        pass_root, pass_result,
    )
    outputs = (
        ("outer report", args.output_report, report_raw),
        ("outer survivors", args.output_survivors, survivors_raw),
        ("outer receipt", args.output_receipt, canonical_json_bytes(receipt)),
    )
    intent_path = pass_root / "outer-publication-intent.json"
    expected_intent = {
        "schema": "12-6.d03-arxiv-languk-outer-publication-intent.v1",
        "status": "PREPARED_UNCOMMITTED",
        "commit_marker": str(args.output_receipt.resolve()),
        "outputs": [
            {"label": label, "path": str(path.resolve()), "sha256": sha256_bytes(raw)}
            for label, path, raw in outputs
        ],
        "canonical_capacity_credited": 0,
        "training_authorized": False,
    }
    if intent_path.is_symlink() or not intent_path.is_file():
        raise RematerializationError("recovery publication intent missing or unsafe")
    try:
        intent_raw = intent_path.read_bytes()
    except OSError as exc:
        raise RematerializationError("cannot read recovery publication intent") from exc
    if intent_raw != canonical_json_bytes(expected_intent):
        raise RematerializationError(
            "recovery publication intent differs from authenticated source identities"
        )
    published: list[str] = []
    for label, path, raw in outputs:
        if path.is_symlink():
            raise RematerializationError(f"recovery {label} must not be a symlink")
        if path.exists():
            if not path.is_file():
                raise RematerializationError(f"recovery {label} is not a file")
            try:
                observed_sha = sha256_bytes(path.read_bytes())
            except OSError as exc:
                raise RematerializationError(f"cannot read recovery {label}") from exc
            if observed_sha != sha256_bytes(raw):
                raise RematerializationError(
                    f"recovery {label} differs from authenticated expected bytes"
                )
            published.append(label)
    if "outer receipt" in published and len(published) != len(outputs):
        raise RematerializationError(
            "recovery receipt exists without both authenticated source authorities"
        )
    status = (
        "COMMITTED_ZERO_CREDIT"
        if len(published) == len(outputs)
        else "PARTIAL_UNCOMMITTED" if published else "PREPARED_UNCOMMITTED"
    )
    return {
        "status": status,
        "published": published,
        "missing": [label for label, _, _ in outputs if label not in published],
        "canonical_capacity_credited": 0,
        "training_authorized": False,
    }


def _publish_verified_outputs(
    args: argparse.Namespace,
    *,
    pass_root: Path,
    pass_result: dict[str, str],
    receipt: dict[str, Any],
) -> None:
    """Verify and capture both source files before creating any outer output."""
    _verify_outer_output_targets(args)
    report_raw, survivors_raw = _capture_verified_publication_bytes(
        pass_root, pass_result
    )
    receipt_raw = canonical_json_bytes(receipt)
    # Three filesystem names cannot be linked as one atomic operation. Stage
    # all three *before* the first final name and publish the receipt LAST as
    # the commit marker. A durable workspace intent makes interrupted batches
    # auditable and recoverable without deleting or replacing unrelated files.
    outputs = (
        ("outer report", args.output_report, report_raw),
        ("outer survivors", args.output_survivors, survivors_raw),
        ("outer receipt", args.output_receipt, receipt_raw),
    )
    staged: list[tuple[str, Path, Path, bytes]] = []
    published: list[tuple[str, Path, str]] = []
    intent_path = pass_root / "outer-publication-intent.json"
    if any(path.resolve() == intent_path.resolve() for _, path, _ in outputs):
        raise RematerializationError("outer output cannot alias publication intent")
    try:
        for label, path, raw in outputs:
            staged.append((label, path, _stage_new_bytes(path, raw, label=label), raw))
        intent = {
            "schema": "12-6.d03-arxiv-languk-outer-publication-intent.v1",
            "status": "PREPARED_UNCOMMITTED",
            "commit_marker": str(args.output_receipt.resolve()),
            "outputs": [
                {"label": label, "path": str(path.resolve()), "sha256": sha256_bytes(raw)}
                for label, path, _, raw in staged
            ],
            "canonical_capacity_credited": 0,
            "training_authorized": False,
        }
        _write_new_bytes(intent_path, canonical_json_bytes(intent), label="publication intent")
        for label, path, temporary, raw in staged:
            try:
                _link_verified_new_bytes(temporary, path, raw, label=label)
            except RematerializationError as exc:
                observed = [
                    {"label": done_label, "path": str(done_path), "sha256": digest}
                    for done_label, done_path, digest in published
                ]
                raise RematerializationError(
                    f"partial outer publication; commit receipt not verified; "
                    f"inspect {intent_path} and verify immutable outputs {observed}; "
                    f"manual reconciliation required before retry"
                ) from exc
            published.append((label, path, sha256_bytes(raw)))
    finally:
        primary_failure = sys.exc_info()[1]
        cleanup_error: OSError | None = None
        cleanup_failures: list[str] = []
        for _, _, temporary, _ in staged:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_failures.append(
                    f"{temporary}: {type(exc).__name__}: {exc}"
                )
                if cleanup_error is None:
                    cleanup_error = exc
        if cleanup_error is not None:
            if len(published) == len(outputs) and primary_failure is None:
                raise RematerializationError(
                    "outer receipt published and byte-verified; staged cleanup "
                    f"pending for {cleanup_failures}; inspect read-only recovery"
                ) from cleanup_error
            original = (
                f"; original publication failure: {primary_failure}"
                if primary_failure is not None else ""
            )
            raise RematerializationError(
                "outer publication incomplete and staged cleanup pending for "
                f"{cleanup_failures}{original}; manual reconciliation required"
            ) from (primary_failure if primary_failure is not None else cleanup_error)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RematerializationError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_current_execution_modules() -> tuple[Any, Any]:
    """Load current source-local/indexed code, then release current package namespace."""
    existing = sorted(
        name
        for name in sys.modules
        if name == "twelve_six" or name.startswith("twelve_six.")
    )
    if existing:
        raise RematerializationError(
            "current execution modules must load before terminal V7 namespace"
        )
    intake = _load_module("_pr1851_source_intake", ROOT / CURRENT_INTAKE_REL)
    indexed = _load_module("_pr1851_indexed_executor", ROOT / INDEXED_EXECUTOR_REL)
    imported = [
        name
        for name in sys.modules
        if name == "twelve_six" or name.startswith("twelve_six.")
    ]
    if not imported:
        raise RematerializationError("current execution namespace was not established")
    for name in imported:
        sys.modules.pop(name, None)
    return intake, indexed


def _namespace_package(name: str, path: Path) -> ModuleType:
    """Create a package shell that exposes only the authenticated directory."""
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]
    spec = ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(path)]
    module.__spec__ = spec
    return module


@contextmanager
def _isolated_terminal_v7_namespace(v7_root: Path) -> Iterator[None]:
    """Load terminal V7 data modules without executing historical package initializers."""
    source_root = (v7_root / "src").resolve()
    package_root = source_root / "twelve_six"
    data_root = package_root / "data"
    if not package_root.is_dir() or not data_root.is_dir():
        raise RematerializationError("terminal V7 package layout missing")

    prefix = "twelve_six"
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == prefix or name.startswith(f"{prefix}.")
    }
    previous_sys_path = list(sys.path)
    for name in list(previous_modules):
        sys.modules.pop(name, None)

    package = _namespace_package("twelve_six", package_root)
    data_package = _namespace_package("twelve_six.data", data_root)
    package.data = data_package
    sys.modules["twelve_six"] = package
    sys.modules["twelve_six.data"] = data_package
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(f"{prefix}."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
        sys.path[:] = previous_sys_path


def _verify_v7_checkout(v7_root: Path) -> None:
    root = v7_root.resolve()
    if not root.is_dir():
        raise RematerializationError(f"terminal V7 checkout missing: {root}")
    if _git_text(root, "for-each-ref", "--format=%(refname)", "refs/replace"):
        raise RematerializationError("terminal V7 replace refs are forbidden")
    if _git_text(root, "rev-parse", "HEAD") != EXPECTED_V7_HEAD:
        raise RematerializationError("terminal V7 HEAD drift")
    if _git_text(root, "rev-parse", "HEAD^{tree}") != EXPECTED_V7_TREE:
        raise RematerializationError("terminal V7 tree drift")
    if _git_text(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RematerializationError("terminal V7 checkout must be clean")


def _declared_capacity_bytes(inventory: dict[str, Any]) -> int:
    rows = inventory.get("sources")
    if not isinstance(rows, list):
        raise RematerializationError("inventory sources missing for capacity arithmetic")
    total = 0
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise RematerializationError(f"inventory source {index} must be object")
        value = row.get("declared_capacity_bytes")
        if type(value) is not int or value <= 0:
            raise RematerializationError(
                f"inventory source {index} declared capacity invalid"
            )
        total += value
    return total


def _verify_current_clean_dependencies() -> dict[str, str]:
    bindings = {
        "clean_successor_tool_git_blob_sha1": (
            CLEAN_SUCCESSOR_REL,
            EXPECTED_CLEAN_SUCCESSOR_BLOB,
        ),
        "survivor_tool_git_blob_sha1": (
            SURVIVOR_TOOL_REL,
            EXPECTED_SURVIVOR_TOOL_BLOB,
        ),
        "indexed_executor_git_blob_sha1": (
            INDEXED_EXECUTOR_REL,
            EXPECTED_INDEXED_EXECUTOR_BLOB,
        ),
        "source_intake_git_blob_sha1": (
            CURRENT_INTAKE_REL,
            EXPECTED_CURRENT_INTAKE_BLOB,
        ),
    }
    result: dict[str, str] = {}
    for key, (relative, expected) in bindings.items():
        candidate = ROOT / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise RematerializationError(f"current clean dependency missing: {relative}")
        verify_git_blob(candidate.read_bytes(), expected, label=relative.as_posix())
        result[key] = expected
    return result


def _reconstruct_current_clean_base(
    *,
    v7_root: Path,
    workspace: Path,
) -> tuple[Any, dict[str, Any], dict[str, bytes], dict[str, Any]]:
    _verify_v7_checkout(v7_root)
    clean = _load_module("_pr1851_clean_successor", ROOT / CLEAN_SUCCESSOR_REL)
    clean.validate_runtime_bindings(ROOT)

    v8 = clean._load_module(
        "_pr1851_incumbent_v8",
        ROOT / clean.INCUMBENT_V8_PATH,
    )
    quarantine = clean._load_module(
        "_pr1851_quarantine",
        ROOT / clean.QUARANTINE_MODULE_PATH,
    )
    quarantine_authority = clean._read_json(ROOT / clean.QUARANTINE_CONFIG_PATH)
    v8_config = v8.load_config(ROOT / clean.INCUMBENT_V8_CONFIG_PATH)

    v7, baseline_report, inventory, payloads = v8._capture_terminal_v7(
        v7_root,
        v8_config,
    )
    clean_inventory, clean_payloads, removal = clean.deauthorize_exact_nomis(
        inventory,
        payloads,
        quarantine_authority,
        quarantine,
    )

    matcher = v7.v6.v3
    historical_dedup = matcher.audit_payloads(clean_inventory, clean_payloads)
    matcher.verify_report(historical_dedup)
    clean._assert_clean_dedup(
        historical_dedup,
        clean.EXPECTED_CLEAN_HISTORICAL,
        label="clean historical",
    )

    bulk_report, bulk_rows, bulk_payloads = v8._materialize_bulk(
        ROOT,
        workspace,
        v8_config,
    )
    combined_inventory = copy.deepcopy(clean_inventory)
    existing_ids = {row["source_id"] for row in combined_inventory["sources"]}
    if existing_ids & set(bulk_payloads):
        raise RematerializationError("bulk source id collides with clean historical graph")
    combined_inventory["sources"] = [*combined_inventory["sources"], *bulk_rows]
    combined_inventory["final_refresh_required"] = False
    combined_inventory["terminal_refresh_cutoff_utc"] = "2026-09-14T19:14:21Z"
    combined_inventory["terminal_refresh_rule"] = (
        "PR1851 reuses merged PR2107 clean-successor mechanics: authenticate terminal "
        "V7, remove exact quarantined Nomis1864 before matcher invocation, then append "
        "unchanged DATA-BULK-CODE-1 source objects. No training/corpus credit is implied."
    )
    combined_payloads = dict(clean_payloads)
    combined_payloads.update(bulk_payloads)

    base_dedup = matcher.audit_payloads(combined_inventory, combined_payloads)
    matcher.verify_report(base_dedup)
    clean._assert_clean_dedup(
        base_dedup,
        clean.EXPECTED_CLEAN_COMPOSED,
        label="clean composed",
    )
    base_vector = clean._source_vector(base_dedup)
    if base_vector["source_object_count"] != EXPECTED_BASE_PRE_DEDUP_SOURCE_COUNT:
        raise RematerializationError("clean base source-count drift")
    if (
        base_vector["source_capacity_bytes_before_global_dedup"]
        != EXPECTED_BASE_PRE_DEDUP_BYTES
    ):
        raise RematerializationError("clean base byte-capacity drift")

    authority = {
        "clean_successor_product_pr": CURRENT_CLEAN_BASE_PR,
        "historical_v7_report_sha256": baseline_report["report_sha256"],
        "bulk_terminal_report_identity_sha256": bulk_report["report_identity_sha256"],
        "deauthorization": removal,
        "clean_base_matcher_report_sha256": base_dedup["report_sha256"],
        "clean_base_source_vector": base_vector,
    }
    return matcher, combined_inventory, combined_payloads, authority


def _build_current_clean_replay(
    args: argparse.Namespace,
    *,
    pass_root: Path,
    arxiv_candidate: Path,
    languk_candidate: Path,
    source_intake: Any,
    indexed_executor: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_bindings = _verify_current_clean_dependencies()
    v7_root = _repo_rooted(args.v7_root)
    with _isolated_terminal_v7_namespace(v7_root):
        matcher, base_inventory, base_payloads, clean_authority = (
            _reconstruct_current_clean_base(
                v7_root=v7_root,
                workspace=pass_root / "clean-base-workspace",
            )
        )
        return _build_current_clean_replay_with_base(
            args,
            pass_root=pass_root,
            arxiv_candidate=arxiv_candidate,
            languk_candidate=languk_candidate,
            source_intake=source_intake,
            indexed_executor=indexed_executor,
            matcher=matcher,
            base_inventory=base_inventory,
            base_payloads=base_payloads,
            clean_authority=clean_authority,
            runtime_bindings=runtime_bindings,
        )


def _build_current_clean_replay_with_base(
    args: argparse.Namespace,
    *,
    pass_root: Path,
    arxiv_candidate: Path,
    languk_candidate: Path,
    source_intake: Any,
    indexed_executor: Any,
    matcher: Any,
    base_inventory: dict[str, Any],
    base_payloads: dict[str, bytes],
    clean_authority: dict[str, Any],
    runtime_bindings: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    intake = source_intake
    arxiv_rows, arxiv_payloads, arxiv_receipt = intake._prepare_source(
        intake.ARXIV,
        authority_raw=_repo_rooted(args.arxiv_authority).read_bytes(),
        candidate_raw=arxiv_candidate.read_bytes(),
    )
    languk_rows, languk_payloads, languk_receipt = intake._prepare_source(
        intake.LANGUK,
        authority_raw=_repo_rooted(args.languk_authority).read_bytes(),
        candidate_raw=languk_candidate.read_bytes(),
    )
    source_overlap = set(arxiv_payloads) & set(languk_payloads)
    if source_overlap:
        raise RematerializationError("ArXiv/LangUK source-local matcher id collision")
    extension_rows = [*arxiv_rows, *languk_rows]
    extension_payloads = {**arxiv_payloads, **languk_payloads}
    source_receipts = [arxiv_receipt, languk_receipt]
    if len(extension_rows) != EXPECTED_EXTENSION_SOURCE_COUNT:
        raise RematerializationError("ArXiv+LangUK extension source-count drift")
    if sum(len(raw) for raw in extension_payloads.values()) != EXPECTED_EXTENSION_BYTES:
        raise RematerializationError("ArXiv+LangUK extension byte-total drift")

    base_ids = set(base_payloads)
    overlap = base_ids & set(extension_payloads)
    if overlap:
        raise RematerializationError("ArXiv+LangUK source id collides with clean base")

    combined_inventory = copy.deepcopy(base_inventory)
    combined_inventory["sources"] = [
        *combined_inventory["sources"],
        *copy.deepcopy(extension_rows),
    ]
    combined_inventory["final_refresh_required"] = False
    combined_inventory["terminal_refresh_rule"] = (
        "Exact merged PR2107 Nomis-free source graph plus authenticated ArXiv/LangUK "
        "source-admission rows; execution delegates duplicate science to exact incumbent "
        "V3 semantics through merged PR1459 performance-equivalent indexing."
    )
    base_declared_capacity = _declared_capacity_bytes(base_inventory)
    extension_declared_capacity = sum(
        row["declared_capacity_bytes"] for row in extension_rows
    )
    if base_declared_capacity != EXPECTED_BASE_PRE_DEDUP_BYTES:
        raise RematerializationError(
            "clean base declared-capacity drift: "
            f"{base_declared_capacity} != {EXPECTED_BASE_PRE_DEDUP_BYTES}"
        )
    if extension_declared_capacity != EXPECTED_EXTENSION_BYTES:
        raise RematerializationError(
            "extension declared-capacity drift: "
            f"{extension_declared_capacity} != {EXPECTED_EXTENSION_BYTES}"
        )

    combined_payloads = dict(base_payloads)
    combined_payloads.update(extension_payloads)
    if len(combined_payloads) != EXPECTED_COMBINED_PRE_DEDUP_SOURCE_COUNT:
        raise RematerializationError("combined source-count drift")
    combined_declared_capacity = _declared_capacity_bytes(combined_inventory)
    if combined_declared_capacity != EXPECTED_COMBINED_PRE_DEDUP_BYTES:
        raise RematerializationError(
            "combined declared-capacity drift: "
            f"{combined_declared_capacity} != {EXPECTED_COMBINED_PRE_DEDUP_BYTES}"
        )

    indexed = indexed_executor
    indexed.attest_incumbent_runtime(matcher)
    dedup = indexed.audit_payloads_indexed(
        matcher,
        combined_inventory,
        combined_payloads,
    )
    matcher.verify_report(dedup)
    terminal = dedup.get("terminal_candidates")
    if not isinstance(terminal, dict):
        raise RematerializationError("current-clean matcher terminal result missing")
    if dedup.get("source_count") != EXPECTED_COMBINED_PRE_DEDUP_SOURCE_COUNT:
        raise RematerializationError("current-clean matcher source-count drift")
    if terminal.get("declared_capacity_bytes_before") != EXPECTED_COMBINED_PRE_DEDUP_BYTES:
        raise RematerializationError("current-clean matcher pre-dedup byte-total drift")

    clean = _load_module("_pr1851_clean_vector", ROOT / CLEAN_SUCCESSOR_REL)
    source_vector = clean._source_vector(dedup)
    core: dict[str, Any] = {
        "schema_version": CURRENT_REPORT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "current_main_at_convergence": CURRENT_MAIN_AT_CONVERGENCE,
        "clean_base_authority": clean_authority,
        "runtime_bindings": runtime_bindings,
        "matcher_lineage": {
            "science": "INCUMBENT_NEXT100_065_V3_UNCHANGED",
            "performance_executor_product_pr": CURRENT_INDEXED_EXECUTOR_PR,
            "performance_equivalent_indexing": True,
        },
        "post_admission_sources": copy.deepcopy(source_receipts),
        "source_vector": source_vector,
        "dedup_v3": copy.deepcopy(dict(dedup)),
        "raw_text_emitted": False,
        "truth_boundary": {
            "clean_nomis_free_base_used": True,
            "arxiv_source_admission_authenticated": True,
            "languk_source_admission_authenticated": True,
            "global_dedup_execution_complete": True,
            "current_corpus_external_llm_free_claimed_by_this_replay": False,
            "reserved_evaluation_decontamination_complete": False,
            "canonical_quality_privacy_complete": False,
            "balance_and_family_caps_complete": False,
            "cluster_safe_split_complete": False,
            "packing_complete": False,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_payload_accessed": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
        },
    }
    report = {
        **core,
        "report_sha256": sha256_bytes(canonical_json_bytes(core)),
    }

    survivor_tool = _load_module("_pr1851_survivor_tool", ROOT / SURVIVOR_TOOL_REL)
    survivors = survivor_tool.derive_survivor_authority(report)
    survivor_tool.verify_survivor_authority(report, survivors)
    return report, survivors


def _write_pass_authorities(
    *,
    report_path: Path,
    survivor_path: Path,
    report: dict[str, Any],
    survivors: dict[str, Any],
) -> None:
    _write_new_bytes(
        report_path,
        canonical_json_bytes(report),
        label="pass report",
    )
    _write_new_bytes(
        survivor_path,
        canonical_json_bytes(survivors),
        label="pass survivor authority",
    )


def _finalize_receipt(
    receipt: dict[str, Any],
    *,
    wrapper_execution_authority: dict[str, str],
    current_clean_execution_authority: dict[str, str],
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

    if current_clean_execution_authority != _verify_current_clean_dependencies():
        raise RematerializationError("current clean execution authority drift")

    result = copy.deepcopy(receipt)
    historical_parent = result.get("historical_parent_lineage")
    if not isinstance(historical_parent, dict):
        raise RematerializationError("historical parent lineage missing")
    if historical_parent.get("exact_head_sha") != PARENT_PR1800_HEAD:
        raise RematerializationError("historical parent exact head drift")
    if historical_parent.get("used_as_current_corpus_execution_authority") is not False:
        raise RematerializationError("historical parent must remain execution-nonauthoritative")
    result["schema_version"] = REPAIRED_RECEIPT_SCHEMA
    result["status"] = (
        "PHYSICAL_REMATERIALIZATION_AND_CURRENT_CLEAN_INDEXED_DEDUP_REPLAY_"
        "EXECUTED_ZERO_CREDIT"
    )
    result["current_clean_execution_authority"] = {
        "clean_successor_product_pr": CURRENT_CLEAN_BASE_PR,
        "indexed_executor_product_pr": CURRENT_INDEXED_EXECUTOR_PR,
        "main_at_convergence": CURRENT_MAIN_AT_CONVERGENCE,
        **current_clean_execution_authority,
    }
    result["wrapper_execution_authority"] = dict(wrapper_execution_authority)
    truth = dict(result.get("truth_boundary", {}))
    truth.pop("external_llm_or_api_used_for_data_or_intelligence", None)
    truth["global_dedup_replay_executed"] = True
    truth["current_corpus_external_llm_free_claimed_by_this_replay"] = False
    result["truth_boundary"] = truth
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--v7-root", type=Path, required=True)
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
        _verify_workspace_targets(args.workspace)
        _require_pyarrow()
        wrapper_authority = _verify_wrapper_checkout(ROOT)
        clean_execution_authority = _verify_current_clean_dependencies()
        source_intake, indexed_executor = _load_current_execution_modules()
        workspace = args.workspace.resolve()
        historical = workspace / "historical"
        for spec in (ARXIV, LANGUK):
            _verify_historical_program(ROOT, spec)
            _extract_commit(ROOT, spec.execution_commit, historical / spec.key)

        results: list[dict[str, Any]] = []
        for pass_no in (1, 2):
            pass_root = workspace / f"pass-{pass_no}"
            pass_root.mkdir(parents=True, exist_ok=False)
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
                report_path = pass_root / "current-clean-report.json"
                survivor_path = pass_root / "current-clean-survivors.json"
                report, survivors = _build_current_clean_replay(
                    args,
                    pass_root=pass_root,
                    arxiv_candidate=arxiv_candidate,
                    languk_candidate=languk_candidate,
                    source_intake=source_intake,
                    indexed_executor=indexed_executor,
                )
                _write_pass_authorities(
                    report_path=report_path,
                    survivor_path=survivor_path,
                    report=report,
                    survivors=survivors,
                )
                results.append(
                    _pass_result(
                        arxiv_candidate=arxiv_candidate,
                        languk_candidate=languk_candidate,
                        report=report_path,
                        survivors=survivor_path,
                    )
                )
            finally:
                _remove_ephemeral_payloads(pass_root)

        receipt = build_receipt(pass_results=results)
        receipt = _finalize_receipt(
            receipt,
            wrapper_execution_authority=wrapper_authority,
            current_clean_execution_authority=clean_execution_authority,
        )
        _publish_verified_outputs(
            args,
            pass_root=workspace / "pass-1",
            pass_result=results[0],
            receipt=receipt,
        )
    except (RematerializationError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}")
        return 2

    print("D03_ARXIV_LANGUK_CURRENT_CLEAN_INDEXED_REPLAY=PASS_ZERO_CREDIT")
    print("TWO_CLEAN_REPLAY_PASSES=true")
    print("CANONICAL_CAPACITY_CREDITED=0")
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TOKENIZER_FIT_AUTHORIZED=false")
    print("OPTIMIZER_UPDATES_EXECUTED_ON_REAL_TARGETS=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
