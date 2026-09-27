#!/usr/bin/env python3
"""Execute the current clean DATA-232 -> G05/G06 zero-credit composition."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

CARRIER_PATH = "tools/execute_current_clean_composition_v1.py"
MODULE_PATH = "src/twelve_six/data/current_clean_execution_v1.py"
AUTHENTICATED_DEPENDENCY_PATHS = (
    "src/twelve_six/data/current_reserved_decontamination_v1.py",
    "src/twelve_six/data/eval647_reserved_decontamination_v1.py",
    "src/twelve_six/data/quality_execution_authority.py",
    "src/twelve_six/data/privacy_execution_authority.py",
    "src/twelve_six/data/decontamination_authority_v2.py",
    "src/twelve_six/data/_data232_decontamination_matching.py",
    "src/twelve_six/data/eval647_future_training_exclusion_v1.py",
    "src/twelve_six/data/document_quality.py",
    "src/twelve_six/data/quality_granularity.py",
    "src/twelve_six/data/privacy_filter_v3.py",
)
OUTPUT_FILES = {
    "composition_receipt": "composition_receipt.json",
    "data232_report": "data232_report.json",
    "decontamination_execution": "decontamination_execution.json",
    "eval647_execution_receipt": "eval647_execution_receipt.json",
    "quality_execution": "quality_execution.json",
    "privacy_execution": "privacy_execution.json",
    "survivor_inventory": "survivor_inventory.json",
}
SURVIVOR_RECORDS_NAME = "survivor_records.jsonl"

ISOLATED_CHILD_BOOTSTRAP = r"""
import subprocess
import sys

repo_root, git_sha, carrier_path, *child_args = sys.argv[1:]
completed = subprocess.run(
    ["git", "show", f"{git_sha}:{carrier_path}"],
    cwd=repo_root,
    capture_output=True,
    check=False,
)
if completed.returncode != 0:
    raise RuntimeError("unable to read authenticated Git carrier bytes")
synthetic_file = f"git:{git_sha}:{carrier_path}"
sys.argv = [synthetic_file, *child_args]
namespace = {
    "__name__": "__main__",
    "__file__": synthetic_file,
    "__package__": None,
}
exec(compile(completed.stdout, synthetic_file, "exec", dont_inherit=True), namespace)
"""


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_bytes(repo_root: Path, git_sha: str, repo_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{git_sha}:{repo_path}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"unable to read authenticated Git bytes for {repo_path}")
    return completed.stdout


def require_exact_checkout(repo_root: Path, expected_git_sha: str) -> str:
    if len(expected_git_sha) != 40 or any(
        c not in "0123456789abcdef" for c in expected_git_sha
    ):
        raise RuntimeError("expected carrier Git SHA must be lowercase 40-hex")
    head = _git(repo_root, "rev-parse", "--verify", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != expected_git_sha:
        raise RuntimeError("checkout HEAD differs from independently expected carrier SHA")
    for args in (
        ("diff", "--quiet", "HEAD", "--"),
        ("diff", "--cached", "--quiet", "HEAD", "--"),
    ):
        completed = _git(repo_root, *args)
        if completed.returncode not in (0, 1):
            raise RuntimeError("unable to verify tracked working-tree cleanliness")
        if completed.returncode:
            raise RuntimeError("tracked working tree differs from expected carrier head")

    for repo_path in (CARRIER_PATH, MODULE_PATH, *AUTHENTICATED_DEPENDENCY_PATHS):
        physical = (repo_root / repo_path).resolve(strict=True).read_bytes()
        authenticated = _git_bytes(repo_root, expected_git_sha, repo_path)
        if physical != authenticated:
            raise RuntimeError(f"physical bytes differ from authenticated Git bytes: {repo_path}")
    return expected_git_sha


def _git_bytes_optional(repo_root: Path, git_sha: str, repo_path: str) -> bytes | None:
    completed = subprocess.run(
        ["git", "show", f"{git_sha}:{repo_path}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout


class _AuthenticatedGitFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Load twelve_six modules directly from immutable Git object bytes."""

    def __init__(self, repo_root: Path, git_sha: str) -> None:
        self.repo_root = repo_root
        self.git_sha = git_sha
        self._sources: dict[str, tuple[bytes, str, bool]] = {}

    def _source(self, fullname: str) -> tuple[bytes, str, bool] | None:
        cached = self._sources.get(fullname)
        if cached is not None:
            return cached
        relative = fullname.replace(".", "/")
        for repo_path, is_package in (
            (f"src/{relative}/__init__.py", True),
            (f"src/{relative}.py", False),
        ):
            payload = _git_bytes_optional(self.repo_root, self.git_sha, repo_path)
            if payload is not None:
                source = (payload, repo_path, is_package)
                self._sources[fullname] = source
                return source
        return None

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname != "twelve_six" and not fullname.startswith("twelve_six."):
            return None
        source = self._source(fullname)
        if source is None:
            raise ImportError(f"authenticated Git module is unavailable: {fullname}")
        return importlib.util.spec_from_loader(fullname, self, is_package=source[2])

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        source = self._source(module.__name__)
        if source is None:
            raise ImportError(f"authenticated Git source disappeared: {module.__name__}")
        payload, repo_path, is_package = source
        physical_path = (self.repo_root / repo_path).resolve(strict=True)
        if physical_path.read_bytes() != payload:
            raise ImportError(
                f"physical source drift before authenticated import: {repo_path}"
            )
        module.__file__ = str(physical_path)
        if is_package:
            module.__path__ = [str(physical_path.parent)]
        code = compile(payload, str(physical_path), "exec", dont_inherit=True)
        exec(code, module.__dict__)


def load_authenticated_executor(
    repo_root: Path,
    expected_git_sha: str,
) -> Callable[..., tuple[dict[str, Any], ...]]:
    """Import all project code exclusively from the independently pinned Git tree."""
    preloaded = sorted(
        name
        for name in sys.modules
        if name == "twelve_six" or name.startswith("twelve_six.")
    )
    if preloaded:
        raise RuntimeError(
            "project modules were preloaded before authenticated Git importer: "
            + ", ".join(preloaded)
        )

    finder = _AuthenticatedGitFinder(repo_root, expected_git_sha)
    sys.meta_path.insert(0, finder)
    try:
        module = importlib.import_module("twelve_six.data.current_clean_execution_v1")
    except Exception:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        raise
    executor = getattr(module, "execute_current_clean_composition", None)
    if not callable(executor):
        raise RuntimeError("authenticated composition executor is unavailable")
    return executor

def _load_json(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not strict UTF-8") from exc
    value = json.loads(decoded)
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            try:
                decoded = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"{label} line {line_number} is not strict UTF-8"
                ) from exc
            if not decoded.strip():
                raise ValueError(f"{label} line {line_number} is blank")
            value = json.loads(decoded)
            if type(value) is not dict:
                raise ValueError(f"{label} line {line_number} must be a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{label} must not be empty")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _verify_survivor_publication(
    path: Path,
    records: list[dict[str, Any]],
    inventory: dict[str, Any],
    composition: dict[str, Any],
) -> None:
    expected_hash = composition.get("survivor_jsonl_sha256")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in expected_hash)
    ):
        raise RuntimeError("composition survivor JSONL identity is invalid")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError("published survivor JSONL differs from execution receipt")

    record_count = len(records)
    payload_bytes = sum(
        len(row["normalized_payload"].encode("utf-8"))
        for row in records
    )
    if (
        type(inventory.get("record_count")) is not int
        or inventory["record_count"] != record_count
        or type(inventory.get("total_payload_bytes")) is not int
        or inventory["total_payload_bytes"] != payload_bytes
    ):
        raise RuntimeError("survivor inventory/count publication drift")
    if (
        type(composition.get("survivor_records")) is not int
        or composition["survivor_records"] != record_count
        or type(composition.get("survivor_payload_bytes")) is not int
        or composition["survivor_payload_bytes"] != payload_bytes
    ):
        raise RuntimeError("survivor receipt/count publication drift")


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise RuntimeError("atomic no-replace publication is unsupported") from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(number, os.strerror(number), destination)
    if number in (errno.ENOSYS, errno.EINVAL):
        raise RuntimeError("atomic no-replace publication is unsupported")
    raise OSError(number, os.strerror(number), destination)


def execute_and_publish(
    args: argparse.Namespace,
    executor: Callable[..., tuple[dict[str, Any], ...]],
) -> dict[str, Any]:
    output_dir = args.output_dir.resolve(strict=False)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")

    training_records = _load_jsonl(args.training_records_jsonl, "training records")
    evaluation_records = _load_jsonl(args.evaluation_records_jsonl, "evaluation records")
    training_handoff = _load_json(args.training_handoff_json, "training handoff")
    base_reserved_binding = _load_json(
        args.base_reserved_binding_json,
        "base reserved binding",
    )
    eval647_manifest = _load_json(args.eval647_manifest_json, "EVAL-647 manifest")
    eval647_evidence = _load_json(
        args.eval647_materialization_evidence_json,
        "EVAL-647 materialization evidence",
    )

    (
        composition,
        report,
        decontamination,
        eval647_receipt,
        quality,
        privacy,
        survivor_records,
        survivor_inventory,
    ) = executor(
        training_records,
        evaluation_records,
        training_handoff_evidence=training_handoff,
        base_reserved_binding=base_reserved_binding,
        eval647_manifest=eval647_manifest,
        eval647_materialization_evidence=eval647_evidence,
        expected_base_reserved_binding_identity_sha256=(
            args.expected_base_reserved_binding_identity_sha256
        ),
        expected_composed_reserved_binding_identity_sha256=(
            args.expected_composed_reserved_binding_identity_sha256
        ),
        expected_eval647_materialization_evidence_identity_sha256=(
            args.expected_eval647_materialization_evidence_identity_sha256
        ),
        expected_eval647_object_set_identity_sha256=(
            args.expected_eval647_object_set_identity_sha256
        ),
        expected_inventory_identity_sha256=args.expected_inventory_identity_sha256,
        expected_survivor_authority_sha256=args.expected_survivor_authority_sha256,
        expected_training_handoff_identity_sha256=(
            args.expected_training_handoff_identity_sha256
        ),
        expected_selection_validation_identity_sha256=(
            args.expected_selection_validation_identity_sha256
        ),
        expected_final_test_identity_sha256=args.expected_final_test_identity_sha256,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        values = {
            "composition_receipt": composition,
            "data232_report": report,
            "decontamination_execution": decontamination,
            "eval647_execution_receipt": eval647_receipt,
            "quality_execution": quality,
            "privacy_execution": privacy,
            "survivor_inventory": survivor_inventory,
        }
        for key, filename in OUTPUT_FILES.items():
            _write_json(temporary / filename, values[key])
        survivor_path = temporary / SURVIVOR_RECORDS_NAME
        _write_jsonl(survivor_path, survivor_records)
        _verify_survivor_publication(
            survivor_path,
            survivor_records,
            survivor_inventory,
            composition,
        )
        _rename_directory_no_replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return composition


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--expected-carrier-git-sha", required=True)
    parser.add_argument("--training-records-jsonl", type=Path, required=True)
    parser.add_argument("--training-handoff-json", type=Path, required=True)
    parser.add_argument("--evaluation-records-jsonl", type=Path, required=True)
    parser.add_argument("--base-reserved-binding-json", type=Path, required=True)
    parser.add_argument("--eval647-manifest-json", type=Path, required=True)
    parser.add_argument(
        "--eval647-materialization-evidence-json",
        type=Path,
        required=True,
    )
    for name in (
        "base-reserved-binding-identity-sha256",
        "composed-reserved-binding-identity-sha256",
        "eval647-materialization-evidence-identity-sha256",
        "eval647-object-set-identity-sha256",
        "inventory-identity-sha256",
        "survivor-authority-sha256",
        "training-handoff-identity-sha256",
        "selection-validation-identity-sha256",
        "final-test-identity-sha256",
    ):
        parser.add_argument(f"--expected-{name}", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--isolated-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(raw_args)
    repo_root = args.repo_root.resolve(strict=True)
    carrier = require_exact_checkout(repo_root, args.expected_carrier_git_sha)
    if not args.isolated_child:
        with tempfile.TemporaryDirectory(prefix="12-6-current-clean-pycache-") as pycache:
            command = [
                sys.executable,
                "-I",
                "-S",
                "-X",
                f"pycache_prefix={pycache}",
                "-c",
                ISOLATED_CHILD_BOOTSTRAP,
                str(repo_root),
                carrier,
                CARRIER_PATH,
                *raw_args,
                "--isolated-child",
            ]
            return subprocess.run(command, check=False).returncode
    executor = load_authenticated_executor(repo_root, carrier)
    receipt = execute_and_publish(args, executor)
    print(receipt["receipt_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
