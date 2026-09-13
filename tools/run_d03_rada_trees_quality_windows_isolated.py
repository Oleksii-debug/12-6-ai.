#!/usr/bin/env python3
"""Execute the authoritative Rada quality/privacy replay from exact authenticated source bytes.

This launcher is intentionally stdlib-only. It does not define quality/privacy
policy. It verifies the exact behavior-bearing project source closure, stages
those already-verified bytes into a fresh temporary tree, verifies the staged
tree again immediately before execution, and invokes the incumbent authoritative
materializer in a fresh ``python -I -S`` process whose import roots are only the
authenticated tree plus the interpreter standard-library roots.

The durable output is a hash-safe execution-closure receipt. It grants zero
corpus/training authority by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "12-6.d03-rada-trees-isolated-source-closure.v1"
ENTRYPOINT = "tools/materialize_d03_rada_trees_quality_windows_authoritative.py"

# Exact source closure composed on the current #1017 lineage after the
# 58711afe... main bridge. Only stdlib dependencies exist outside this tree.
AUTHENTICATED_SOURCE_CLOSURE: dict[str, str] = {
    "tools/materialize_d03_rada_trees_quality_windows_authoritative.py": "e6789f82301dc9e559a431a6b776097bef28d393",
    "tools/materialize_d03_rada_trees_quality_windows.py": "5958e4b859902b60e06ba6a4019974ee1088e093",
    "src/twelve_six/__init__.py": "5433166c507bc845bd12d8d5c4145f1fbedda204",
    "src/twelve_six/data/document_quality.py": "b1461263034b4fb9510479b20c9697e22faa5f97",
    "src/twelve_six/data/quality_granularity.py": "513523b86824c423cad97352b3abb3d1241531b9",
    "src/twelve_six/data/privacy_filter_v3.py": "bcc5938395724f6728ab212f98b39f2334b0f37d",
}
EXPECTED_CANDIDATE_SHA256 = "778be40135facef3b86168643673cb1df819e81a376679ac8418130f3104f59a"
EXPECTED_UPSTREAM_REPORT_SHA256 = "83b2cc636aa4cf9a754cd55534035c44782722d2b457755c00f6780112ebfffa"

_CHILD_BOOTSTRAP = r"""
import runpy
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
entry = (root / sys.argv[2]).resolve(strict=True)
src = (root / "src").resolve(strict=True)
tools = (root / "tools").resolve(strict=True)
if root not in entry.parents:
    raise SystemExit("entrypoint escaped authenticated root")
sys.path.insert(0, str(src))
sys.path.insert(0, str(tools))
sys.argv = [str(entry), *sys.argv[3:]]
runpy.run_path(str(entry), run_name="__main__")
""".strip()


class IsolatedAuthorityError(RuntimeError):
    """Raised when authenticated execution cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IsolatedAuthorityError(message)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha_bytes(payload: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _read_bound_source(path: Path, expected_blob: str) -> bytes:
    _require(path.is_file() and not path.is_symlink(), f"source is not a regular file: {path}")
    payload = path.read_bytes()
    _require(
        _git_blob_sha_bytes(payload) == expected_blob,
        f"source Git blob drift: {path}",
    )
    return payload


def _collect_authenticated_sources(
    root: Path, expected: Mapping[str, str]
) -> dict[str, bytes]:
    root = root.resolve(strict=True)
    result: dict[str, bytes] = {}
    for relative, expected_blob in sorted(expected.items()):
        candidate = (root / relative).resolve(strict=True)
        _require(root in candidate.parents, f"source escaped repository root: {relative}")
        result[relative] = _read_bound_source(candidate, expected_blob)
    return result


def _stage_authenticated_tree(
    destination: Path,
    payloads: Mapping[str, bytes],
) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for relative, payload in sorted(payloads.items()):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def _verify_staged_tree(
    root: Path,
    expected: Mapping[str, str],
) -> None:
    root = root.resolve(strict=True)
    for relative, expected_blob in sorted(expected.items()):
        candidate = (root / relative).resolve(strict=True)
        _require(root in candidate.parents, f"staged source escaped root: {relative}")
        _read_bound_source(candidate, expected_blob)


def _isolated_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    # Preserve the runner/toolchain environment while explicitly dropping Python
    # import-injection variables. ``-I`` also implies ``-E -s -P`` and ``-S``
    # prevents ``site``/``sitecustomize``/``usercustomize`` execution entirely.
    environment = dict(os.environ)
    if extra:
        environment.update(extra)
    for key in list(environment):
        upper = key.upper()
        if upper in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"}:
            environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _run_isolated_entrypoint(
    authenticated_root: Path,
    entrypoint: str,
    argv: list[str],
    *,
    cwd: Path,
    extra_env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    authenticated_root = authenticated_root.resolve(strict=True)
    cwd.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-I",
        "-S",
        "-c",
        _CHILD_BOOTSTRAP,
        str(authenticated_root),
        entrypoint,
        *argv,
    ]
    return subprocess.run(
        command,
        cwd=cwd,
        env=_isolated_environment(extra_env),
        text=True,
        capture_output=True,
        check=False,
    )


def _runtime_identity() -> dict[str, Any]:
    executable = Path(sys.executable).resolve(strict=True)
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "cache_tag": sys.implementation.cache_tag,
        "executable_sha256": _sha256_file(executable),
        "unicode_database_version": unicodedata.unidata_version,
        "isolated_flag": "-I",
        "no_site_flag": "-S",
        "ignore_environment": True,
        "no_user_site": True,
        "no_system_site_initialization": True,
        "safe_path": True,
    }


def _load_report(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "authority report missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IsolatedAuthorityError("cannot load authority report") from exc
    _require(type(value) is dict, "authority report root must be an object")
    return value


def execute(
    *,
    candidate: Path,
    upstream_report: Path,
    output: Path,
    report: Path,
    closure_receipt: Path,
    expected_launcher_git_blob_sha: str,
) -> dict[str, Any]:
    launcher = Path(__file__).resolve(strict=True)
    repo_root = launcher.parents[1]
    _require(
        _git_blob_sha_bytes(launcher.read_bytes()) == expected_launcher_git_blob_sha,
        "isolated launcher Git blob drift",
    )
    _require(not launcher.is_symlink(), "isolated launcher must not be a symlink")
    _require(
        expected_launcher_git_blob_sha
        not in set(AUTHENTICATED_SOURCE_CLOSURE.values()),
        "launcher blob unexpectedly aliases behavior closure",
    )
    for path, label in (
        (candidate, "candidate"),
        (upstream_report, "upstream report"),
    ):
        _require(path.is_file() and not path.is_symlink(), f"{label} must be a regular file")
    for path, label in (
        (output, "output"),
        (report, "report"),
        (closure_receipt, "closure receipt"),
    ):
        _require(not path.exists(), f"{label} already exists")
        _require(not path.is_symlink(), f"{label} must not be a symlink")
    _require(
        len(
            {
                candidate.resolve(strict=True),
                upstream_report.resolve(strict=True),
                output.resolve(strict=False),
                report.resolve(strict=False),
                closure_receipt.resolve(strict=False),
            }
        )
        == 5,
        "execution paths must be distinct",
    )

    source_payloads = _collect_authenticated_sources(
        repo_root, AUTHENTICATED_SOURCE_CLOSURE
    )

    with tempfile.TemporaryDirectory(prefix="rada-auth-source-") as temp_name:
        temp_root = Path(temp_name)
        authenticated_root = temp_root / "authenticated"
        work_dir = temp_root / "work"
        _stage_authenticated_tree(authenticated_root, source_payloads)
        # Mandatory second read immediately before process creation closes staged
        # source tamper between authentication and execution.
        _verify_staged_tree(authenticated_root, AUTHENTICATED_SOURCE_CLOSURE)

        child = _run_isolated_entrypoint(
            authenticated_root,
            ENTRYPOINT,
            [
                "--candidate-jsonl",
                str(candidate.resolve(strict=True)),
                "--upstream-handoff-report",
                str(upstream_report.resolve(strict=True)),
                "--expected-upstream-report-sha256",
                EXPECTED_UPSTREAM_REPORT_SHA256,
                "--output-jsonl",
                str(output.resolve(strict=False)),
                "--report",
                str(report.resolve(strict=False)),
            ],
            cwd=work_dir,
        )
        if child.stdout:
            print(child.stdout, end="")
        if child.stderr:
            print(child.stderr, end="", file=sys.stderr)
        _require(child.returncode == 0, f"isolated authoritative child failed: {child.returncode}")

    authority = _load_report(report)
    binding = authority.get("authority_binding")
    boundary = authority.get("claim_boundary")
    _require(type(binding) is dict, "authority binding missing")
    _require(type(boundary) is dict, "claim boundary missing")
    _require(
        authority.get("input_candidate_jsonl_sha256") == EXPECTED_CANDIDATE_SHA256,
        "child candidate identity drift",
    )
    _require(
        binding.get("upstream_handoff_report_sha256") == EXPECTED_UPSTREAM_REPORT_SHA256,
        "child upstream authority drift",
    )
    _require(
        authority.get("output_jsonl_sha256") == _sha256_file(output),
        "child output hash mismatch",
    )
    _require(boundary.get("training_authorized_bytes") == 0, "training credit widened")
    _require(
        boundary.get("unique_causal_loss_positions_authorized") == 0,
        "unique-loss credit widened",
    )
    _require(boundary.get("tokenizer_fit_authorized") is False, "tokenizer authority widened")
    _require(boundary.get("optimizer_updates") == 0, "optimizer authority widened")
    _require(boundary.get("model_training_executed") is False, "training boundary widened")
    _require(boundary.get("final_test_payload_accessed") is False, "final-test boundary widened")
    _require(boundary.get("paid_compute_used") is False, "paid-compute boundary widened")

    receipt_core = {
        "schema_version": SCHEMA,
        "classification": "AUTHENTICATED_ISOLATED_EXECUTION_ZERO_CREDIT",
        "execution_profile": "LOCAL_FREE",
        "launcher_git_blob_sha": expected_launcher_git_blob_sha,
        "authenticated_source_closure_git_blobs": dict(
            sorted(AUTHENTICATED_SOURCE_CLOSURE.items())
        ),
        "source_bytes_read_once_then_staged": True,
        "staged_tree_reverified_before_child": True,
        "child_fresh_process": True,
        "child_python_isolated_mode": True,
        "child_python_no_site_mode": True,
        "child_inherited_pythonpath": False,
        "child_authenticated_import_roots_only": True,
        "runtime": _runtime_identity(),
        "candidate_jsonl_sha256": authority["input_candidate_jsonl_sha256"],
        "upstream_handoff_report_sha256": binding["upstream_handoff_report_sha256"],
        "authority_report_sha256": authority["report_sha256"],
        "authority_report_file_sha256": _sha256_file(report),
        "survivor_jsonl_sha256": authority["output_jsonl_sha256"],
        "truth_boundary": {
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights": False,
            "external_llm_or_api_used_for_data_or_intelligence": False,
        },
    }
    receipt = {
        **receipt_core,
        "receipt_sha256": hashlib.sha256(_canonical_bytes(receipt_core)).hexdigest(),
    }
    closure_receipt.parent.mkdir(parents=True, exist_ok=True)
    closure_receipt.write_bytes(_canonical_bytes(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--upstream-handoff-report", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--closure-receipt", type=Path, required=True)
    parser.add_argument("--expected-launcher-git-blob-sha", required=True)
    args = parser.parse_args()
    try:
        receipt = execute(
            candidate=args.candidate_jsonl,
            upstream_report=args.upstream_handoff_report,
            output=args.output_jsonl,
            report=args.report,
            closure_receipt=args.closure_receipt,
            expected_launcher_git_blob_sha=args.expected_launcher_git_blob_sha,
        )
    except (IsolatedAuthorityError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print("RADA_AUTHENTICATED_ISOLATED_EXECUTION=PASS_ZERO_CREDIT")
    print("CLOSURE_RECEIPT_SHA256=" + receipt["receipt_sha256"])
    print("AUTHORITY_REPORT_SHA256=" + receipt["authority_report_sha256"])
    print("SURVIVOR_JSONL_SHA256=" + receipt["survivor_jsonl_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
