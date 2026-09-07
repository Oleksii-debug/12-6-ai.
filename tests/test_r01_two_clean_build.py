from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from twelve_six.two_clean_build import (
    CleanBuildError,
    compare_clean_builds,
    scan_build_root,
    validate_binding,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/research/r01_two_clean_build_binding_v1.json"
SHA40 = "a" * 40
SHA64 = "b" * 64


def _load_template() -> dict:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def _ready_binding() -> dict:
    data = _load_template()
    data["status"] = "READY_CANDIDATE"
    data["identities"] = {
        "source_git_sha": SHA40,
        "corpus_manifest_sha256": SHA64,
        "split_sha256": "c" * 64,
        "tokenizer_sha256": "d" * 64,
        "packing_sha256": "e" * 64,
        "build_config_sha256": "f" * 64,
    }
    return data


def _write_build(root: Path, *, reverse: bool = False) -> None:
    rows = [
        ("manifests/corpus.json", b'{"corpus":"exact"}\n'),
        ("packed/shard-000.bin", b"\x00\x01\x02\x03"),
        ("packed/shard-001.bin", b"\x04\x05\x06"),
    ]
    if reverse:
        rows.reverse()
    for rel, payload in rows:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def test_checked_in_template_is_valid_but_blocked() -> None:
    data = _load_template()
    assert validate_binding(data) == []
    result = compare_clean_builds(data, Path("missing-a"), Path("missing-b"))
    assert not result.identical
    assert result.blockers == ("binding_status_not_ready_candidate",)


def test_identical_independent_roots_produce_root_free_terminal_report(
    tmp_path: Path,
) -> None:
    root_a = tmp_path / "build-a"
    root_b = tmp_path / "build-b"
    _write_build(root_a)
    _write_build(root_b, reverse=True)

    result = compare_clean_builds(_ready_binding(), root_a, root_b)

    assert result.identical
    assert result.blockers == ()
    assert result.report is not None
    assert result.report["two_clean_builds_identical"] is True
    assert result.report["file_count"] == 3
    assert result.report["total_bytes"] == 26
    assert validate_report(result.report) == []
    serialized = json.dumps(result.report, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "build-a" not in serialized
    assert "build-b" not in serialized


def test_creation_order_and_mtime_do_not_change_build_identity(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_build(root_a)
    _write_build(root_b, reverse=True)
    for path in root_b.rglob("*"):
        if path.is_file():
            os.utime(path, (1_000_000, 1_000_000))

    tree_a = scan_build_root(root_a)
    tree_b = scan_build_root(root_b)
    assert tree_a == tree_b


def test_content_drift_fails_closed(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_build(root_a)
    _write_build(root_b)
    (root_b / "packed/shard-001.bin").write_bytes(b"changed")

    result = compare_clean_builds(_ready_binding(), root_a, root_b)
    assert not result.identical
    assert result.blockers == ("build_content_mismatch",)
    assert result.report is None


def test_path_set_drift_fails_closed(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_build(root_a)
    _write_build(root_b)
    (root_b / "packed/shard-001.bin").unlink()

    result = compare_clean_builds(_ready_binding(), root_a, root_b)
    assert not result.identical
    assert result.blockers == ("build_path_set_mismatch",)


def test_same_root_is_not_two_clean_builds(tmp_path: Path) -> None:
    root = tmp_path / "build"
    _write_build(root)
    result = compare_clean_builds(_ready_binding(), root, root)
    assert not result.identical
    assert result.blockers == ("clean_build_roots_must_be_distinct",)


def test_empty_roots_cannot_pass(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    result = compare_clean_builds(_ready_binding(), root_a, root_b)
    assert not result.identical
    assert result.blockers == ("build_root_empty",)


def test_zero_byte_only_roots_cannot_pass(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "empty.bin").write_bytes(b"")
    (root_b / "empty.bin").write_bytes(b"")
    result = compare_clean_builds(_ready_binding(), root_a, root_b)
    assert not result.identical
    assert result.blockers == ("build_root_zero_bytes",)


def test_symlink_in_build_root_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "payload.bin"
    target.write_bytes(b"payload")
    link = root / "alias.bin"
    try:
        link.symlink_to(target.name)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")

    with pytest.raises(CleanBuildError, match="symlink_forbidden"):
        scan_build_root(root)


def test_binding_identity_drift_fails_before_payload_scan(tmp_path: Path) -> None:
    data = _ready_binding()
    data["identities"]["packing_sha256"] = "not-a-hash"
    result = compare_clean_builds(data, tmp_path / "a", tmp_path / "b")
    assert not result.identical
    assert "packing_sha256_invalid" in result.blockers


def test_report_tamper_is_detected(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_build(root_a)
    _write_build(root_b)
    result = compare_clean_builds(_ready_binding(), root_a, root_b)
    assert result.report is not None

    tampered = copy.deepcopy(result.report)
    tampered["entries"][0]["bytes"] += 1
    errors = validate_report(tampered)
    assert "report_total_bytes_mismatch" in errors
    assert "report_build_tree_identity_mismatch" in errors
    assert "report_self_hash_mismatch" in errors


def test_truth_boundary_cannot_authorize_training_or_loss_ledger() -> None:
    for key in (
        "tokenizer_fit_executed_by_verifier",
        "final_test_accessed",
        "model_training_executed",
        "paid_compute_used",
        "unique_loss_ledger_authorized_by_verifier",
    ):
        data = _ready_binding()
        data["truth_boundary"][key] = True
        assert f"truth_boundary_{key}_must_be_false" in validate_binding(data)


def test_cli_writes_once_only_after_exact_match(tmp_path: Path) -> None:
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps(_ready_binding()), encoding="utf-8")
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_build(root_a)
    _write_build(root_b)
    output = tmp_path / "report.json"
    command = [
        sys.executable,
        str(ROOT / "tools/verify_r01_two_clean_build.py"),
        "--binding",
        str(binding),
        "--root-a",
        str(root_a),
        "--root-b",
        str(root_b),
        "--output",
        str(output),
    ]

    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr or first.stdout
    assert validate_report(json.loads(output.read_text(encoding="utf-8"))) == []

    second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert second.returncode == 2
    assert "refusing to overwrite existing output" in second.stdout
