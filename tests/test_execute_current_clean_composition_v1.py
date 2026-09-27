from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from tools import execute_current_clean_composition_v1 as cli


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _args(tmp_path: Path) -> argparse.Namespace:
    training = tmp_path / "training.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    handoff = tmp_path / "handoff.json"
    binding = tmp_path / "binding.json"
    manifest = tmp_path / "manifest.json"
    evidence = tmp_path / "evidence.json"
    _write_jsonl(
        training,
        [
            {
                "record_id": "training-1",
                "source_id": "s1",
                "source_family": "f1",
                "modality": "uk",
                "text": "Секретний сирий навчальний текст",
            }
        ],
    )
    _write_jsonl(
        evaluation,
        [
            {
                "record_id": "evaluation-1",
                "source_id": "e1",
                "source_family": "ef1",
                "modality": "en",
                "text": "SECRET RAW EVALUATION TEXT",
            }
        ],
    )
    for path in (handoff, binding, manifest, evidence):
        _write_json(path, {"placeholder": True})
    return argparse.Namespace(
        output_dir=tmp_path / "out",
        training_records_jsonl=training,
        evaluation_records_jsonl=evaluation,
        training_handoff_json=handoff,
        base_reserved_binding_json=binding,
        eval647_manifest_json=manifest,
        eval647_materialization_evidence_json=evidence,
        expected_base_reserved_binding_identity_sha256="1" * 64,
        expected_composed_reserved_binding_identity_sha256="2" * 64,
        expected_eval647_materialization_evidence_identity_sha256="3" * 64,
        expected_eval647_object_set_identity_sha256="4" * 64,
        expected_inventory_identity_sha256="5" * 64,
        expected_survivor_authority_sha256="6" * 64,
        expected_training_handoff_identity_sha256="7" * 64,
        expected_selection_validation_identity_sha256="8" * 64,
        expected_final_test_identity_sha256="9" * 64,
    )


def _executor(*args, **kwargs):
    assert args[0][0]["text"] == "Секретний сирий навчальний текст"
    assert args[1][0]["text"] == "SECRET RAW EVALUATION TEXT"
    composition = {
        "receipt_identity_sha256": "a" * 64,
        "durable_evidence_hash_only": True,
    }
    report = {"report_sha256": "b" * 64, "hash_only_evidence": True}
    decontam = {"execution_identity_sha256": "c" * 64}
    eval647 = {"receipt_identity_sha256": "d" * 64}
    quality = {"execution_identity_sha256": "e" * 64}
    privacy = {"execution_identity_sha256": "f" * 64}
    return composition, report, decontam, eval647, quality, privacy


def test_execute_and_publish_writes_only_text_free_evidence(tmp_path: Path) -> None:
    args = _args(tmp_path)
    receipt = cli.execute_and_publish(args, _executor)

    assert receipt["receipt_identity_sha256"] == "a" * 64
    assert {path.name for path in args.output_dir.iterdir()} == set(
        cli.OUTPUT_FILES.values()
    )
    published = "\n".join(
        path.read_text(encoding="utf-8") for path in args.output_dir.iterdir()
    )
    assert "Секретний сирий навчальний текст" not in published
    assert "SECRET RAW EVALUATION TEXT" not in published


def test_execute_and_publish_refuses_existing_output(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite output"):
        cli.execute_and_publish(args, _executor)


def test_require_exact_checkout_authenticates_carrier_and_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    carrier_sha = "1" * 40
    files = {
        cli.CARRIER_PATH: b"carrier bytes",
        cli.MODULE_PATH: b"module bytes",
    }
    for repo_path, payload in files.items():
        path = tmp_path / repo_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def fake_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        if args[:3] == ("rev-parse", "--verify", "HEAD"):
            return subprocess.CompletedProcess(args, 0, carrier_sha + "\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(cli, "_git", fake_git)
    monkeypatch.setattr(
        cli,
        "_git_bytes",
        lambda repo_root, git_sha, repo_path: files[repo_path],
    )
    assert cli.require_exact_checkout(tmp_path, carrier_sha) == carrier_sha


def test_require_exact_checkout_rejects_dirty_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    carrier_sha = "2" * 40
    for repo_path in (cli.CARRIER_PATH, cli.MODULE_PATH):
        path = tmp_path / repo_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    calls = 0

    def fake_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if args[:3] == ("rev-parse", "--verify", "HEAD"):
            return subprocess.CompletedProcess(args, 0, carrier_sha + "\n", "")
        return subprocess.CompletedProcess(args, 1, "", "")

    monkeypatch.setattr(cli, "_git", fake_git)
    with pytest.raises(RuntimeError, match="working tree differs"):
        cli.require_exact_checkout(tmp_path, carrier_sha)
    assert calls >= 2


def test_authenticated_import_rejects_preloaded_external_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    carrier_sha = "3" * 40
    expected = tmp_path / cli.MODULE_PATH
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_text("def execute_current_clean_composition(): pass\n", encoding="utf-8")
    external = tmp_path / "external.py"
    external.write_text("pass\n", encoding="utf-8")
    fake = types.ModuleType("twelve_six.data.current_clean_execution_v1")
    fake.__file__ = str(external)
    module_name = "twelve_six.data.current_clean_execution_v1"
    monkeypatch.setitem(sys.modules, module_name, fake)

    with pytest.raises(RuntimeError, match="outside authenticated checkout"):
        cli.load_authenticated_executor(tmp_path, carrier_sha)
