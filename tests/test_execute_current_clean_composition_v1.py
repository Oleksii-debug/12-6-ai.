from __future__ import annotations

import argparse
import hashlib
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


def _args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> argparse.Namespace:
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
    monkeypatch.setattr(
        cli,
        "PRODUCTION_TRAINING_RECORDS_SHA256",
        hashlib.sha256(training.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        cli,
        "PRODUCTION_TRAINING_HANDOFF_SHA256",
        hashlib.sha256(handoff.read_bytes()).hexdigest(),
    )
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
    survivor_records = [
        {
            "record_id": "survivor-1",
            "source_id": "s1",
            "family": "f1",
            "modality": "uk",
            "normalized_payload": "Дозволений матеріалізований текст",
        }
    ]
    survivor_raw = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in survivor_records
    ).encode("utf-8")
    survivor_inventory = cli._recompute_survivor_inventory(survivor_records)
    payload_bytes = survivor_inventory["total_payload_bytes"]
    composition = {
        "receipt_identity_sha256": "a" * 64,
        "durable_evidence_hash_only": True,
        "survivor_jsonl_sha256": hashlib.sha256(survivor_raw).hexdigest(),
        "survivor_record_inventory_digest_sha256": survivor_inventory[
            "record_inventory_digest_sha256"
        ],
        "survivor_payload_inventory_digest_sha256": survivor_inventory[
            "payload_inventory_digest_sha256"
        ],
        "survivor_records": 1,
        "survivor_payload_bytes": payload_bytes,
        "survivor_source_objects": 1,
    }
    report = {"report_sha256": "b" * 64, "hash_only_evidence": True}
    decontam = {"execution_identity_sha256": "c" * 64}
    eval647 = {"receipt_identity_sha256": "d" * 64}
    quality = {"execution_identity_sha256": "e" * 64}
    privacy = {"execution_identity_sha256": "f" * 64}
    return (
        composition,
        report,
        decontam,
        eval647,
        quality,
        privacy,
        survivor_records,
        survivor_inventory,
    )


def test_execute_and_publish_writes_only_text_free_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, monkeypatch)
    receipt = cli.execute_and_publish(args, _executor)

    assert receipt["receipt_identity_sha256"] == "a" * 64
    assert {path.name for path in args.output_dir.iterdir()} == {
        *cli.OUTPUT_FILES.values(),
        cli.SURVIVOR_RECORDS_NAME,
    }
    evidence = "\n".join(
        (args.output_dir / filename).read_text(encoding="utf-8")
        for filename in cli.OUTPUT_FILES.values()
    )
    assert "Секретний сирий навчальний текст" not in evidence
    assert "SECRET RAW EVALUATION TEXT" not in evidence
    survivors = (args.output_dir / cli.SURVIVOR_RECORDS_NAME).read_text(
        encoding="utf-8"
    )
    assert "Дозволений матеріалізований текст" in survivors
    assert "SECRET RAW EVALUATION TEXT" not in survivors


def test_execute_and_publish_rejects_survivor_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, monkeypatch)

    def bad_executor(*args, **kwargs):
        result = list(_executor(*args, **kwargs))
        composition = dict(result[0])
        composition["survivor_jsonl_sha256"] = "0" * 64
        result[0] = composition
        return tuple(result)

    with pytest.raises(
        RuntimeError,
        match="survivor JSONL differs from execution receipt",
    ):
        cli.execute_and_publish(args, bad_executor)
    assert not args.output_dir.exists()


def test_execute_and_publish_rejects_inventory_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, monkeypatch)

    def bad_executor(*args, **kwargs):
        result = list(_executor(*args, **kwargs))
        inventory = dict(result[7])
        inventory["record_inventory_digest_sha256"] = "0" * 64
        result[7] = inventory
        return tuple(result)

    with pytest.raises(
        RuntimeError,
        match="differs from independently rebuilt inventory",
    ):
        cli.execute_and_publish(args, bad_executor)
    assert not args.output_dir.exists()


def test_execute_and_publish_rejects_receipt_inventory_root_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, monkeypatch)

    def bad_executor(*args, **kwargs):
        result = list(_executor(*args, **kwargs))
        composition = dict(result[0])
        composition["survivor_payload_inventory_digest_sha256"] = "0" * 64
        result[0] = composition
        return tuple(result)

    with pytest.raises(
        RuntimeError,
        match="survivor receipt inventory root drift",
    ):
        cli.execute_and_publish(args, bad_executor)
    assert not args.output_dir.exists()


def test_execute_and_publish_rejects_source_object_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, monkeypatch)

    def bad_executor(*args, **kwargs):
        result = list(_executor(*args, **kwargs))
        composition = dict(result[0])
        composition["survivor_source_objects"] = 2
        result[0] = composition
        return tuple(result)

    with pytest.raises(
        RuntimeError,
        match="survivor receipt/count publication drift",
    ):
        cli.execute_and_publish(args, bad_executor)
    assert not args.output_dir.exists()


def test_execute_and_publish_rejects_raw_training_transport_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, monkeypatch)
    args.training_records_jsonl.write_bytes(
        args.training_records_jsonl.read_bytes() + b"\n"
    )
    with pytest.raises(RuntimeError, match="training records raw file identity drift"):
        cli.execute_and_publish(args, _executor)
    assert not args.output_dir.exists()


def test_execute_and_publish_rejects_raw_handoff_transport_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, monkeypatch)
    original = json.loads(args.training_handoff_json.read_text(encoding="utf-8"))
    args.training_handoff_json.write_text(
        json.dumps(original, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="training handoff raw file identity drift"):
        cli.execute_and_publish(args, _executor)
    assert not args.output_dir.exists()


def test_execute_and_publish_refuses_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, monkeypatch)
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
        **{
            path: f"dependency:{path}".encode()
            for path in cli.AUTHENTICATED_DEPENDENCY_PATHS
        },
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
    for repo_path in (cli.CARRIER_PATH, cli.MODULE_PATH, *cli.AUTHENTICATED_DEPENDENCY_PATHS):
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



def test_require_exact_checkout_rejects_tampered_execution_dependency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    carrier_sha = "5" * 40
    files = {
        cli.CARRIER_PATH: b"carrier bytes",
        cli.MODULE_PATH: b"module bytes",
        **{
            path: f"dependency:{path}".encode()
            for path in cli.AUTHENTICATED_DEPENDENCY_PATHS
        },
    }
    for repo_path, payload in files.items():
        path = tmp_path / repo_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    tampered_path = cli.AUTHENTICATED_DEPENDENCY_PATHS[0]
    (tmp_path / tampered_path).write_bytes(b"tampered dependency")

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
    with pytest.raises(RuntimeError, match="physical bytes differ from authenticated Git bytes"):
        cli.require_exact_checkout(tmp_path, carrier_sha)


def test_authenticated_import_rejects_any_preloaded_project_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    carrier_sha = "3" * 40
    fake = types.ModuleType("twelve_six.data.injected")
    monkeypatch.setitem(sys.modules, "twelve_six.data.injected", fake)

    with pytest.raises(RuntimeError, match="preloaded before authenticated Git importer"):
        cli.load_authenticated_executor(tmp_path, carrier_sha)


def test_authenticated_git_finder_uses_verified_physical_source_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    carrier_sha = "a" * 40
    payload = b"VALUE = 'authenticated'\n"
    physical = tmp_path / "src" / "twelve_six" / "demo.py"
    physical.parent.mkdir(parents=True, exist_ok=True)
    physical.write_bytes(payload)
    sources = {
        "src/twelve_six/__init__.py": b"",
        "src/twelve_six/demo.py": payload,
    }
    monkeypatch.setattr(
        cli,
        "_git_bytes_optional",
        lambda repo_root, git_sha, repo_path: sources.get(repo_path),
    )
    finder = cli._AuthenticatedGitFinder(tmp_path, carrier_sha)
    spec = finder.find_spec("twelve_six.demo")
    assert spec is not None and spec.loader is finder
    module = types.ModuleType("twelve_six.demo")
    module.__spec__ = spec
    finder.exec_module(module)
    assert module.VALUE == "authenticated"
    assert module.__file__ == str(physical.resolve(strict=True))


def test_authenticated_git_finder_rejects_physical_source_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    carrier_sha = "c" * 40
    physical = tmp_path / "src" / "twelve_six" / "demo.py"
    physical.parent.mkdir(parents=True, exist_ok=True)
    physical.write_text("VALUE = 'tampered'\n", encoding="utf-8")
    sources = {
        "src/twelve_six/__init__.py": b"",
        "src/twelve_six/demo.py": b"VALUE = 'authenticated'\n",
    }
    monkeypatch.setattr(
        cli,
        "_git_bytes_optional",
        lambda repo_root, git_sha, repo_path: sources.get(repo_path),
    )
    finder = cli._AuthenticatedGitFinder(tmp_path, carrier_sha)
    module = types.ModuleType("twelve_six.demo")
    with pytest.raises(ImportError, match="physical source drift"):
        finder.exec_module(module)


def test_authenticated_git_finder_never_falls_back_to_physical_project_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli,
        "_git_bytes_optional",
        lambda repo_root, git_sha, repo_path: None,
    )
    finder = cli._AuthenticatedGitFinder(tmp_path, "b" * 40)
    with pytest.raises(ImportError, match="authenticated Git module is unavailable"):
        finder.find_spec("twelve_six.data.untrusted_fallback")


def test_main_reexecutes_in_isolated_child_before_project_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    carrier_sha = "4" * 40
    raw_args = [
        "--repo-root",
        str(tmp_path),
        "--expected-carrier-git-sha",
        carrier_sha,
        "--training-records-jsonl",
        str(tmp_path / "training.jsonl"),
        "--training-handoff-json",
        str(tmp_path / "handoff.json"),
        "--evaluation-records-jsonl",
        str(tmp_path / "evaluation.jsonl"),
        "--base-reserved-binding-json",
        str(tmp_path / "binding.json"),
        "--eval647-manifest-json",
        str(tmp_path / "manifest.json"),
        "--eval647-materialization-evidence-json",
        str(tmp_path / "evidence.json"),
        "--expected-base-reserved-binding-identity-sha256",
        "1" * 64,
        "--expected-composed-reserved-binding-identity-sha256",
        "2" * 64,
        "--expected-eval647-materialization-evidence-identity-sha256",
        "3" * 64,
        "--expected-eval647-object-set-identity-sha256",
        "4" * 64,
        "--expected-inventory-identity-sha256",
        "5" * 64,
        "--expected-survivor-authority-sha256",
        "6" * 64,
        "--expected-training-handoff-identity-sha256",
        "7" * 64,
        "--expected-selection-validation-identity-sha256",
        "8" * 64,
        "--expected-final-test-identity-sha256",
        "9" * 64,
        "--output-dir",
        str(tmp_path / "out"),
    ]
    monkeypatch.setattr(
        cli,
        "require_exact_checkout",
        lambda repo_root, expected: carrier_sha,
    )
    monkeypatch.setattr(
        cli,
        "load_authenticated_executor",
        lambda *args, **kwargs: pytest.fail("project import occurred in parent"),
    )
    observed: list[str] = []

    def fake_run(command, *, check):
        observed.extend(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli.main(raw_args) == 0
    assert observed[1:3] == ["-I", "-S"]
    assert "-X" in observed
    assert any(value.startswith("pycache_prefix=") for value in observed)
    assert "-c" in observed
    assert cli.ISOLATED_CHILD_BOOTSTRAP in observed
    assert cli.CARRIER_PATH in observed
    assert str(tmp_path / cli.CARRIER_PATH) not in observed
    assert "--isolated-child" in observed
