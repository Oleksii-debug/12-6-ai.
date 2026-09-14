from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import tools.run_current_reserved_decontamination_v1 as runner
from twelve_six.data.current_reserved_decontamination_v1 import (
    build_reserved_payload_binding,
)

INVENTORY = "1" * 64
SURVIVOR = "2" * 64
SELECTION = "3" * 64
FINAL = "4" * 64
IMPLEMENTATION_SHA = "a" * 40


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _row(record_id: str, text: str, *, source: str, family: str) -> dict[str, str]:
    return {
        "record_id": record_id,
        "source_id": source,
        "source_family": family,
        "modality": "en",
        "text": text,
    }


def _projection(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    projection: list[dict[str, object]] = []
    for row in rows:
        raw = row["text"].encode("utf-8")
        projection.append(
            {
                "record_id": row["record_id"],
                "source_id": row["source_id"],
                "source_family": row["source_family"],
                "modality": row["modality"],
                "text_sha256": _sha(raw),
                "text_utf8_bytes": len(raw),
            }
        )
    return sorted(projection, key=lambda item: str(item["record_id"]))


def _training_handoff(rows: list[dict[str, str]]) -> dict[str, object]:
    projection = _projection(rows)
    core: dict[str, object] = {
        "schema_version": "12-6.postdedup-decontam-handoff.v1",
        "postdedup_inventory_identity_sha256": INVENTORY,
        "input_survivor_authority_sha256": SURVIVOR,
        "retained_source_count": len(rows),
        "matcher_input_projection": projection,
        "matcher_input_projection_sha256": _sha(_canonical_bytes(projection)),
        "raw_text_persisted_in_evidence": False,
        "final_test_payload_accessed": False,
        "final_test_outcomes_accessed": False,
        "authorized_training_exposure": 0,
    }
    core["handoff_identity_sha256"] = _sha(_canonical_bytes(core))
    return core


def _member(row: dict[str, str]) -> dict[str, object]:
    raw = row["text"].encode("utf-8")
    return {
        "record_id": row["record_id"],
        "source_id": row["source_id"],
        "source_family": row["source_family"],
        "modality": row["modality"],
        "content_sha256": _sha(raw),
        "utf8_bytes": len(raw),
        "training_prohibited": True,
        "outcomes_included": False,
    }


def _reserved_binding(
    selection_rows: list[dict[str, str]], final_rows: list[dict[str, str]]
) -> dict[str, object]:
    return build_reserved_payload_binding(
        [
            {
                "authority_id": "selection-authority",
                "identity_sha256": SELECTION,
                "role": "selection_validation",
                "source_sha": "b" * 40,
                "source_membership_identity_sha256": "5" * 64,
                "members": [_member(row) for row in selection_rows],
            },
            {
                "authority_id": "final-authority",
                "identity_sha256": FINAL,
                "role": "final_test",
                "source_sha": "c" * 40,
                "source_membership_identity_sha256": "6" * 64,
                "members": [_member(row) for row in final_rows],
            },
        ]
    )


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(runner._file_bytes(value))


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    payload = b"".join(runner._file_bytes(row) for row in rows)
    path.write_bytes(payload)


def _fixture(tmp_path: Path) -> tuple[list[str], dict[str, str]]:
    training = [
        _row(
            "train-1",
            "independent training material about ocean currents",
            source="train-source",
            family="train-family",
        )
    ]
    selection = [
        _row(
            "selection-1",
            "reserved selection material about authentication",
            source="selection-source",
            family="selection-family",
        )
    ]
    final = [
        _row(
            "final-1",
            "sealed final material about astronomy",
            source="final-source",
            family="final-family",
        )
    ]
    handoff = _training_handoff(training)
    binding = _reserved_binding(selection, final)

    training_path = tmp_path / "training.jsonl"
    handoff_path = tmp_path / "handoff.json"
    evaluation_path = tmp_path / "evaluation.jsonl"
    binding_path = tmp_path / "reserved-binding.json"
    output_dir = tmp_path / "bundle"
    _write_jsonl(training_path, training)
    _write_json(handoff_path, handoff)
    _write_jsonl(evaluation_path, [*selection, *final])
    _write_json(binding_path, binding)

    argv = [
        "--repo-root",
        str(tmp_path),
        "--training-records-jsonl",
        str(training_path),
        "--training-handoff-json",
        str(handoff_path),
        "--evaluation-records-jsonl",
        str(evaluation_path),
        "--reserved-binding-json",
        str(binding_path),
        "--output-dir",
        str(output_dir),
        "--expected-inventory-identity-sha256",
        INVENTORY,
        "--expected-survivor-authority-sha256",
        SURVIVOR,
        "--expected-training-handoff-identity-sha256",
        str(handoff["handoff_identity_sha256"]),
        "--expected-reserved-binding-identity-sha256",
        str(binding["binding_identity_sha256"]),
        "--expected-selection-validation-identity-sha256",
        SELECTION,
        "--expected-final-test-identity-sha256",
        FINAL,
        "--expected-decontamination-implementation-git-sha",
        IMPLEMENTATION_SHA,
    ]
    secrets = {
        "training_text": training[0]["text"],
        "selection_text": selection[0]["text"],
        "final_text": final[0]["text"],
        "training_id": training[0]["record_id"],
        "selection_id": selection[0]["record_id"],
        "final_id": final[0]["record_id"],
    }
    return argv, secrets


def test_main_publishes_atomic_hash_only_bundle(tmp_path: Path, monkeypatch):
    argv, secrets = _fixture(tmp_path)
    monkeypatch.setattr(runner, "require_exact_checkout", lambda *_: IMPLEMENTATION_SHA)

    assert runner.main(argv) == 0

    bundle = tmp_path / "bundle"
    assert sorted(path.name for path in bundle.iterdir()) == [
        runner.REPORT_NAME,
        runner.EVIDENCE_NAME,
        runner.RECEIPT_NAME,
    ]
    report = json.loads((bundle / runner.REPORT_NAME).read_text(encoding="utf-8"))
    evidence = json.loads((bundle / runner.EVIDENCE_NAME).read_text(encoding="utf-8"))
    receipt = json.loads((bundle / runner.RECEIPT_NAME).read_text(encoding="utf-8"))
    runner.verify_run_receipt(receipt, report, evidence)
    assert receipt["decontamination_implementation_git_sha"] == IMPLEMENTATION_SHA
    assert receipt["final_test_payload_accessed_for_decontamination"] is True
    assert receipt["final_test_outcomes_read"] is False
    assert receipt["authorized_training_exposure"] == 0
    assert receipt["tokenizer_fit_authorized"] is False
    assert receipt["training_executed"] is False

    durable = json.dumps(
        {"report": report, "evidence": evidence, "receipt": receipt},
        sort_keys=True,
    )
    for secret in secrets.values():
        assert secret not in durable


def test_wrong_head_fails_before_any_payload_read(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "bundle"

    def reject(*_args, **_kwargs):
        raise RuntimeError("head mismatch")

    monkeypatch.setattr(runner, "require_exact_checkout", reject)
    with pytest.raises(RuntimeError, match="head mismatch"):
        runner.main(
            [
                "--repo-root",
                str(tmp_path),
                "--training-records-jsonl",
                str(tmp_path / "missing-training.jsonl"),
                "--training-handoff-json",
                str(tmp_path / "missing-handoff.json"),
                "--evaluation-records-jsonl",
                str(tmp_path / "missing-evaluation.jsonl"),
                "--reserved-binding-json",
                str(tmp_path / "missing-binding.json"),
                "--output-dir",
                str(output_dir),
                "--expected-inventory-identity-sha256",
                INVENTORY,
                "--expected-survivor-authority-sha256",
                SURVIVOR,
                "--expected-training-handoff-identity-sha256",
                "7" * 64,
                "--expected-reserved-binding-identity-sha256",
                "8" * 64,
                "--expected-selection-validation-identity-sha256",
                SELECTION,
                "--expected-final-test-identity-sha256",
                FINAL,
                "--expected-decontamination-implementation-git-sha",
                IMPLEMENTATION_SHA,
            ]
        )
    assert not output_dir.exists()


def test_payload_identity_failure_leaves_bundle_unpublished(
    tmp_path: Path, monkeypatch
):
    argv, _ = _fixture(tmp_path)
    monkeypatch.setattr(runner, "require_exact_checkout", lambda *_: IMPLEMENTATION_SHA)
    training_path = tmp_path / "training.jsonl"
    tampered = _row(
        "train-1",
        "mutated training material",
        source="train-source",
        family="train-family",
    )
    _write_jsonl(training_path, [tampered])

    with pytest.raises(Exception, match="matcher projection"):
        runner.main(argv)
    assert not (tmp_path / "bundle").exists()


def test_existing_bundle_is_rejected_before_payload_reads(tmp_path: Path, monkeypatch):
    argv, _ = _fixture(tmp_path)
    (tmp_path / "bundle").mkdir()
    monkeypatch.setattr(runner, "require_exact_checkout", lambda *_: IMPLEMENTATION_SHA)
    (tmp_path / "training.jsonl").unlink()

    with pytest.raises(FileExistsError, match="output bundle already exists"):
        runner.main(argv)


def test_receipt_rejects_durable_report_tamper(tmp_path: Path, monkeypatch):
    argv, _ = _fixture(tmp_path)
    monkeypatch.setattr(runner, "require_exact_checkout", lambda *_: IMPLEMENTATION_SHA)
    runner.main(argv)
    bundle = tmp_path / "bundle"
    report = json.loads((bundle / runner.REPORT_NAME).read_text(encoding="utf-8"))
    evidence = json.loads((bundle / runner.EVIDENCE_NAME).read_text(encoding="utf-8"))
    receipt = json.loads((bundle / runner.RECEIPT_NAME).read_text(encoding="utf-8"))
    report["status"] = "TAMPERED"

    with pytest.raises(ValueError, match="output-file hash mismatch"):
        runner.verify_run_receipt(receipt, report, evidence)


def test_exact_checkout_rejects_tracked_worktree_drift(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert runner.require_exact_checkout(tmp_path, head) == head

    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked working tree"):
        runner.require_exact_checkout(tmp_path, head)
