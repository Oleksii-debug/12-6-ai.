from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

import tools.prepare_data232_ephemeral_handoff_v1 as runner
import twelve_six.data.expanded_postdedup_inventory_v1 as canonical_inventory
from twelve_six.data.expanded_postdedup_inventory_v1 import (
    DATA526_RECORD_INVENTORY_SHA256,
    ZERO_TRUTH,
)

CARRIER_SHA = "a" * 40
UPSTREAM_SHA = "b" * 40
REPORT_SHA = "1" * 64
MATCHER_SHA = "2" * 64
SURVIVOR_SHA = "3" * 64


def _canonical(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _record(record_id: str, text: str) -> dict[str, object]:
    raw = text.encode("utf-8")
    return {
        "record_id": record_id,
        "source_id": f"source-{record_id}",
        "family": "rada-trees",
        "modality": "uk",
        "payload_sha256": _sha(raw),
        "payload_bytes": len(raw),
        "training_eligible": False,
        "evaluation_eligible": False,
    }


def _inventory(texts: dict[str, str]) -> dict[str, object]:
    records = [_record(record_id, text) for record_id, text in sorted(texts.items())]
    body: dict[str, object] = {
        "schema": "twelve-six.expanded-postdedup-inventory.v1",
        "input_report_sha256": REPORT_SHA,
        "input_matcher_report_sha256": MATCHER_SHA,
        "input_survivor_authority_sha256": SURVIVOR_SHA,
        "input_data526_record_inventory_sha256": DATA526_RECORD_INVENTORY_SHA256,
        "source_count": len(records),
        "record_count": len(records),
        "retained_payload_bytes": sum(int(row["payload_bytes"]) for row in records),
        "records": records,
        "truth_boundary": dict(ZERO_TRUTH),
    }
    result = deepcopy(body)
    result["inventory_identity_sha256"] = _sha(_canonical(body, newline=True))
    return result


def _write_inputs(
    tmp_path: Path,
    *,
    payload_rows: list[dict[str, str]] | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    texts = {
        "record-a": "Український canonical payload alpha",
        "record-b": "Український canonical payload beta",
    }
    inventory = _inventory(texts)
    inventory_path = tmp_path / "inventory.json"
    payload_path = tmp_path / "payload.jsonl"
    inventory_path.write_bytes(_canonical(inventory, newline=True))
    rows = payload_rows or [
        {"record_id": "record-b", "text": texts["record-b"]},
        {"record_id": "record-a", "text": texts["record-a"]},
    ]
    payload_path.write_bytes(b"".join(_canonical(row, newline=True) for row in rows))
    return inventory_path, payload_path, inventory


def _run(tmp_path: Path) -> tuple[dict[str, object], Path, dict[str, object]]:
    inventory_path, payload_path, inventory = _write_inputs(tmp_path)
    output = tmp_path / "handoff-workspace"
    receipt = runner.prepare_and_publish(
        inventory_path=inventory_path,
        payload_path=payload_path,
        output_dir=output,
        expected_inventory_identity_sha256=str(
            inventory["inventory_identity_sha256"]
        ),
        carrier_git_sha=CARRIER_SHA,
        upstream_git_sha=UPSTREAM_SHA,
    )
    return receipt, output, inventory


def test_end_to_end_workspace_matches_current_data232_consumer_contract(
    tmp_path: Path,
) -> None:
    receipt, output, inventory = _run(tmp_path)

    records = [
        json.loads(line)
        for line in (output / runner.TRAINING_RECORDS_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    handoff = json.loads(
        (output / runner.TRAINING_HANDOFF_NAME).read_text(encoding="utf-8")
    )
    stored_receipt = json.loads(
        (output / runner.RECEIPT_NAME).read_text(encoding="utf-8")
    )

    assert [row["record_id"] for row in records] == ["record-a", "record-b"]
    assert set(records[0]) == {
        "record_id",
        "source_id",
        "source_family",
        "modality",
        "text",
    }
    assert set(handoff) == {
        "schema_version",
        "postdedup_inventory_identity_sha256",
        "input_survivor_authority_sha256",
        "retained_source_count",
        "matcher_input_projection",
        "matcher_input_projection_sha256",
        "raw_text_persisted_in_evidence",
        "final_test_payload_accessed",
        "final_test_outcomes_accessed",
        "authorized_training_exposure",
        "handoff_identity_sha256",
    }
    assert handoff["schema_version"] == "12-6.postdedup-decontam-handoff.v1"
    assert handoff["postdedup_inventory_identity_sha256"] == inventory[
        "inventory_identity_sha256"
    ]
    assert handoff["input_survivor_authority_sha256"] == SURVIVOR_SHA
    assert handoff["retained_source_count"] == 2
    assert handoff["raw_text_persisted_in_evidence"] is False
    assert handoff["authorized_training_exposure"] == 0
    assert stored_receipt == receipt
    runner.verify_receipt(stored_receipt)

    durable = _canonical(stored_receipt)
    assert b"record-a" not in durable
    assert b"record-b" not in durable
    assert b"canonical payload" not in durable
    assert stored_receipt["record_ids_persisted_in_receipt"] is False
    assert stored_receipt["raw_text_persisted_in_receipt"] is False
    assert stored_receipt["authorized_optimized_target_exposure"] == 0
    assert stored_receipt["training_executed"] is False


def test_physical_input_and_output_hashes_are_exact(tmp_path: Path) -> None:
    inventory_path, payload_path, inventory = _write_inputs(tmp_path)
    output = tmp_path / "out"
    receipt = runner.prepare_and_publish(
        inventory_path=inventory_path,
        payload_path=payload_path,
        output_dir=output,
        expected_inventory_identity_sha256=str(
            inventory["inventory_identity_sha256"]
        ),
        carrier_git_sha=CARRIER_SHA,
        upstream_git_sha=UPSTREAM_SHA,
    )
    assert receipt["input_files_sha256"] == {
        "payload_rows_jsonl": _sha(payload_path.read_bytes()),
        "retained_inventory_json": _sha(inventory_path.read_bytes()),
    }
    assert receipt["output_files_sha256"] == {
        runner.TRAINING_HANDOFF_NAME: _sha(
            (output / runner.TRAINING_HANDOFF_NAME).read_bytes()
        ),
        runner.TRAINING_RECORDS_NAME: _sha(
            (output / runner.TRAINING_RECORDS_NAME).read_bytes()
        ),
    }


def test_wrong_independent_inventory_identity_publishes_nothing(
    tmp_path: Path,
) -> None:
    inventory_path, payload_path, _ = _write_inputs(tmp_path)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="retained inventory identity mismatch"):
        runner.prepare_and_publish(
            inventory_path=inventory_path,
            payload_path=payload_path,
            output_dir=output,
            expected_inventory_identity_sha256="f" * 64,
            carrier_git_sha=CARRIER_SHA,
            upstream_git_sha=UPSTREAM_SHA,
        )
    assert not output.exists()


@pytest.mark.parametrize("kind", ["tampered", "missing", "extra", "duplicate"])
def test_payload_coverage_or_identity_failure_publishes_nothing(
    tmp_path: Path,
    kind: str,
) -> None:
    texts = {
        "record-a": "Український canonical payload alpha",
        "record-b": "Український canonical payload beta",
    }
    rows = [
        {"record_id": "record-a", "text": texts["record-a"]},
        {"record_id": "record-b", "text": texts["record-b"]},
    ]
    if kind == "tampered":
        rows[0]["text"] += "!"
    elif kind == "missing":
        rows.pop()
    elif kind == "extra":
        rows.append({"record_id": "record-extra", "text": "unexpected"})
    else:
        rows.append(dict(rows[0]))
    inventory_path, payload_path, inventory = _write_inputs(
        tmp_path,
        payload_rows=rows,
    )
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        runner.prepare_and_publish(
            inventory_path=inventory_path,
            payload_path=payload_path,
            output_dir=output,
            expected_inventory_identity_sha256=str(
                inventory["inventory_identity_sha256"]
            ),
            carrier_git_sha=CARRIER_SHA,
            upstream_git_sha=UPSTREAM_SHA,
        )
    assert not output.exists()


def test_existing_workspace_is_never_overwritten(tmp_path: Path) -> None:
    inventory_path, payload_path, inventory = _write_inputs(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner.prepare_and_publish(
            inventory_path=inventory_path,
            payload_path=payload_path,
            output_dir=output,
            expected_inventory_identity_sha256=str(
                inventory["inventory_identity_sha256"]
            ),
            carrier_git_sha=CARRIER_SHA,
            upstream_git_sha=UPSTREAM_SHA,
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_duplicate_json_keys_fail_closed_before_publication(tmp_path: Path) -> None:
    _, _, inventory = _write_inputs(tmp_path)
    inventory_path = tmp_path / "duplicate-inventory.json"
    inventory_path.write_text('{"schema":"x","schema":"y"}\n', encoding="utf-8")
    payload_path = tmp_path / "payload.jsonl"
    payload_path.write_text(
        '{"record_id":"record-a","text":"x"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner.prepare_and_publish(
            inventory_path=inventory_path,
            payload_path=payload_path,
            output_dir=output,
            expected_inventory_identity_sha256=str(
                inventory["inventory_identity_sha256"]
            ),
            carrier_git_sha=CARRIER_SHA,
            upstream_git_sha=UPSTREAM_SHA,
        )
    assert not output.exists()


def test_receipt_tamper_is_rejected(tmp_path: Path) -> None:
    receipt, _, _ = _run(tmp_path)
    tampered = deepcopy(receipt)
    tampered["training_executed"] = True
    with pytest.raises(ValueError):
        runner.verify_receipt(tampered)


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")


def test_exact_checkout_binds_upstream_module_and_clean_tree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    module = repo / runner.UPSTREAM_MODULE
    module.parent.mkdir(parents=True)
    module.write_text("authority = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "upstream")
    upstream = _git(repo, "rev-parse", "HEAD")

    (repo / "carrier.txt").write_text("carrier\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "carrier")
    carrier = _git(repo, "rev-parse", "HEAD")
    assert runner.require_exact_checkout(
        repo,
        expected_carrier_git_sha=carrier,
        expected_retained_inventory_git_sha=upstream,
    ) == (carrier, upstream)

    module.write_text("authority = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked working tree"):
        runner.require_exact_checkout(
            repo,
            expected_carrier_git_sha=carrier,
            expected_retained_inventory_git_sha=upstream,
        )


def test_verified_loader_ignores_preloaded_canonical_callable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    module = repo / runner.UPSTREAM_MODULE
    module.parent.mkdir(parents=True)
    module.write_text(
        "def prepare_ephemeral_data232_rows(*args, **kwargs):\n"
        "    return ([{'origin': 'fresh'}], {'origin': 'fresh'})\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "upstream")
    upstream = _git(repo, "rev-parse", "HEAD")

    def poisoned(*args: object, **kwargs: object) -> object:
        raise AssertionError("preloaded canonical callable must not execute")

    monkeypatch.setattr(
        canonical_inventory,
        "prepare_ephemeral_data232_rows",
        poisoned,
    )
    fresh = runner.load_verified_prepare_rows(repo, upstream)
    rows, handoff = fresh({}, [], expected_inventory_identity_sha256="0" * 64)
    assert rows == [{"origin": "fresh"}]
    assert handoff == {"origin": "fresh"}


def test_copied_runner_cannot_borrow_clean_repo_authority(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    module = repo / runner.UPSTREAM_MODULE
    module.parent.mkdir(parents=True)
    module.write_text(
        "def prepare_ephemeral_data232_rows(*args, **kwargs):\n"
        "    return ([], {})\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "upstream")
    upstream = _git(repo, "rev-parse", "HEAD")

    carrier_path = repo / runner.CARRIER_MODULE
    carrier_path.parent.mkdir(parents=True)
    carrier_path.write_text(
        Path(runner.__file__).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "carrier")
    carrier = _git(repo, "rev-parse", "HEAD")
    assert runner.require_executing_carrier(
        repo,
        expected_carrier_git_sha=carrier,
        actual_path=carrier_path,
    ) == carrier

    copied = tmp_path / "copied-runner.py"
    copied.write_bytes(carrier_path.read_bytes())
    output = tmp_path / "must-not-publish"
    completed = subprocess.run(
        [
            sys.executable,
            str(copied),
            "--repo-root",
            str(repo),
            "--inventory-json",
            str(tmp_path / "missing-inventory.json"),
            "--payload-jsonl",
            str(tmp_path / "missing-payload.jsonl"),
            "--output-dir",
            str(output),
            "--expected-inventory-identity-sha256",
            "0" * 64,
            "--expected-carrier-git-sha",
            carrier,
            "--expected-retained-inventory-git-sha",
            upstream,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert (
        "executing carrier path is not the authenticated repository carrier"
        in completed.stderr
    )
    assert not output.exists()
