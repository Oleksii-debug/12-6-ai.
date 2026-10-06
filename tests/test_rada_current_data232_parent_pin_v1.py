from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path

import pytest

import tools.validate_d03_rada_current_data232_parent_pin_v1 as target


IDENTITY_FIELDS = {
    logical: identity_field
    for logical, (identity_field, _output) in target.LOGICAL_FILES.items()
}


def _payload(logical: str) -> bytes:
    identity_field = IDENTITY_FIELDS[logical]
    value = {
        "schema_version": f"synthetic-{logical}.v1",
        identity_field: target.sha256(f"{logical}-identity".encode()),
    }
    return target.canonical(value) + b"\n"


def _build(tmp_path: Path) -> tuple[Path, Path, dict]:
    members = {
        logical: f"pass-a/{logical}.json"
        for logical in target.LOGICAL_FILES
    }
    payloads = {logical: _payload(logical) for logical in target.LOGICAL_FILES}
    artifact = tmp_path / "artifact.zip"
    with zipfile.ZipFile(artifact, "w", compression=zipfile.ZIP_STORED) as archive:
        for logical in sorted(members):
            archive.writestr(members[logical], payloads[logical])
        archive.writestr("pass-a/auxiliary.json", b'{"ok":true}\n')

    allowed = sorted([*members.values(), "pass-a/auxiliary.json"])
    files = {}
    for logical, member in members.items():
        document = json.loads(payloads[logical].decode("utf-8"))
        files[logical] = {
            "member": member,
            "file_sha256": target.sha256(payloads[logical]),
            "identity_sha256": document[IDENTITY_FIELDS[logical]],
        }
    pin = {
        "schema_version": target.SCHEMA,
        "parent_execution_head_sha": target.PARENT_EXECUTION_HEAD,
        "workflow_run_id": 456,
        "artifact_id": 123,
        "artifact_name": target.EXPECTED_ARTIFACT_NAME,
        "artifact_zip_sha256": target.sha256(artifact.read_bytes()),
        "allowed_members": allowed,
        "files": files,
        "qualification": {
            "fresh_execution_count": 2,
            "independent_runner_jobs": True,
            "two_fresh_processes_byte_identical": True,
            "durable_evidence_hash_only": True,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "tokenizer_fit_authorized": False,
            "training_executed": False,
            "learned_weights_created": False,
            "paid_compute_used": False,
            "scale_promotion_authorized": False,
        },
    }
    pin_path = tmp_path / "pin.json"
    pin_path.write_bytes(target.canonical(pin) + b"\n")
    return pin_path, artifact, pin


def test_validate_and_extract_exact_closed_world_artifact(tmp_path: Path) -> None:
    pin_path, artifact, pin = _build(tmp_path)
    assert target.load_pin(pin_path) == pin
    output = tmp_path / "out"
    manifest = target.extract(pin_path, artifact, output)

    assert manifest["artifact_id"] == 123
    assert manifest["artifact_zip_sha256"] == pin["artifact_zip_sha256"]
    assert sorted(path.name for path in output.iterdir()) == sorted(
        [
            "validated-parent-manifest.json",
            *[item[1] for item in target.LOGICAL_FILES.values()],
        ]
    )
    for logical, (_identity_field, output_name) in target.LOGICAL_FILES.items():
        assert (output / output_name).read_bytes() == _payload(logical)


def test_rejects_outer_zip_hash_drift(tmp_path: Path) -> None:
    pin_path, artifact, _pin = _build(tmp_path)
    artifact.write_bytes(artifact.read_bytes() + b"x")
    with pytest.raises(target.ParentPinError, match="artifact ZIP SHA-256 drift"):
        target.extract(pin_path, artifact, tmp_path / "out")


def test_rejects_unpinned_extra_zip_member(tmp_path: Path) -> None:
    pin_path, artifact, pin = _build(tmp_path)
    modified = tmp_path / "modified.zip"
    with zipfile.ZipFile(artifact, "r") as source, zipfile.ZipFile(
        modified, "w", compression=zipfile.ZIP_STORED
    ) as destination:
        for info in source.infolist():
            destination.writestr(info.filename, source.read(info.filename))
        destination.writestr("unexpected.json", b"{}\n")

    pin["artifact_zip_sha256"] = target.sha256(modified.read_bytes())
    pin_path.write_bytes(target.canonical(pin) + b"\n")
    with pytest.raises(target.ParentPinError, match="artifact ZIP member set drift"):
        target.extract(pin_path, modified, tmp_path / "out")


@pytest.mark.parametrize(
    "unsafe",
    [
        "../escape.json",
        "/absolute.json",
        "a/../../escape.json",
        "a\\windows.json",
    ],
)
def test_pin_rejects_unsafe_member_paths(tmp_path: Path, unsafe: str) -> None:
    pin_path, _artifact, pin = _build(tmp_path)
    pin["allowed_members"][0] = unsafe
    pin["allowed_members"].sort()
    pin_path.write_bytes(target.canonical(pin) + b"\n")
    with pytest.raises(target.ParentPinError, match="unsafe|relative|forward slashes"):
        target.load_pin(pin_path)


def test_rejects_file_hash_drift_even_with_pinned_outer_zip(tmp_path: Path) -> None:
    pin_path, artifact, pin = _build(tmp_path)
    pin["files"]["result"]["file_sha256"] = "0" * 64
    pin_path.write_bytes(target.canonical(pin) + b"\n")
    with pytest.raises(target.ParentPinError, match="result file SHA-256 drift"):
        target.extract(pin_path, artifact, tmp_path / "out")


def test_rejects_identity_drift_even_with_exact_file_hash(tmp_path: Path) -> None:
    pin_path, artifact, pin = _build(tmp_path)
    pin["files"]["proof"]["identity_sha256"] = "0" * 64
    pin_path.write_bytes(target.canonical(pin) + b"\n")
    with pytest.raises(target.ParentPinError, match="proof identity drift"):
        target.extract(pin_path, artifact, tmp_path / "out")


def test_rejects_weakened_qualification_boundary(tmp_path: Path) -> None:
    pin_path, _artifact, pin = _build(tmp_path)
    pin["qualification"]["independent_runner_jobs"] = False
    pin_path.write_bytes(target.canonical(pin) + b"\n")
    with pytest.raises(target.ParentPinError, match="qualification boundary drift"):
        target.load_pin(pin_path)


def test_extract_refuses_nonempty_destination(tmp_path: Path) -> None:
    pin_path, artifact, _pin = _build(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "stale.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(target.ParentPinError, match="output directory must be empty"):
        target.extract(pin_path, artifact, output)
