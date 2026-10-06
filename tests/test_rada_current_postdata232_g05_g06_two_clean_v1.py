from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import tools.qualify_d03_rada_current_postdata232_g05_g06_v1 as target


def _sha(label: str) -> str:
    return target.sha256(label.encode("utf-8"))


def _self_hashed(core: dict, field: str) -> dict:
    return {**core, field: target.sha256(target.canonical(core))}


def _bundle(
    path: Path,
    *,
    execution_head: str = "c" * 40,
    parent_artifact_id: int = 123,
    parent_zip: str = "d" * 64,
) -> None:
    path.mkdir()
    quality = {"execution_identity_sha256": _sha("quality"), "records": []}
    privacy = {"execution_identity_sha256": _sha("privacy"), "records": []}
    inventory_rows = [
        {
            "record_id": "r1",
            "source_id": "s1",
            "family": "f1",
            "modality": "uk",
            "payload_sha256": _sha("payload-1"),
            "payload_bytes": 10,
        },
        {
            "record_id": "r2",
            "source_id": "s2",
            "family": "f2",
            "modality": "en",
            "payload_sha256": _sha("payload-2"),
            "payload_bytes": 10,
        },
    ]
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in inventory_rows
    ]
    inventory = {
        "schema_version": "12-6.data526-record-inventory.v1",
        "record_count": 2,
        "total_payload_bytes": 20,
        "record_inventory_digest_sha256": target.sha256(
            target.canonical(inventory_rows)
        ),
        "payload_inventory_digest_sha256": target.sha256(
            target.canonical(payload_projection)
        ),
        "records": inventory_rows,
    }
    quality_raw = target.canonical_line(quality)
    privacy_raw = target.canonical_line(privacy)
    inventory_raw = target.canonical_line(inventory)
    evidence_core = {
        "schema_version": target.EVIDENCE_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": execution_head,
        "parent": {
            "execution_head_sha": target.PARENT_EXECUTION_HEAD,
            "artifact_id": parent_artifact_id,
            "artifact_zip_sha256": parent_zip,
            "data232_execution_identity_sha256": _sha("data232-execution"),
            "two_fresh_data232_processes_byte_identical": True,
        },
        "g05": {
            "input_rows_sha256": _sha("g05-input"),
            "execution_identity_sha256": quality["execution_identity_sha256"],
        },
        "g06": {
            "input_rows_sha256": _sha("g06-input"),
            "execution_identity_sha256": privacy["execution_identity_sha256"],
            "exact_payload_collision_free": True,
            "unique_payload_count": 2,
            "payload_set_identity_sha256": _sha("payload-set"),
        },
        "durable_artifacts": {
            "g05_authority_file_sha256": target.sha256(quality_raw),
            "g06_authority_file_sha256": target.sha256(privacy_raw),
            "survivor_inventory_file_sha256": target.sha256(inventory_raw),
            "completion_marker": "POST_G05_G06_EVIDENCE_WRITTEN_LAST",
        },
        "survivor_inventory": {
            "record_payload_jsonl_sha256": _sha("payload-jsonl"),
            "record_count": 2,
            "total_payload_bytes": 20,
            "record_inventory_digest_sha256": inventory[
                "record_inventory_digest_sha256"
            ],
            "payload_inventory_digest_sha256": inventory[
                "payload_inventory_digest_sha256"
            ],
        },
        "content_boundary": dict(target.EXPECTED_CONTENT_BOUNDARY),
        "truth_boundary": dict(target.EXPECTED_TRUTH_BOUNDARY),
        "next_gate": target.EXPECTED_NEXT_GATE,
    }
    evidence = _self_hashed(evidence_core, "evidence_identity_sha256")
    values = {
        target.FILES["evidence"]: evidence,
        target.FILES["quality"]: quality,
        target.FILES["privacy"]: privacy,
        target.FILES["survivor_inventory"]: inventory,
    }
    for filename, value in values.items():
        (path / filename).write_bytes(target.canonical_line(value))


def _qualify(tmp_path: Path, *, enforce: bool = False) -> dict:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _bundle(a)
    _bundle(b)
    return target.qualify(
        a,
        b,
        tmp_path / "proof.json",
        expected_execution_head="c" * 40,
        expected_parent_artifact_id=123,
        expected_parent_artifact_zip_sha256="d" * 64,
        enforce_checkout_provenance=enforce,
    )


def test_two_clean_accepts_byte_identical_bound_bundles(tmp_path: Path) -> None:
    proof = _qualify(tmp_path)
    assert proof["fresh_execution_count"] == 2
    assert proof["independent_runner_jobs"] is True
    assert proof["byte_identical_outputs"] is True
    assert proof["parent_execution_head_sha"] == target.PARENT_EXECUTION_HEAD
    assert proof["proof_identity_sha256"] == target.self_hash(
        proof, "proof_identity_sha256"
    )
    assert target.load_json_bytes(
        (tmp_path / "proof.json").read_bytes(), "proof"
    ) == proof


def test_two_clean_rejects_same_directory(tmp_path: Path) -> None:
    a = tmp_path / "same"
    _bundle(a)
    with pytest.raises(
        target.G05G06TwoCleanError,
        match="directories must be distinct",
    ):
        target.qualify(
            a,
            a,
            tmp_path / "proof.json",
            expected_execution_head="c" * 40,
            expected_parent_artifact_id=123,
            expected_parent_artifact_zip_sha256="d" * 64,
            enforce_checkout_provenance=False,
        )


def test_two_clean_rejects_byte_drift(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _bundle(a)
    _bundle(b)
    path = b / target.FILES["quality"]
    value = json.loads(path.read_text(encoding="utf-8"))
    value["extra"] = True
    path.write_bytes(target.canonical_line(value))
    with pytest.raises(target.G05G06TwoCleanError, match="two-clean output differs"):
        target.qualify(
            a,
            b,
            tmp_path / "proof.json",
            expected_execution_head="c" * 40,
            expected_parent_artifact_id=123,
            expected_parent_artifact_zip_sha256="d" * 64,
            enforce_checkout_provenance=False,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("artifact_id", 999, "parent artifact ID drift"),
        ("artifact_zip_sha256", "e" * 64, "parent artifact ZIP SHA-256 drift"),
    ],
)
def test_two_clean_rejects_resealed_parent_transport_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _bundle(a)
    _bundle(b)
    for output in (a, b):
        path = output / target.FILES["evidence"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["parent"][field] = value
        evidence["evidence_identity_sha256"] = target.self_hash(
            evidence, "evidence_identity_sha256"
        )
        path.write_bytes(target.canonical_line(evidence))
    with pytest.raises(target.G05G06TwoCleanError, match=message):
        target.qualify(
            a,
            b,
            tmp_path / "proof.json",
            expected_execution_head="c" * 40,
            expected_parent_artifact_id=123,
            expected_parent_artifact_zip_sha256="d" * 64,
            enforce_checkout_provenance=False,
        )


def test_two_clean_rejects_resealed_zero_credit_drift(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _bundle(a)
    _bundle(b)
    for output in (a, b):
        path = output / target.FILES["evidence"]
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["truth_boundary"]["training_executed"] = True
        evidence["evidence_identity_sha256"] = target.self_hash(
            evidence, "evidence_identity_sha256"
        )
        path.write_bytes(target.canonical_line(evidence))
    with pytest.raises(target.G05G06TwoCleanError, match="zero-credit truth boundary drift"):
        target.qualify(
            a,
            b,
            tmp_path / "proof.json",
            expected_execution_head="c" * 40,
            expected_parent_artifact_id=123,
            expected_parent_artifact_zip_sha256="d" * 64,
            enforce_checkout_provenance=False,
        )


def test_two_clean_rejects_noncanonical_json(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _bundle(a)
    _bundle(b)
    for output in (a, b):
        path = output / target.FILES["privacy"]
        value = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(target.G05G06TwoCleanError, match="noncanonical durable JSON"):
        target.qualify(
            a,
            b,
            tmp_path / "proof.json",
            expected_execution_head="c" * 40,
            expected_parent_artifact_id=123,
            expected_parent_artifact_zip_sha256="d" * 64,
            enforce_checkout_provenance=False,
        )


def test_two_clean_rejects_unbound_extra_member(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _bundle(a)
    _bundle(b)
    (a / "stale.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(target.G05G06TwoCleanError, match="bundle member set drift"):
        target.qualify(
            a,
            b,
            tmp_path / "proof.json",
            expected_execution_head="c" * 40,
            expected_parent_artifact_id=123,
            expected_parent_artifact_zip_sha256="d" * 64,
            enforce_checkout_provenance=False,
        )


def test_proof_write_is_immutable_and_resumable(tmp_path: Path) -> None:
    proof = _qualify(tmp_path)
    proof_path = tmp_path / "proof.json"
    target.write_immutable_bytes(proof_path, target.canonical_line(proof))
    with pytest.raises(target.G05G06TwoCleanError, match="divergent proof overwrite"):
        target.write_immutable_bytes(proof_path, b"{}\n")


def test_checkout_provenance_runs_before_proof_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    _bundle(a)
    _bundle(b)
    calls: list[str] = []

    def reject(value: str) -> str:
        calls.append(value)
        raise target.G05G06TwoCleanError("execution HEAD drift")

    monkeypatch.setattr(target, "verify_checkout", reject)
    proof_path = tmp_path / "proof.json"
    with pytest.raises(target.G05G06TwoCleanError, match="execution HEAD drift"):
        target.qualify(
            a,
            b,
            proof_path,
            expected_execution_head="c" * 40,
            expected_parent_artifact_id=123,
            expected_parent_artifact_zip_sha256="d" * 64,
            enforce_checkout_provenance=True,
        )
    assert calls == ["c" * 40]
    assert not proof_path.exists()


def test_two_clean_calls_pinned_text_free_authority_root_verifiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = tmp_path / "a-root"
    b = tmp_path / "b-root"
    _bundle(a)
    _bundle(b)
    calls: list[tuple[str, str, str]] = []

    monkeypatch.setattr(
        target,
        "verify_checkout",
        lambda value: value,
    )

    def quality_verify(
        authority: dict,
        *,
        expected_input_manifest_sha256: str,
        expected_input_rows_sha256: str,
        expected_execution_identity_sha256: str,
    ) -> str:
        calls.append(
            (
                "quality",
                expected_input_manifest_sha256,
                expected_input_rows_sha256,
            )
        )
        assert authority["execution_identity_sha256"] == (
            expected_execution_identity_sha256
        )
        return expected_execution_identity_sha256

    def privacy_verify(
        authority: dict,
        *,
        expected_input_rows_sha256: str,
        expected_execution_identity_sha256: str,
    ) -> str:
        calls.append(
            (
                "privacy",
                expected_input_rows_sha256,
                expected_execution_identity_sha256,
            )
        )
        assert authority["execution_identity_sha256"] == (
            expected_execution_identity_sha256
        )
        return expected_execution_identity_sha256

    monkeypatch.setattr(
        target,
        "load_authority_verifiers",
        lambda: (quality_verify, privacy_verify),
    )
    target.qualify(
        a,
        b,
        tmp_path / "proof-root.json",
        expected_execution_head="c" * 40,
        expected_parent_artifact_id=123,
        expected_parent_artifact_zip_sha256="d" * 64,
        enforce_checkout_provenance=True,
    )

    assert [row[0] for row in calls] == [
        "quality",
        "privacy",
        "quality",
        "privacy",
    ]


def test_two_clean_rejects_coherently_resealed_inventory_digest_drift(
    tmp_path: Path,
) -> None:
    a = tmp_path / "a-inventory"
    b = tmp_path / "b-inventory"
    _bundle(a)
    _bundle(b)
    for output in (a, b):
        inventory_path = output / target.FILES["survivor_inventory"]
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        inventory["record_inventory_digest_sha256"] = "0" * 64
        inventory_path.write_bytes(target.canonical_line(inventory))

        evidence_path = output / target.FILES["evidence"]
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["survivor_inventory"][
            "record_inventory_digest_sha256"
        ] = "0" * 64
        evidence["durable_artifacts"][
            "survivor_inventory_file_sha256"
        ] = target.sha256(target.canonical_line(inventory))
        evidence["evidence_identity_sha256"] = target.self_hash(
            evidence, "evidence_identity_sha256"
        )
        evidence_path.write_bytes(target.canonical_line(evidence))

    with pytest.raises(
        target.G05G06TwoCleanError,
        match="survivor record inventory digest drift",
    ):
        target.qualify(
            a,
            b,
            tmp_path / "proof-inventory.json",
            expected_execution_head="c" * 40,
            expected_parent_artifact_id=123,
            expected_parent_artifact_zip_sha256="d" * 64,
            enforce_checkout_provenance=False,
        )
