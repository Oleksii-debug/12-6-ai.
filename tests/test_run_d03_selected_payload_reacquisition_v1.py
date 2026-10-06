from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.run_d03_selected_payload_reacquisition_v1 as reacquire


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def seal(value: dict, field: str) -> dict:
    output = copy.deepcopy(value)
    output[field] = hashlib.sha256(canonical(output)).hexdigest()
    return output


def authority_row(
    record_id: str = "record-a",
    source_id: str = "source-a",
    family: str = "common-pile/ubuntu_irc",
    payload: bytes = b"alpha",
) -> dict:
    return {
        "record_id": record_id,
        "source_id": source_id,
        "family": family,
        "modality": "en",
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_bytes": len(payload),
    }


def fixture_documents(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, bytes]:
    payload = b"alpha"
    row = authority_row(payload=payload)
    composition_core = {
        "combined_inventory": {
            "records": [row],
        },
    }
    composition = seal(composition_core, "composition_identity_sha256")

    lane = reacquire.LANES["ubuntu"]
    family_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
    ]
    plan_core = {
        "selected_record_count": 1,
        "selected_source_bytes": len(payload),
        "missing_record_count": 1,
        "missing_payload_bytes": len(payload),
        "family_materialization_plan": [
            {
                "family": row["family"],
                "stratum": "en",
                "missing_record_count": 1,
                "missing_payload_bytes": len(payload),
                "missing_record_ids": [row["record_id"]],
                "missing_payload_projection_sha256": hashlib.sha256(
                    canonical(family_projection)
                ).hexdigest(),
                "materializer_authority": {
                    "origin_pr": 2780,
                    "head_sha": lane["head"],
                    "mode": "PINNED_SOURCE_REMATERIALIZATION_EPHEMERAL_RAW",
                },
            }
        ],
    }
    plan = seal(plan_core, "plan_identity_sha256")

    monkeypatch.setattr(
        reacquire,
        "COMPOSITION_ID",
        composition["composition_identity_sha256"],
    )
    monkeypatch.setattr(reacquire, "PLAN_ID", plan["plan_identity_sha256"])
    monkeypatch.setattr(reacquire, "EXPECTED_SELECTED_RECORDS", 1)
    monkeypatch.setattr(reacquire, "EXPECTED_SELECTED_BYTES", len(payload))
    monkeypatch.setattr(reacquire, "EXPECTED_MISSING_RECORDS", 1)
    monkeypatch.setattr(reacquire, "EXPECTED_MISSING_BYTES", len(payload))
    return plan, composition, payload


def write_json(path: Path, value: dict) -> None:
    path.write_bytes(canonical(value) + b"\n")


def test_reproduce_post_qp_payloads_returns_transformed_survivor_bytes() -> None:
    class CleanStub:
        @staticmethod
        def _post_decontamination_records(records, report):
            assert report == {"excluded_records": []}
            record = records[0]
            return (
                [{"id": record["record_id"], "text": record["text"], "mode": "code"}],
                {
                    record["record_id"]: {
                        "source_id": record["source_id"],
                        "family": record["source_family"],
                        "mode": "code",
                    }
                },
                0,
            )

        @staticmethod
        def _input_projection(records):
            return list(records)

        @staticmethod
        def _cjson(value):
            return canonical(value)

        @staticmethod
        def _sha256(raw):
            return hashlib.sha256(raw).hexdigest()

        @staticmethod
        def build_quality_execution_authority(
            records,
            *,
            input_manifest_sha256,
            expected_input_rows_sha256,
        ):
            assert records
            assert len(input_manifest_sha256) == 64
            assert len(expected_input_rows_sha256) == 64
            return {"execution_identity_sha256": "a" * 64}

        @staticmethod
        def verify_quality_execution_authority(*args, **kwargs):
            assert args
            assert kwargs["expected_execution_identity_sha256"] == "a" * 64

        @staticmethod
        def _materialize_quality_survivors(records, metadata, quality):
            assert quality["execution_identity_sha256"] == "a" * 64
            row = records[0]
            return (
                [
                    {
                        "record_id": row["id"],
                        "source_id": metadata[row["id"]]["source_id"],
                        "family": metadata[row["id"]]["family"],
                        "modality": "code",
                        "normalized_payload": row["text"],
                    }
                ],
                {},
            )

        @staticmethod
        def _quality_records_for_privacy(records):
            return [
                {
                    "id": row["record_id"],
                    "text": row["normalized_payload"],
                    "mode": row["modality"],
                }
                for row in records
            ]

        @staticmethod
        def build_privacy_execution_authority(
            records,
            *,
            expected_input_rows_sha256,
        ):
            assert records
            assert len(expected_input_rows_sha256) == 64
            return {"execution_identity_sha256": "b" * 64}

        @staticmethod
        def verify_privacy_execution_authority(*args, **kwargs):
            assert args
            assert kwargs["expected_execution_identity_sha256"] == "b" * 64

        @staticmethod
        def _materialize_privacy_survivors(records, privacy):
            assert privacy["execution_identity_sha256"] == "b" * 64
            changed = dict(records[0])
            changed["normalized_payload"] = "redacted"
            return [changed], {"g06_redacted_records": 1}

    def build_training_authorities(rows, payloads):
        assert rows == [{"source_id": "source-a"}]
        assert payloads == {"source-a": b"raw"}
        return (
            [
                {
                    "record_id": "record-a",
                    "source_id": "source-a",
                    "source_family": "family-a",
                    "modality": "code",
                    "text": "raw",
                }
            ],
            {},
            {},
            {},
            "x",
            "y",
        )

    module = SimpleNamespace(
        build_training_authorities=build_training_authorities,
        clean=CleanStub,
    )
    result = reacquire.reproduce_post_qp_payloads(
        module,
        [{"source_id": "source-a"}],
        {"source-a": b"raw"},
    )
    assert result == {"record-a": b"redacted"}


def test_reproduce_post_qp_payloads_uses_lane_partial_materializer() -> None:
    calls = {"partial": 0}

    class CleanStub:
        @staticmethod
        def _post_decontamination_records(records, report):
            record = records[0]
            return (
                [{"id": record["record_id"], "text": record["text"], "mode": "prose"}],
                {
                    record["record_id"]: {
                        "source_id": record["source_id"],
                        "family": record["source_family"],
                        "mode": "prose",
                    }
                },
                0,
            )

        @staticmethod
        def _input_projection(records):
            return list(records)

        @staticmethod
        def _cjson(value):
            return canonical(value)

        @staticmethod
        def _sha256(raw):
            return hashlib.sha256(raw).hexdigest()

        @staticmethod
        def build_quality_execution_authority(
            records,
            *,
            input_manifest_sha256,
            expected_input_rows_sha256,
        ):
            assert records and input_manifest_sha256 and expected_input_rows_sha256
            return {"execution_identity_sha256": "a" * 64}

        @staticmethod
        def verify_quality_execution_authority(*args, **kwargs):
            assert args and kwargs

        @staticmethod
        def _materialize_quality_survivors(*args, **kwargs):
            raise AssertionError("generic materializer must not run")

        @staticmethod
        def _quality_records_for_privacy(records):
            return [
                {
                    "id": row["record_id"],
                    "text": row["normalized_payload"],
                    "mode": row["modality"],
                }
                for row in records
            ]

        @staticmethod
        def build_privacy_execution_authority(
            records,
            *,
            expected_input_rows_sha256,
        ):
            assert records and expected_input_rows_sha256
            return {"execution_identity_sha256": "b" * 64}

        @staticmethod
        def verify_privacy_execution_authority(*args, **kwargs):
            assert args and kwargs

        @staticmethod
        def _materialize_privacy_survivors(records, privacy):
            assert privacy["execution_identity_sha256"] == "b" * 64
            return list(records), {}

    def build_training_authorities(rows, payloads):
        assert rows and payloads
        return (
            [
                {
                    "record_id": "document-a",
                    "source_id": "source-a",
                    "source_family": "family-a",
                    "modality": "en",
                    "text": "alpha bravo",
                }
            ],
            {},
            {},
            {},
            "x",
            "y",
        )

    def partial_materializer(inputs, metadata, quality):
        assert quality["execution_identity_sha256"] == "a" * 64
        calls["partial"] += 1
        return (
            [
                {
                    "record_id": "document-a#quality-window-0000:deadbeef",
                    "source_id": metadata["document-a"]["source_id"],
                    "family": metadata["document-a"]["family"],
                    "modality": "prose",
                    "normalized_payload": "alpha",
                }
            ],
            {},
            {},
        )

    module = SimpleNamespace(
        build_training_authorities=build_training_authorities,
        clean=CleanStub,
        _materialize_quality_survivors_with_partial=partial_materializer,
    )
    result = reacquire.reproduce_post_qp_payloads(
        module,
        [{"source_id": "source-a"}],
        {"source-a": b"alpha bravo"},
    )
    assert calls == {"partial": 1}
    assert result == {
        "document-a#quality-window-0000:deadbeef": b"alpha"
    }


def test_selected_context_binds_family_projection_and_materializer_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, composition, payload = fixture_documents(monkeypatch)
    plan_path = tmp_path / "plan.json"
    composition_path = tmp_path / "composition.json"
    write_json(plan_path, plan)
    write_json(composition_path, composition)

    _plan, selected, families = reacquire.selected_context(
        plan_path,
        composition_path,
        "ubuntu",
    )

    assert families == ("common-pile/ubuntu_irc",)
    assert set(selected) == {"record-a"}
    assert selected["record-a"]["payload_bytes"] == len(payload)


def test_selected_context_rejects_resealed_materializer_head_substitution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, composition, _payload = fixture_documents(monkeypatch)
    plan["family_materialization_plan"][0]["materializer_authority"][
        "head_sha"
    ] = "0" * 40
    plan.pop("plan_identity_sha256")
    plan = seal(plan, "plan_identity_sha256")
    monkeypatch.setattr(reacquire, "PLAN_ID", plan["plan_identity_sha256"])

    plan_path = tmp_path / "plan.json"
    composition_path = tmp_path / "composition.json"
    write_json(plan_path, plan)
    write_json(composition_path, composition)

    with pytest.raises(reacquire.ReacquisitionError, match="materializer head drift"):
        reacquire.selected_context(plan_path, composition_path, "ubuntu")


def test_materialize_fragment_accepts_exact_source_payload() -> None:
    payload = "Привіт".encode()
    selected = {
        "record-a": authority_row(
            record_id="record-a",
            source_id="source-a",
            payload=payload,
        )
    }

    rows = reacquire.materialize_fragment(
        selected=selected,
        payloads={"source-a": payload},
    )

    assert rows == [
        {
            "record_id": "record-a",
            "source_id": "source-a",
            "family": "common-pile/ubuntu_irc",
            "modality": "en",
            "normalized_payload": "Привіт",
        }
    ]


@pytest.mark.parametrize(
    ("payloads", "error"),
    [
        ({"source-a": b"alphb"}, "payload SHA drift"),
        ({"source-a": b"alpha!"}, "payload byte drift"),
        ({}, "payload missing"),
    ],
)
def test_materialize_fragment_rejects_wrong_or_missing_payload(
    payloads: dict[str, bytes],
    error: str,
) -> None:
    selected = {"record-a": authority_row(payload=b"alpha")}
    with pytest.raises(reacquire.ReacquisitionError, match=error):
        reacquire.materialize_fragment(selected=selected, payloads=payloads)


def test_materialize_fragment_rejects_ambiguous_source_and_record_payloads() -> None:
    selected = {
        "record-a": authority_row(
            record_id="record-a",
            source_id="source-a",
            payload=b"alpha",
        )
    }
    with pytest.raises(reacquire.ReacquisitionError, match="ambiguous payload mapping"):
        reacquire.materialize_fragment(
            selected=selected,
            payloads={
                "source-a": b"alpha",
                "record-a": b"bravo",
            },
        )


def test_write_fragment_is_create_only(tmp_path: Path) -> None:
    destination = tmp_path / "fragment.jsonl"
    rows = [
        {
            "record_id": "record-a",
            "source_id": "source-a",
            "family": "common-pile/ubuntu_irc",
            "modality": "en",
            "normalized_payload": "alpha",
        }
    ]
    reacquire.write_fragment(destination, rows)
    first = destination.read_bytes()

    with pytest.raises(reacquire.ReacquisitionError, match="refusing to overwrite"):
        reacquire.write_fragment(destination, rows)

    assert destination.read_bytes() == first


def test_selected_context_rejects_payload_projection_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, composition, _payload = fixture_documents(monkeypatch)
    plan["family_materialization_plan"][0][
        "missing_payload_projection_sha256"
    ] = "f" * 64
    plan.pop("plan_identity_sha256")
    plan = seal(plan, "plan_identity_sha256")
    monkeypatch.setattr(reacquire, "PLAN_ID", plan["plan_identity_sha256"])

    plan_path = tmp_path / "plan.json"
    composition_path = tmp_path / "composition.json"
    write_json(plan_path, plan)
    write_json(composition_path, composition)

    with pytest.raises(reacquire.ReacquisitionError, match="payload projection drift"):
        reacquire.selected_context(plan_path, composition_path, "ubuntu")
