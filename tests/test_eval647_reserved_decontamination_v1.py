from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data.current_reserved_decontamination_v1 import (
    CurrentDecontaminationExecutionError,
    build_reserved_payload_binding,
)
from twelve_six.data.eval647_future_training_exclusion_v1 import (
    build_eval647_auxiliary_reserved_set,
    compose_eval647_future_training_exclusion,
)
from twelve_six.data.eval647_reserved_decontamination_v1 import (
    Eval647ReservedDecontaminationError,
    execute_eval647_reserved_decontamination,
    verify_eval647_reserved_decontamination_receipt,
)

INVENTORY = "1" * 64
SURVIVOR = "2" * 64
SELECTION = "3" * 64
FINAL = "4" * 64
SELECTION_MEMBERSHIP = "5" * 64
FINAL_MEMBERSHIP = "6" * 64
EVAL647_EVIDENCE = "7" * 64
EVAL647_MEMBERSHIP = "8" * 64
DISCOVERY_HEAD = "c" * 40
REV_A = "d" * 40
REV_B = "e" * 40
RAW_A = "def reserved_alpha():\n    return 7\n"
RAW_B = "def reserved_beta(value):\n    return value * 3\n"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _row(
    record_id: str,
    text: str,
    *,
    source: str,
    family: str,
    modality: str = "en",
) -> dict[str, str]:
    return {
        "record_id": record_id,
        "source_id": source,
        "source_family": family,
        "modality": modality,
        "text": text,
    }


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


def _projection(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    projection: list[dict[str, object]] = []
    for row in rows:
        raw = row["text"].encode("utf-8")
        projection.append(
            {
                "record_id": row["record_id"],
                "source_id": row["source_id"],
                "source_family": row["source_family"],
                "modality": row["modality"].lower(),
                "text_sha256": _sha(raw),
                "text_utf8_bytes": len(raw),
            }
        )
    return sorted(projection, key=lambda item: str(item["record_id"]))


def _training_handoff(training: list[dict[str, str]]) -> dict[str, object]:
    projection = _projection(training)
    core: dict[str, object] = {
        "schema_version": "12-6.postdedup-decontam-handoff.v1",
        "postdedup_inventory_identity_sha256": INVENTORY,
        "input_survivor_authority_sha256": SURVIVOR,
        "retained_source_count": len(training),
        "matcher_input_projection": projection,
        "matcher_input_projection_sha256": _sha(_canonical_bytes(projection)),
        "raw_text_persisted_in_evidence": False,
        "final_test_payload_accessed": False,
        "final_test_outcomes_accessed": False,
        "authorized_training_exposure": 0,
    }
    core["handoff_identity_sha256"] = _sha(_canonical_bytes(core))
    return core


def _base_rows() -> tuple[dict[str, str], dict[str, str]]:
    selection = _row(
        "selection-existing",
        "selection validation payload about isolated graph traversal",
        source="selection-source",
        family="selection-family",
    )
    final = _row(
        "final-existing",
        "sealed final payload about unrelated orbital mechanics",
        source="final-source",
        family="final-family",
    )
    return selection, final


def _base_binding() -> dict[str, object]:
    selection, final = _base_rows()
    return build_reserved_payload_binding(
        [
            {
                "authority_id": "eval303-selection",
                "identity_sha256": SELECTION,
                "role": "selection_validation",
                "source_sha": "a" * 40,
                "source_membership_identity_sha256": SELECTION_MEMBERSHIP,
                "members": [_member(selection)],
            },
            {
                "authority_id": "eval233-final",
                "identity_sha256": FINAL,
                "role": "final_test",
                "source_sha": "b" * 40,
                "source_membership_identity_sha256": FINAL_MEMBERSHIP,
                "members": [_member(final)],
            },
        ]
    )


def _manifest_and_materialization() -> tuple[dict[str, object], dict[str, object]]:
    specs = [
        {
            "source_family": "github:example/alpha",
            "repository": "example/alpha",
            "revision": REV_A,
            "path": "alpha.py",
            "git_blob_sha1": "1" * 40,
            "license_spdx": "MIT",
            "raw_sha256": _sha(RAW_A.encode("utf-8")),
            "expected_raw_bytes": len(RAW_A.encode("utf-8")),
        },
        {
            "source_family": "github:example/beta",
            "repository": "example/beta",
            "revision": REV_B,
            "path": "beta.py",
            "git_blob_sha1": "2" * 40,
            "license_spdx": "Apache-2.0",
            "raw_sha256": _sha(RAW_B.encode("utf-8")),
            "expected_raw_bytes": len(RAW_B.encode("utf-8")),
        },
    ]
    objects: list[dict[str, object]] = []
    sealed: list[dict[str, object]] = []
    for spec in specs:
        objects.append(
            {
                **spec,
                "evaluation_use": "selection_validation",
                "training_allowed": False,
                "tokenizer_fit_allowed": False,
                "permanent_future_training_exclusion": True,
            }
        )
        sealed.append(
            {
                "source_family": spec["source_family"],
                "repository": spec["repository"],
                "revision": spec["revision"],
                "path": spec["path"],
                "git_blob_sha1": spec["git_blob_sha1"],
                "license_spdx": spec["license_spdx"],
                "raw_sha256": spec["raw_sha256"],
                "raw_bytes": spec["expected_raw_bytes"],
                "training_allowed": False,
                "tokenizer_fit_allowed": False,
                "permanent_future_training_exclusion": True,
            }
        )
    manifest: dict[str, object] = {
        "schema_version": "12-6.eval-code-reserve-v1.contract.v1",
        "issue": 647,
        "execution_class": "LOCAL_FREE",
        "purpose": "selection_validation_only",
        "reservation": {
            "training_allowed": False,
            "tokenizer_fit_allowed": False,
            "permanent_future_training_exclusion": True,
            "historical_training_exposure_required": 0,
            "historical_tokenizer_fit_exposure_required": 0,
        },
        "objects": objects,
        "materialization_evidence": {"identity_sha256": EVAL647_EVIDENCE},
    }
    materialization: dict[str, object] = {
        "schema_version": "12-6.eval-code-reserve-v1.source-materialization-terminal.v1",
        "reservation_authority_issue": 647,
        "execution_profile": "LOCAL_FREE",
        "repeat_execution_byte_identical": True,
        "raw_payload_persisted_in_repository": False,
        "selection_validation_records_authorized": 0,
        "evidence_identity_sha256": EVAL647_EVIDENCE,
        "object_set_identity_sha256": EVAL647_MEMBERSHIP,
        "discovery_head_sha": DISCOVERY_HEAD,
        "objects": sealed,
    }
    return manifest, materialization


def _eval647_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, (repository, revision, path, family, text) in enumerate(
        [
            ("example/alpha", REV_A, "alpha.py", "github:example/alpha", RAW_A),
            ("example/beta", REV_B, "beta.py", "github:example/beta", RAW_B),
        ]
    ):
        raw_sha = _sha(text.encode("utf-8"))
        rows.append(
            _row(
                f"eval647:{index}:{raw_sha}",
                text,
                source=f"{repository}@{revision}:{path}",
                family=family,
                modality="code",
            )
        )
    return rows


def _fixture() -> dict[str, object]:
    manifest, materialization = _manifest_and_materialization()
    base = _base_binding()
    composed, _ = compose_eval647_future_training_exclusion(
        base,
        manifest,
        materialization,
    )
    training = [
        _row(
            "training-1",
            "independent training prose about agricultural weather forecasting",
            source="training-source",
            family="training-family",
        )
    ]
    handoff = _training_handoff(training)
    selection, final = _base_rows()
    evaluation = [selection, final, *_eval647_rows()]
    return {
        "manifest": manifest,
        "materialization": materialization,
        "base": base,
        "composed": composed,
        "training": training,
        "handoff": handoff,
        "evaluation": evaluation,
    }


def _execute(fixture: dict[str, object]):
    base = fixture["base"]
    composed = fixture["composed"]
    handoff = fixture["handoff"]
    assert isinstance(base, dict)
    assert isinstance(composed, dict)
    assert isinstance(handoff, dict)
    return execute_eval647_reserved_decontamination(
        fixture["training"],
        fixture["evaluation"],
        training_handoff_evidence=handoff,
        base_reserved_binding=base,
        manifest=fixture["manifest"],
        materialization_evidence=fixture["materialization"],
        expected_base_reserved_binding_identity_sha256=str(
            base["binding_identity_sha256"]
        ),
        expected_composed_reserved_binding_identity_sha256=str(
            composed["binding_identity_sha256"]
        ),
        expected_eval647_materialization_evidence_identity_sha256=EVAL647_EVIDENCE,
        expected_eval647_object_set_identity_sha256=EVAL647_MEMBERSHIP,
        expected_inventory_identity_sha256=INVENTORY,
        expected_survivor_authority_sha256=SURVIVOR,
        expected_training_handoff_identity_sha256=str(
            handoff["handoff_identity_sha256"]
        ),
        expected_selection_validation_identity_sha256=SELECTION,
        expected_final_test_identity_sha256=FINAL,
        quarantine_cross_source_families=False,
    )


def test_real_eval647_manifest_builds_exact_sealed_auxiliary_identity():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "configs/evaluation/eval_code_reserve_v1.json").read_text(
            encoding="utf-8"
        )
    )
    materialization = json.loads(
        (
            root
            / "evidence/eval647/code_selection_source_materialization_v1.json"
        ).read_text(encoding="utf-8")
    )
    auxiliary = build_eval647_auxiliary_reserved_set(manifest, materialization)
    assert auxiliary["identity_sha256"] == (
        "3401db10bad35fd1c6fac2839413fc6afffac58fa5f2135d2202b944bc2fda82"
    )
    assert auxiliary["source_membership_identity_sha256"] == (
        "0557410622403b411ac9d6d8fb01a57ef461617c19d06f112aa61dd57f670048"
    )
    assert len(auxiliary["members"]) == 2
    assert {member["content_sha256"] for member in auxiliary["members"]} == {
        "ed8fecab2e515676af051d00098b0f043c6d30ca56480d85d00902a49ae5d0c0",
        "6aff1f84b0a70b96c102e3b92a70539255f1489765fc54140b1c1478f95b4828",
    }


def test_execution_consumes_auxiliary_reserved_set_without_elevating_authority():
    report, evidence, receipt = _execute(_fixture())
    assert report["status"] == "PASS_CLEAN"
    assert evidence["reserved_payload_binding_identity_sha256"] == (
        receipt["composed_reserved_binding_identity_sha256"]
    )
    assert receipt["future_training_exclusion_consumed_by_decontamination_execution"] is True
    assert receipt["current_corpus_launch_authority_promoted"] is False
    assert receipt["selection_validation_records_authorized"] == 0
    assert receipt["authorized_optimized_target_exposure"] == 0
    assert receipt["optimizer_updates_executed_on_real_targets"] == 0
    assert receipt["tokenizer_fit_authorized"] is False
    assert receipt["training_executed"] is False
    assert receipt["learned_weights_created"] is False
    verify_eval647_reserved_decontamination_receipt(
        receipt,
        expected_composed_reserved_binding_identity_sha256=str(
            receipt["composed_reserved_binding_identity_sha256"]
        ),
        expected_eval647_materialization_evidence_identity_sha256=EVAL647_EVIDENCE,
        expected_eval647_object_set_identity_sha256=EVAL647_MEMBERSHIP,
    )


def test_missing_eval647_raw_payload_is_rejected_by_incumbent_executor():
    fixture = _fixture()
    fixture["evaluation"] = list(fixture["evaluation"])[:-1]
    with pytest.raises(
        CurrentDecontaminationExecutionError,
        match="coverage differs from reserved membership",
    ):
        _execute(fixture)


def test_eval647_raw_payload_mutation_is_rejected_by_incumbent_executor():
    fixture = _fixture()
    evaluation = copy.deepcopy(fixture["evaluation"])
    evaluation[-1]["text"] += "# tampered\n"
    fixture["evaluation"] = evaluation
    with pytest.raises(
        CurrentDecontaminationExecutionError,
        match="content SHA-256 drift",
    ):
        _execute(fixture)


def test_self_consistent_composed_binding_requires_independent_expected_identity():
    fixture = _fixture()
    base = fixture["base"]
    handoff = fixture["handoff"]
    assert isinstance(base, dict)
    assert isinstance(handoff, dict)
    with pytest.raises(
        Eval647ReservedDecontaminationError,
        match="composed EVAL-647 reserved binding is not independently expected",
    ):
        execute_eval647_reserved_decontamination(
            fixture["training"],
            fixture["evaluation"],
            training_handoff_evidence=handoff,
            base_reserved_binding=base,
            manifest=fixture["manifest"],
            materialization_evidence=fixture["materialization"],
            expected_base_reserved_binding_identity_sha256=str(
                base["binding_identity_sha256"]
            ),
            expected_composed_reserved_binding_identity_sha256="f" * 64,
            expected_eval647_materialization_evidence_identity_sha256=(
                EVAL647_EVIDENCE
            ),
            expected_eval647_object_set_identity_sha256=EVAL647_MEMBERSHIP,
            expected_inventory_identity_sha256=INVENTORY,
            expected_survivor_authority_sha256=SURVIVOR,
            expected_training_handoff_identity_sha256=str(
                handoff["handoff_identity_sha256"]
            ),
            expected_selection_validation_identity_sha256=SELECTION,
            expected_final_test_identity_sha256=FINAL,
            quarantine_cross_source_families=False,
        )


def test_materialization_substitution_requires_independent_expected_identity():
    fixture = _fixture()
    materialization = copy.deepcopy(fixture["materialization"])
    materialization["evidence_identity_sha256"] = "9" * 64
    manifest = copy.deepcopy(fixture["manifest"])
    manifest["materialization_evidence"]["identity_sha256"] = "9" * 64
    fixture["materialization"] = materialization
    fixture["manifest"] = manifest
    base = fixture["base"]
    assert isinstance(base, dict)
    fixture["composed"], _ = compose_eval647_future_training_exclusion(
        base,
        manifest,
        materialization,
    )
    with pytest.raises(
        Eval647ReservedDecontaminationError,
        match="materialization evidence is not independently expected",
    ):
        _execute(fixture)


def test_rehashed_receipt_cannot_widen_training_boundary():
    _, _, receipt = _execute(_fixture())
    tampered = copy.deepcopy(receipt)
    tampered["training_executed"] = True
    tampered.pop("receipt_identity_sha256")
    tampered["receipt_identity_sha256"] = _sha(_canonical_bytes(tampered))
    with pytest.raises(
        Eval647ReservedDecontaminationError,
        match="receipt weakened boundary: training_executed",
    ):
        verify_eval647_reserved_decontamination_receipt(
            tampered,
            expected_composed_reserved_binding_identity_sha256=str(
                receipt["composed_reserved_binding_identity_sha256"]
            ),
            expected_eval647_materialization_evidence_identity_sha256=(
                EVAL647_EVIDENCE
            ),
            expected_eval647_object_set_identity_sha256=EVAL647_MEMBERSHIP,
        )
