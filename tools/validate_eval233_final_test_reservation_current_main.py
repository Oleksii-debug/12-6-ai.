#!/usr/bin/env python3
"""Validate the metadata-only current-main EVAL-233 final-test reservation authority."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "12-6.eval233-final-test-reservation-current-main.v1"
EXPECTED = {
    "historical_eval233_head_sha": "b5512b4648cb09dd052b08884dc53f291e1ce935",
    "historical_eval233_evidence_identity_sha256": "37473834df31c69faf39f5c1152e9fe1f7d4aeb1487fcf7489059e8ec444d4a7",
    "recover174_head_sha": "976c101b20ad2e31b0b3e2dda2beed8e7b03c2f3",
    "recover174_seed_git_blob_sha1": "4bfbfbf29fa9538cabda6068efd3a1fd036a9479",
    "recover174_seed_sha256": "7e6827d22d573dda4c9ff2b5f0ab2b8fe3fdf5aa577970ecb77e94a10cf72367",
    "recover174_source_authority_git_blob_sha1": "3ba9f221a82468f971c17eda518cd6f1642fd311",
    "recover174_source_authority_identity_sha256": "c7211b3e1e6a4f22463d0e6174f0d6162c2452585704efad5564a35de8de609f",
    "final_test_identity_sha256": "86d51eb106524cd8e4d0f94d4ff6e2e3426c6321e0698279877dfc4d5fce3116",
    "source_membership_identity_sha256": "6b012efc4d627b113b8adc2166e6ab50d9001284083f7a429c665b7752ca18d7",
}
EXPECTED_FAMILIES = {
    "en.standardebooks.manual": {
        "modality": "en",
        "source_version": "d1143a9b459b5e6f9cdda93a7c1e04676bff4f6b",
        "source_identity_sha256": "ba622171b752c4d411bd0b93a94dad14b7ff0e5ac88064678d1b91a551c01be3",
        "raw_sha256": [
            "21582c7f0e4ad39f2b0ed97bbc2c082d275e898b7a63c28e6d9badb8ee0f7860",
            "7ac53dfb4bf6f73f178560e09f33160d0250c69fb679802f3254dc0eb4c9f509",
        ],
        "admitted_source_snapshots_sha256": [
            "154fb4034929714087e75150d678bf65049ddac32e79dcdf97162c8972c2be83",
            "94eb2f529922d125b3bd40691778886f4d5d80b128b925d0274fb3d94646ec5a",
        ],
    },
    "ua.rada.open-data.laws-texts": {
        "modality": "ua",
        "source_version": "laws-texts/bounded-2026-08-25",
        "source_identity_sha256": "b8f1d2f99a3db71d894a3233e9417d6283d11768c41b1634bc8b096ab77aba4e",
        "raw_sha256": ["36eae31c3b0676ea7c02236fa05bd695c240c9a8eade5febc00457b8103ee1a4"],
        "admitted_source_snapshots_sha256": ["72c301db0b2539f3f7a73c9c15e2e425700a6b758a1114f1a861e2d60c704c50"],
    },
}
FORBIDDEN_DATA_KEYS = {
    "text",
    "record_text",
    "payload_text",
    "outcome",
    "outcomes",
    "score",
    "scores",
    "prediction",
    "predictions",
    "answer",
    "answers",
}


class ReservationValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReservationValidationError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_sha(value: Any, n: int, label: str) -> str:
    _require(
        isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{n}}}", value) is not None,
        f"{label} invalid",
    )
    return value


def _walk_no_payload(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require(
                key not in FORBIDDEN_DATA_KEYS,
                f"forbidden payload/outcome key at {path}.{key}",
            )
            _walk_no_payload(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_no_payload(child, f"{path}[{index}]")


def validate_authority(data: dict[str, Any]) -> dict[str, Any]:
    _require(data.get("schema_version") == SCHEMA, "schema drift")
    claimed = _require_sha(data.get("authority_identity_sha256"), 64, "authority identity")
    body = copy.deepcopy(data)
    body.pop("authority_identity_sha256", None)
    _require(
        hashlib.sha256(_canonical_bytes(body)).hexdigest() == claimed,
        "authority self-hash mismatch",
    )
    _walk_no_payload(data)

    port = data.get("current_main_port")
    _require(isinstance(port, dict), "current_main_port missing")
    _require_sha(port.get("base_sha"), 40, "base sha")
    _require(port.get("historical_eval233_pr") == 365, "historical EVAL-233 PR drift")
    for key in (
        "historical_eval233_head_sha",
        "recover174_head_sha",
        "recover174_seed_git_blob_sha1",
        "recover174_source_authority_git_blob_sha1",
    ):
        _require(port.get(key) == EXPECTED[key], f"{key} drift")
        _require_sha(port.get(key), 40, key)
    for key in (
        "historical_eval233_evidence_identity_sha256",
        "recover174_seed_sha256",
        "recover174_source_authority_identity_sha256",
    ):
        _require(port.get(key) == EXPECTED[key], f"{key} drift")
        _require_sha(port.get(key), 64, key)

    reservation = data.get("reservation")
    _require(isinstance(reservation, dict), "reservation missing")
    _require(reservation.get("classification") == "final_test", "classification drift")
    _require(
        reservation.get("final_test_identity_sha256") == EXPECTED["final_test_identity_sha256"],
        "final-test identity drift",
    )
    _require(
        reservation.get("source_membership_identity_sha256")
        == EXPECTED["source_membership_identity_sha256"],
        "source membership identity drift",
    )
    _require(reservation.get("documents") == 16, "document count drift")
    _require(
        reservation.get("modality_documents") == {"ua": 8, "en": 8, "code": 0},
        "modality counts drift",
    )
    _require(reservation.get("immutable") is True, "reservation must remain immutable")
    families = reservation.get("source_families")
    _require(
        isinstance(families, list) and len(families) == 2,
        "exactly two final-test source families required",
    )
    by_id = {family.get("source_id"): family for family in families if isinstance(family, dict)}
    _require(set(by_id) == set(EXPECTED_FAMILIES), "source-family set drift")
    for source_id, expected in EXPECTED_FAMILIES.items():
        actual = by_id[source_id]
        _require(actual.get("source_family") == source_id, f"family identity drift: {source_id}")
        _require(
            actual.get("evaluation_status") == "APPROVED_FOR_HELDOUT_EVALUATION",
            f"evaluation authority drift: {source_id}",
        )
        for key, expected_value in expected.items():
            _require(actual.get(key) == expected_value, f"{source_id}.{key} drift")
        _require_sha(actual.get("source_identity_sha256"), 64, f"{source_id} identity")
        hashes = actual.get("raw_sha256", []) + actual.get("admitted_source_snapshots_sha256", [])
        for value in hashes:
            _require_sha(value, 64, f"{source_id} source hash")

    firewall = data.get("firewall")
    _require(isinstance(firewall, dict), "firewall missing")
    false_keys = [
        "training_allowed",
        "tokenizer_fit_allowed",
        "selection_allowed",
        "hyperparameter_selection_allowed",
        "scoring_before_selection_lock_allowed",
        "final_test_outcomes_read",
        "final_test_outcomes_published",
        "payload_embedded",
        "model_architecture_or_hyperparameters_selected",
        "training_executed",
        "paid_compute",
    ]
    for key in false_keys:
        _require(firewall.get(key) is False, f"firewall weakened: {key}")
    true_keys = [
        "payload_access_for_decontamination_matching_only",
        "durable_evidence_hash_and_source_metadata_only",
        "local_free_only",
    ]
    for key in true_keys:
        _require(firewall.get(key) is True, f"firewall weakened: {key}")
    _require(
        firewall.get("authorized_training_exposure") == 0,
        "training exposure must remain zero",
    )

    late = data.get("late_bound_decontamination")
    _require(isinstance(late, dict), "late-bound decontamination contract missing")
    _require(late.get("consumer_pr") == 874, "downstream consumer drift")
    _require(
        late.get("expected_final_test_identity_sha256")
        == EXPECTED["final_test_identity_sha256"],
        "downstream final-test identity drift",
    )
    _require(
        late.get("payload_resolution_status") == "LATE_BOUND_DECONTAMINATION_ONLY",
        "payload resolution boundary drift",
    )
    _require(
        late.get("member_level_payload_binding_identity_sha256") is None,
        "member-level binding must remain late-bound",
    )
    _require(
        late.get("member_level_content_hashes_published_here") is False,
        "member-level hashes must not be fabricated here",
    )
    _require(
        late.get("final_test_payload_may_be_read_for_matching") is True,
        "decontamination payload-read boundary drift",
    )
    _require(
        late.get("final_test_outcomes_must_remain_unread") is True,
        "final-test outcome firewall weakened",
    )
    return data


def validate_path(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(parsed, dict), "authority must be an object")
    return validate_authority(parsed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("configs/evaluation/eval233_final_test_reservation_current_main_v1.json"),
    )
    args = parser.parse_args()
    data = validate_path(args.path)
    print(
        json.dumps(
            {
                "status": "PASS",
                "authority_identity_sha256": data["authority_identity_sha256"],
                "documents": data["reservation"]["documents"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
