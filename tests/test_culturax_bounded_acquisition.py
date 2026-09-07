from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.culturax_bounded_acquisition import (
    CulturaXContractError,
    build_bounded_plan,
    parse_checksum_manifest,
    validate_contract,
    validate_record,
    verify_download_receipt,
)

CONTRACT_PATH = Path("configs/data/d03_culturax_bounded_acquisition_v1.json")
H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
H4 = "4" * 64


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _checksums() -> dict[str, str]:
    return {
        "en": f"{H2}  en_part_00001.parquet\n{H1}  en_part_00000.parquet\n",
        "uk": f"{H4}  uk_part_00001.parquet\n{H3}  uk_part_00000.parquet\n",
    }


def test_contract_is_fail_closed() -> None:
    validate_contract(_contract())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("upstream", "payload_download_executed"), True),
        (("upstream", "access_acceptance_observed"), True),
        (("downstream_gates", "privacy_pii"), "PASS"),
        (("claim_boundary", "training_authorized_bytes"), 1),
        (("claim_boundary", "paid_compute_used"), True),
    ],
)
def test_contract_rejects_promotion_drift(path: tuple[str, str], value: object) -> None:
    candidate = copy.deepcopy(_contract())
    candidate[path[0]][path[1]] = value
    with pytest.raises(CulturaXContractError):
        validate_contract(candidate)


def test_checksum_parse_and_bounded_selection_are_deterministic() -> None:
    plan_a = build_bounded_plan(_contract(), _checksums())
    plan_b = build_bounded_plan(_contract(), _checksums())
    assert plan_a == plan_b
    assert [(row["language"], row["filename"]) for row in plan_a["selected_shards"]] == [
        ("en", "en_part_00000.parquet"),
        ("uk", "uk_part_00000.parquet"),
    ]
    assert plan_a["training_authorized_bytes"] == 0
    assert plan_a["authorized_unique_causal_loss_positions"] == 0


def test_checksum_manifest_rejects_wrong_language_and_duplicate_digest() -> None:
    with pytest.raises(CulturaXContractError):
        parse_checksum_manifest(f"{H1} uk_part_00000.parquet\n", "en")
    with pytest.raises(CulturaXContractError):
        parse_checksum_manifest(
            f"{H1} uk_part_00000.parquet\n{H1} uk_part_00001.parquet\n",
            "uk",
        )


def test_download_receipt_must_match_selected_sha256_and_cannot_authorize() -> None:
    plan = build_bounded_plan(_contract(), _checksums())
    receipt = {
        "shards": [
            {
                "language": row["language"],
                "filename": row["filename"],
                "project_sha256": row["upstream_sha256"],
                "downloaded_bytes": 123,
            }
            for row in plan["selected_shards"]
        ],
        "training_authorized_bytes": 0,
    }
    verify_download_receipt(plan, receipt)

    bad_hash = copy.deepcopy(receipt)
    bad_hash["shards"][0]["project_sha256"] = "f" * 64
    with pytest.raises(CulturaXContractError):
        verify_download_receipt(plan, bad_hash)

    bad_credit = copy.deepcopy(receipt)
    bad_credit["training_authorized_bytes"] = 1
    with pytest.raises(CulturaXContractError):
        verify_download_receipt(plan, bad_credit)


def test_record_schema_preserves_origin_provenance() -> None:
    validate_record(
        {
            "text": "Тестовий український документ.",
            "timestamp": "2023-01-01T00:00:00Z",
            "url": "https://example.org/doc",
            "source": "OSCAR-23.01",
        }
    )
    with pytest.raises(CulturaXContractError):
        validate_record(
            {
                "text": "text",
                "timestamp": "2023",
                "url": "https://example.org",
                "source": "collapsed-culturax",
            }
        )
