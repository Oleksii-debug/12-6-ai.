import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twelve_six.data.expanded_postdedup_inventory_v1 import (
    UPSTREAM_SURVIVOR_SCHEMA,
    ZERO_TRUTH,
    freeze_expanded_inventory,
    prepare_ephemeral_data232_rows,
    verify_inventory,
)


def canonical(value):
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (text + "\n").encode()


def self_hash(value, key):
    body = dict(value)
    body.pop(key, None)
    return hashlib.sha256(canonical(body)).hexdigest()


def payload(text):
    raw = text.encode()
    return hashlib.sha256(raw).hexdigest(), len(raw)


def build_valid():
    pa, pab = payload("alpha")
    ca, cab = payload("ALPHA")
    pb, pbb = payload("beta")
    cb, cbb = payload("BETA")
    survivors = {
        "schema": UPSTREAM_SURVIVOR_SCHEMA,
        "survivor_count": 2,
        "retained_payload_bytes": pab + pbb,
        "survivors": [
            {
                "record_id": "r2",
                "source_id": "s1",
                "family": "ua.rada",
                "modality": "text",
                "payload_sha256": pb,
                "payload_bytes": pbb,
                "comparison_policy_id": "casefold-v1",
                "comparison_sha256": cb,
                "comparison_bytes": cbb,
                "training_eligible": False,
                "evaluation_eligible": False,
            },
            {
                "record_id": "r1",
                "source_id": "s1",
                "family": "ua.rada",
                "modality": "text",
                "payload_sha256": pa,
                "payload_bytes": pab,
                "comparison_policy_id": "casefold-v1",
                "comparison_sha256": ca,
                "comparison_bytes": cab,
                "training_eligible": False,
                "evaluation_eligible": False,
            },
        ],
        "truth_boundary": dict(ZERO_TRUTH),
    }
    survivors["survivor_authority_sha256"] = self_hash(
        survivors,
        "survivor_authority_sha256",
    )
    report = {
        "schema": "twelve-six.expanded-global-dedup-report.v9",
        "survivor_authority_sha256": survivors["survivor_authority_sha256"],
        "survivor_count": 2,
        "retained_payload_bytes": pab + pbb,
        "truth_boundary": dict(ZERO_TRUTH),
    }
    report["report_sha256"] = self_hash(report, "report_sha256")
    payloads = [
        {
            "record_id": "r1",
            "normalized_payload": "alpha",
            "comparison_payload": "ALPHA",
        },
        {
            "record_id": "r2",
            "normalized_payload": "beta",
            "comparison_payload": "BETA",
        },
    ]
    return report, survivors, payloads


def freeze_valid():
    report, survivors, payloads = build_valid()
    inventory = freeze_expanded_inventory(
        report,
        survivors,
        expected_report_sha256=report["report_sha256"],
        expected_survivor_authority_sha256=survivors["survivor_authority_sha256"],
    )
    return report, survivors, inventory, payloads


def test_freeze_consumes_external_survivors_without_local_reselection():
    _, survivors, inventory, _ = freeze_valid()
    assert [row["record_id"] for row in survivors["survivors"]] == ["r2", "r1"]
    assert [row["record_id"] for row in inventory["records"]] == ["r1", "r2"]
    assert {row["record_id"] for row in inventory["records"]} == {"r1", "r2"}
    assert inventory["record_count"] == 2
    assert inventory["source_count"] == 1
    assert inventory["truth_boundary"] == ZERO_TRUTH


def test_report_substitution_fails_even_when_self_consistent():
    report, survivors, _ = build_valid()
    expected = report["report_sha256"]
    report["retained_payload_bytes"] += 1
    report["report_sha256"] = self_hash(report, "report_sha256")
    with pytest.raises(ValueError, match="report identity mismatch"):
        freeze_expanded_inventory(
            report,
            survivors,
            expected_report_sha256=expected,
            expected_survivor_authority_sha256=survivors["survivor_authority_sha256"],
        )


def test_survivor_authority_substitution_fails_even_when_self_consistent():
    report, survivors, _ = build_valid()
    expected = survivors["survivor_authority_sha256"]
    survivors["survivors"][0]["family"] = "substitute"
    survivors["survivor_authority_sha256"] = self_hash(
        survivors,
        "survivor_authority_sha256",
    )
    report["survivor_authority_sha256"] = survivors["survivor_authority_sha256"]
    report["report_sha256"] = self_hash(report, "report_sha256")
    with pytest.raises(ValueError, match="external identity mismatch"):
        freeze_expanded_inventory(
            report,
            survivors,
            expected_report_sha256=report["report_sha256"],
            expected_survivor_authority_sha256=expected,
        )


def test_duplicate_record_id_fails_closed():
    report, survivors, _ = build_valid()
    survivors["survivors"][1]["record_id"] = survivors["survivors"][0]["record_id"]
    survivors["survivor_authority_sha256"] = self_hash(
        survivors,
        "survivor_authority_sha256",
    )
    report["survivor_authority_sha256"] = survivors["survivor_authority_sha256"]
    report["report_sha256"] = self_hash(report, "report_sha256")
    with pytest.raises(ValueError, match="duplicate record_id"):
        freeze_expanded_inventory(
            report,
            survivors,
            expected_report_sha256=report["report_sha256"],
            expected_survivor_authority_sha256=survivors["survivor_authority_sha256"],
        )


def test_missing_comparison_metadata_fails_closed():
    report, survivors, _ = build_valid()
    del survivors["survivors"][0]["comparison_sha256"]
    survivors["survivor_authority_sha256"] = self_hash(
        survivors,
        "survivor_authority_sha256",
    )
    report["survivor_authority_sha256"] = survivors["survivor_authority_sha256"]
    report["report_sha256"] = self_hash(report, "report_sha256")
    with pytest.raises(ValueError, match="missing keys"):
        freeze_expanded_inventory(
            report,
            survivors,
            expected_report_sha256=report["report_sha256"],
            expected_survivor_authority_sha256=survivors["survivor_authority_sha256"],
        )


def test_count_and_byte_drift_fail_closed():
    report, survivors, _ = build_valid()
    report["retained_payload_bytes"] += 10
    report["report_sha256"] = self_hash(report, "report_sha256")
    with pytest.raises(ValueError, match="byte arithmetic mismatch"):
        freeze_expanded_inventory(
            report,
            survivors,
            expected_report_sha256=report["report_sha256"],
            expected_survivor_authority_sha256=survivors["survivor_authority_sha256"],
        )


def test_truth_boundary_cannot_be_widened():
    report, survivors, _ = build_valid()
    survivors["truth_boundary"]["training_eligible"] = True
    survivors["survivor_authority_sha256"] = self_hash(
        survivors,
        "survivor_authority_sha256",
    )
    report["survivor_authority_sha256"] = survivors["survivor_authority_sha256"]
    report["report_sha256"] = self_hash(report, "report_sha256")
    with pytest.raises(ValueError, match="training_eligible|training eligibility"):
        freeze_expanded_inventory(
            report,
            survivors,
            expected_report_sha256=report["report_sha256"],
            expected_survivor_authority_sha256=survivors["survivor_authority_sha256"],
        )


def test_zero_truth_rejects_bool_alias_for_integer_zero():
    report, survivors, _ = build_valid()
    report["truth_boundary"]["authorized_optimized_target_exposure"] = False
    report["report_sha256"] = self_hash(report, "report_sha256")
    with pytest.raises(ValueError, match="authorized_optimized_target_exposure"):
        freeze_expanded_inventory(
            report,
            survivors,
            expected_report_sha256=report["report_sha256"],
            expected_survivor_authority_sha256=survivors["survivor_authority_sha256"],
        )


def test_zero_truth_rejects_unknown_keys():
    report, survivors, _ = build_valid()
    report["truth_boundary"]["future_training_authorized"] = False
    report["report_sha256"] = self_hash(report, "report_sha256")
    with pytest.raises(ValueError, match="exactly the canonical zero-truth keys"):
        freeze_expanded_inventory(
            report,
            survivors,
            expected_report_sha256=report["report_sha256"],
            expected_survivor_authority_sha256=survivors["survivor_authority_sha256"],
        )


def test_survivor_authority_bool_count_fails_before_alias_equality():
    report, survivors, _ = build_valid()
    survivors["survivors"] = survivors["survivors"][:1]
    survivors["survivor_count"] = True
    survivors["retained_payload_bytes"] = survivors["survivors"][0]["payload_bytes"]
    survivors["survivor_authority_sha256"] = self_hash(
        survivors,
        "survivor_authority_sha256",
    )
    report["survivor_count"] = 1
    report["retained_payload_bytes"] = survivors["retained_payload_bytes"]
    report["survivor_authority_sha256"] = survivors["survivor_authority_sha256"]
    report["report_sha256"] = self_hash(report, "report_sha256")
    with pytest.raises(ValueError, match="survivor_authority.survivor_count"):
        freeze_expanded_inventory(
            report,
            survivors,
            expected_report_sha256=report["report_sha256"],
            expected_survivor_authority_sha256=survivors["survivor_authority_sha256"],
        )


def test_inventory_is_deterministic_and_self_hashed():
    report, survivors, inventory1, _ = freeze_valid()
    survivors2 = copy.deepcopy(survivors)
    survivors2["survivors"].reverse()
    survivors2["survivor_authority_sha256"] = self_hash(
        survivors2,
        "survivor_authority_sha256",
    )
    report2 = copy.deepcopy(report)
    report2["survivor_authority_sha256"] = survivors2["survivor_authority_sha256"]
    report2["report_sha256"] = self_hash(report2, "report_sha256")
    inventory2 = freeze_expanded_inventory(
        report2,
        survivors2,
        expected_report_sha256=report2["report_sha256"],
        expected_survivor_authority_sha256=survivors2["survivor_authority_sha256"],
    )
    assert [row["record_id"] for row in inventory1["records"]] == ["r1", "r2"]
    assert [row["record_id"] for row in inventory2["records"]] == ["r1", "r2"]
    verify_inventory(
        inventory1,
        expected_inventory_identity_sha256=inventory1["inventory_identity_sha256"],
    )


def test_inventory_bool_alias_counts_fail_closed():
    _, _, inventory, _ = freeze_valid()
    inventory["source_count"] = True
    inventory["inventory_identity_sha256"] = self_hash(
        inventory,
        "inventory_identity_sha256",
    )
    with pytest.raises(ValueError, match="inventory.source_count"):
        verify_inventory(
            inventory,
            expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
        )


def test_inventory_ancestry_sha_must_be_lowercase_hex64():
    _, _, inventory, _ = freeze_valid()
    inventory["input_report_sha256"] = "A" * 64
    inventory["inventory_identity_sha256"] = self_hash(
        inventory,
        "inventory_identity_sha256",
    )
    with pytest.raises(ValueError, match="inventory.input_report_sha256"):
        verify_inventory(
            inventory,
            expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
        )


def test_ephemeral_handoff_rejects_self_consistent_inventory_substitution():
    _, _, inventory, payloads = freeze_valid()
    expected = inventory["inventory_identity_sha256"]
    substituted = copy.deepcopy(inventory)
    substituted["records"][0]["family"] = "substitute"
    substituted["inventory_identity_sha256"] = self_hash(
        substituted,
        "inventory_identity_sha256",
    )
    with pytest.raises(ValueError, match="retained inventory identity mismatch"):
        prepare_ephemeral_data232_rows(
            substituted,
            payloads,
            expected_inventory_identity_sha256=expected,
        )


def test_ephemeral_handoff_binds_payload_without_durable_text():
    _, _, inventory, payloads = freeze_valid()
    rows, evidence = prepare_ephemeral_data232_rows(
        inventory,
        payloads,
        expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
    )
    assert len(rows) == 2
    assert "alpha" in rows[0].values()
    evidence_text = json.dumps(evidence, sort_keys=True)
    assert "alpha" not in evidence_text and "ALPHA" not in evidence_text
    assert evidence["record_count"] == 2
    assert evidence["truth_boundary"] == ZERO_TRUTH


def test_payload_or_comparison_drift_fails_closed():
    _, _, inventory, payloads = freeze_valid()
    drift = copy.deepcopy(payloads)
    drift[0]["normalized_payload"] = "alphx"
    with pytest.raises(ValueError, match="payload identity mismatch"):
        prepare_ephemeral_data232_rows(
            inventory,
            drift,
            expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
        )
    drift = copy.deepcopy(payloads)
    drift[0]["comparison_payload"] = "ALPHX"
    with pytest.raises(ValueError, match="comparison payload identity mismatch"):
        prepare_ephemeral_data232_rows(
            inventory,
            drift,
            expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
        )


def test_missing_or_extra_payload_rows_fail_closed():
    _, _, inventory, payloads = freeze_valid()
    with pytest.raises(ValueError, match="missing payload rows"):
        prepare_ephemeral_data232_rows(
            inventory,
            payloads[:1],
            expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
        )
    extra = payloads + [
        {
            "record_id": "r3",
            "normalized_payload": "x",
            "comparison_payload": "x",
        }
    ]
    with pytest.raises(ValueError, match="unexpected payload row"):
        prepare_ephemeral_data232_rows(
            inventory,
            extra,
            expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
        )
