import copy
import hashlib
import json

import pytest

import twelve_six.data.expanded_postdedup_inventory_v1 as target


def canonical(value, *, ascii_only=False, newline=False):
    rendered = json.dumps(
        value,
        ensure_ascii=ascii_only,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return rendered + (b"\n" if newline else b"")


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def self_hash(value, key, *, ascii_only=False, newline=False):
    body = dict(value)
    body.pop(key, None)
    return sha256(canonical(body, ascii_only=ascii_only, newline=newline))


def payload(text):
    raw = text.encode("utf-8")
    return sha256(raw), len(raw)


def reseal_matcher(bundle):
    matcher = bundle["report"]["dedup_v3"]
    matcher["report_sha256"] = self_hash(
        matcher,
        "report_sha256",
        ascii_only=True,
        newline=True,
    )


def reseal_report(bundle):
    report = bundle["report"]
    report["report_sha256"] = self_hash(report, "report_sha256")


def reseal_survivor(bundle):
    survivor = bundle["survivor"]
    survivor["survivor_authority_sha256"] = self_hash(
        survivor,
        "survivor_authority_sha256",
    )


def build_valid(monkeypatch):
    pa, pab = payload("alpha")
    pb, pbb = payload("beta")
    records = [
        {
            "record_id": "r1",
            "source_id": "s1",
            "family": "fixture.base",
            "modality": "text",
            "payload_sha256": pa,
            "payload_bytes": pab,
        },
        {
            "record_id": "r2",
            "source_id": "s1",
            "family": "fixture.base",
            "modality": "text",
            "payload_sha256": pb,
            "payload_bytes": pbb,
        },
    ]
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in records
    ]
    record_digest = sha256(canonical(records))
    payload_digest = sha256(canonical(payload_projection))
    monkeypatch.setattr(target, "DATA526_RECORD_INVENTORY_SHA256", record_digest)
    monkeypatch.setattr(target, "DATA526_PAYLOAD_INVENTORY_SHA256", payload_digest)

    data526 = {
        "schema_version": target.DATA526_RECORD_INVENTORY_SCHEMA,
        "record_count": len(records),
        "record_inventory_digest_sha256": record_digest,
        "payload_inventory_digest_sha256": payload_digest,
        "records": copy.deepcopy(records),
    }
    retained_bytes = sum(row["payload_bytes"] for row in records)
    matcher = {
        "schema_version": target.UPSTREAM_MATCHER_SCHEMA,
        "local_free_only": True,
        "model_training_executed": False,
        "raw_text_emitted": False,
        "source_count": 1,
        "sources": [
            {
                "source_id": "s1",
                "source_family": "fixture.base",
                "modality": "text",
                "declared_capacity_bytes": retained_bytes,
                "verified_raw_bytes": retained_bytes,
                "verified_raw_sha256": sha256(b"fixture-source"),
            }
        ],
    }
    matcher["report_sha256"] = self_hash(
        matcher,
        "report_sha256",
        ascii_only=True,
        newline=True,
    )
    report = {
        "schema_version": target.UPSTREAM_REPORT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "raw_text_emitted": False,
        "claim_boundary": dict(target.UPSTREAM_REPORT_BOUNDARY),
        "data526": {"record_inventory_digest_sha256": record_digest},
        "dedup_v3": matcher,
        "source_vector": {
            "post_dedup_conservative_unique_bytes": retained_bytes,
        },
    }
    report["report_sha256"] = self_hash(report, "report_sha256")
    survivor = {
        "schema_version": target.UPSTREAM_SURVIVOR_SCHEMA,
        "matcher_report_sha256": matcher["report_sha256"],
        "truth_boundary": dict(target.UPSTREAM_SURVIVOR_BOUNDARY),
        "survivor_source_ids": ["s1"],
        "post_dedup_survivor_source_object_count": 1,
        "post_dedup_declared_capacity_bytes": retained_bytes,
    }
    survivor["survivor_authority_sha256"] = self_hash(
        survivor,
        "survivor_authority_sha256",
    )
    return {
        "report": report,
        "survivor": survivor,
        "data526": data526,
        "payloads": [
            {"record_id": "r1", "text": "alpha"},
            {"record_id": "r2", "text": "beta"},
        ],
        "retained_bytes": retained_bytes,
    }


def freeze_bundle(bundle):
    return target.freeze_expanded_inventory(
        bundle["report"],
        bundle["survivor"],
        bundle["data526"],
        expected_report_sha256=bundle["report"]["report_sha256"],
        expected_survivor_authority_sha256=bundle["survivor"][
            "survivor_authority_sha256"
        ],
    )


def test_freeze_consumes_real_v9_contract_without_local_reselection(monkeypatch):
    bundle = build_valid(monkeypatch)
    inventory = freeze_bundle(bundle)
    assert [row["record_id"] for row in inventory["records"]] == ["r1", "r2"]
    assert inventory["source_count"] == 1
    assert inventory["record_count"] == 2
    assert inventory["retained_payload_bytes"] == bundle["retained_bytes"]
    assert inventory["truth_boundary"] == target.ZERO_TRUTH
    target.verify_inventory(
        inventory,
        expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
    )


def test_report_substitution_fails_against_external_expected_identity(monkeypatch):
    bundle = build_valid(monkeypatch)
    expected = bundle["report"]["report_sha256"]
    bundle["report"]["source_vector"]["post_dedup_conservative_unique_bytes"] += 1
    reseal_report(bundle)
    with pytest.raises(ValueError, match="report identity mismatch"):
        target.freeze_expanded_inventory(
            bundle["report"],
            bundle["survivor"],
            bundle["data526"],
            expected_report_sha256=expected,
            expected_survivor_authority_sha256=bundle["survivor"][
                "survivor_authority_sha256"
            ],
        )


def test_report_schema_and_profile_fail_closed(monkeypatch):
    bundle = build_valid(monkeypatch)
    bundle["report"]["schema_version"] = "wrong"
    reseal_report(bundle)
    with pytest.raises(ValueError, match="report schema"):
        freeze_bundle(bundle)

    bundle = build_valid(monkeypatch)
    bundle["report"]["execution_profile"] = "PAID"
    reseal_report(bundle)
    with pytest.raises(ValueError, match="LOCAL_FREE"):
        freeze_bundle(bundle)


def test_report_boundary_is_closed_world_and_type_safe(monkeypatch):
    bundle = build_valid(monkeypatch)
    bundle["report"]["claim_boundary"]["future_training_authorized"] = False
    reseal_report(bundle)
    with pytest.raises(ValueError, match="canonical keys"):
        freeze_bundle(bundle)

    bundle = build_valid(monkeypatch)
    bundle["report"]["claim_boundary"]["training_authorized_bytes"] = False
    reseal_report(bundle)
    with pytest.raises(ValueError, match="training_authorized_bytes"):
        freeze_bundle(bundle)


def test_matcher_self_hash_and_source_count_fail_closed(monkeypatch):
    bundle = build_valid(monkeypatch)
    bundle["report"]["dedup_v3"]["source_count"] = 2
    reseal_matcher(bundle)
    reseal_report(bundle)
    with pytest.raises(ValueError, match="source_count drift"):
        freeze_bundle(bundle)

    bundle = build_valid(monkeypatch)
    bundle["report"]["dedup_v3"]["sources"][0]["source_family"] = "drift"
    reseal_report(bundle)
    with pytest.raises(ValueError, match="matcher report self-hash mismatch"):
        freeze_bundle(bundle)


def test_survivor_substitution_fails_against_external_expected_identity(monkeypatch):
    bundle = build_valid(monkeypatch)
    expected = bundle["survivor"]["survivor_authority_sha256"]
    bundle["survivor"]["post_dedup_declared_capacity_bytes"] += 1
    reseal_survivor(bundle)
    with pytest.raises(ValueError, match="survivor authority identity mismatch"):
        target.freeze_expanded_inventory(
            bundle["report"],
            bundle["survivor"],
            bundle["data526"],
            expected_report_sha256=bundle["report"]["report_sha256"],
            expected_survivor_authority_sha256=expected,
        )


def test_survivor_must_bind_exact_matcher_report(monkeypatch):
    bundle = build_valid(monkeypatch)
    bundle["report"]["dedup_v3"]["sources"][0]["verified_raw_sha256"] = sha256(
        b"changed"
    )
    reseal_matcher(bundle)
    reseal_report(bundle)
    with pytest.raises(ValueError, match="survivor/matcher report binding mismatch"):
        freeze_bundle(bundle)


def test_survivor_unknown_source_and_count_alias_fail_closed(monkeypatch):
    bundle = build_valid(monkeypatch)
    bundle["survivor"]["survivor_source_ids"] = ["unknown"]
    reseal_survivor(bundle)
    with pytest.raises(ValueError, match="unknown matcher source"):
        freeze_bundle(bundle)

    bundle = build_valid(monkeypatch)
    bundle["survivor"]["post_dedup_survivor_source_object_count"] = True
    reseal_survivor(bundle)
    with pytest.raises(ValueError, match="post_dedup_survivor_source_object_count"):
        freeze_bundle(bundle)


def test_survivor_source_byte_arithmetic_fails_closed(monkeypatch):
    bundle = build_valid(monkeypatch)
    bundle["survivor"]["post_dedup_declared_capacity_bytes"] += 1
    reseal_survivor(bundle)
    with pytest.raises(ValueError, match="source byte arithmetic mismatch"):
        freeze_bundle(bundle)


def test_data526_external_digest_cannot_be_self_resealed(monkeypatch):
    bundle = build_valid(monkeypatch)
    bundle["data526"]["records"][0]["family"] = "substitute"
    normalized = sorted(bundle["data526"]["records"], key=lambda row: row["record_id"])
    bundle["data526"]["record_inventory_digest_sha256"] = sha256(canonical(normalized))
    with pytest.raises(ValueError, match="record inventory authority drift"):
        freeze_bundle(bundle)


def test_data526_record_source_bytes_and_metadata_bind_matcher(monkeypatch):
    bundle = build_valid(monkeypatch)
    source = bundle["report"]["dedup_v3"]["sources"][0]
    source["declared_capacity_bytes"] += 1
    bundle["survivor"]["post_dedup_declared_capacity_bytes"] += 1
    bundle["report"]["source_vector"]["post_dedup_conservative_unique_bytes"] += 1
    reseal_matcher(bundle)
    bundle["survivor"]["matcher_report_sha256"] = bundle["report"]["dedup_v3"][
        "report_sha256"
    ]
    reseal_survivor(bundle)
    reseal_report(bundle)
    with pytest.raises(ValueError, match="record/source byte arithmetic mismatch"):
        freeze_bundle(bundle)

    bundle = build_valid(monkeypatch)
    bundle["report"]["dedup_v3"]["sources"][0]["source_family"] = "drift"
    reseal_matcher(bundle)
    bundle["survivor"]["matcher_report_sha256"] = bundle["report"]["dedup_v3"][
        "report_sha256"
    ]
    reseal_survivor(bundle)
    reseal_report(bundle)
    with pytest.raises(ValueError, match="record/source family mismatch"):
        freeze_bundle(bundle)


def test_missing_surviving_data526_source_fails_closed(monkeypatch):
    bundle = build_valid(monkeypatch)
    for row in bundle["data526"]["records"]:
        row["source_id"] = "other"
    records = sorted(bundle["data526"]["records"], key=lambda row: row["record_id"])
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in records
    ]
    record_digest = sha256(canonical(records))
    payload_digest = sha256(canonical(payload_projection))
    monkeypatch.setattr(target, "DATA526_RECORD_INVENTORY_SHA256", record_digest)
    monkeypatch.setattr(target, "DATA526_PAYLOAD_INVENTORY_SHA256", payload_digest)
    bundle["data526"]["record_inventory_digest_sha256"] = record_digest
    bundle["data526"]["payload_inventory_digest_sha256"] = payload_digest
    bundle["report"]["data526"]["record_inventory_digest_sha256"] = record_digest
    reseal_report(bundle)
    with pytest.raises(ValueError, match="missing from record inventory"):
        freeze_bundle(bundle)


def test_inventory_is_deterministic_and_self_hashed(monkeypatch):
    bundle = build_valid(monkeypatch)
    inventory1 = freeze_bundle(bundle)
    bundle2 = copy.deepcopy(bundle)
    bundle2["data526"]["records"].reverse()
    inventory2 = freeze_bundle(bundle2)
    assert inventory1 == inventory2
    target.verify_inventory(
        inventory1,
        expected_inventory_identity_sha256=inventory1["inventory_identity_sha256"],
    )


def test_inventory_rejects_bool_alias_and_malformed_ancestry(monkeypatch):
    bundle = build_valid(monkeypatch)
    inventory = freeze_bundle(bundle)
    inventory["source_count"] = True
    inventory["inventory_identity_sha256"] = self_hash(
        inventory,
        "inventory_identity_sha256",
        newline=True,
    )
    with pytest.raises(ValueError, match="inventory.source_count"):
        target.verify_inventory(
            inventory,
            expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
        )

    inventory = freeze_bundle(bundle)
    inventory["input_report_sha256"] = "A" * 64
    inventory["inventory_identity_sha256"] = self_hash(
        inventory,
        "inventory_identity_sha256",
        newline=True,
    )
    with pytest.raises(ValueError, match="inventory.input_report_sha256"):
        target.verify_inventory(
            inventory,
            expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
        )


def test_ephemeral_handoff_binds_text_without_persisting_raw_text(monkeypatch):
    bundle = build_valid(monkeypatch)
    inventory = freeze_bundle(bundle)
    rows, evidence = target.prepare_ephemeral_data232_rows(
        inventory,
        bundle["payloads"],
        expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
    )
    assert [row["record_id"] for row in rows] == ["r1", "r2"]
    assert rows[0]["text"] == "alpha"
    evidence_text = json.dumps(evidence, sort_keys=True)
    assert "alpha" not in evidence_text and "beta" not in evidence_text
    assert evidence["retained_source_count"] == 2
    assert evidence["raw_text_persisted_in_evidence"] is False
    assert evidence["authorized_training_exposure"] == 0
    assert evidence["final_test_payload_accessed"] is False
    assert evidence["final_test_outcomes_accessed"] is False


def test_ephemeral_handoff_rejects_inventory_substitution(monkeypatch):
    bundle = build_valid(monkeypatch)
    inventory = freeze_bundle(bundle)
    expected = inventory["inventory_identity_sha256"]
    inventory["records"][0]["family"] = "substitute"
    inventory["inventory_identity_sha256"] = self_hash(
        inventory,
        "inventory_identity_sha256",
        newline=True,
    )
    with pytest.raises(ValueError, match="retained inventory identity mismatch"):
        target.prepare_ephemeral_data232_rows(
            inventory,
            bundle["payloads"],
            expected_inventory_identity_sha256=expected,
        )


def test_ephemeral_payload_drift_fails_closed(monkeypatch):
    bundle = build_valid(monkeypatch)
    inventory = freeze_bundle(bundle)
    drift = copy.deepcopy(bundle["payloads"])
    drift[0]["text"] = "alphx"
    with pytest.raises(ValueError, match="payload identity mismatch"):
        target.prepare_ephemeral_data232_rows(
            inventory,
            drift,
            expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
        )


def test_missing_extra_and_duplicate_payload_rows_fail_closed(monkeypatch):
    bundle = build_valid(monkeypatch)
    inventory = freeze_bundle(bundle)
    with pytest.raises(ValueError, match="missing payload rows"):
        target.prepare_ephemeral_data232_rows(
            inventory,
            bundle["payloads"][:1],
            expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
        )

    extra = bundle["payloads"] + [{"record_id": "r3", "text": "x"}]
    with pytest.raises(ValueError, match="unexpected payload row"):
        target.prepare_ephemeral_data232_rows(
            inventory,
            extra,
            expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
        )

    duplicate = [bundle["payloads"][0], bundle["payloads"][0], bundle["payloads"][1]]
    with pytest.raises(ValueError, match="duplicate payload row"):
        target.prepare_ephemeral_data232_rows(
            inventory,
            duplicate,
            expected_inventory_identity_sha256=inventory["inventory_identity_sha256"],
        )
