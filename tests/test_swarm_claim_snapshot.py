from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.build_swarm_claim_snapshot import build_from_input, write_create_only
from twelve_six.swarm_claim_snapshot import (
    CLAIM_SNAPSHOT_SCHEMA,
    DEFAULT_CLAIM_SNAPSHOT_MAX_AGE_SECONDS,
    RISK_TIER_POLICY_SCHEMA,
    canonical_json_sha256,
    risk_tier_policy_document,
    risk_tier_policy_identity_sha256,
    risk_tier_requirements,
    snapshot_grants_mutation_authority,
    strict_json_loads,
    validate_claim_snapshot,
    validate_risk_tier_policy_config,
)

ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-09-29T00:50:00Z"
MAIN = "7b3df41c10a826183fab0b04ae85a90cdf0ce351"


def _claim(
    issue: int = 2289,
    *,
    ownership_key: str = "PRODUCT|SWARM-PROTOCOL|CLAIM-SNAPSHOT-AND-RISK-TIERS|V1",
    status: str = "ACTIVE_ARBITRATION_WIN_PRODUCT_IMPLEMENTATION",
    ownership_active: bool = True,
    lease_until: str | None = "2026-09-29T06:47:16Z",
    released_at: str | None = None,
    pull_request: int | None = None,
    head_sha: str | None = None,
) -> dict:
    return {
        "issue_number": issue,
        "ownership_key": ownership_key,
        "lane_key": "COORD|SWARM-ACCEL|CLAIM-SNAPSHOT|RISK-TIER-ROUTING-V1",
        "status": status,
        "ownership_active": ownership_active,
        "created_at_utc": "2026-09-29T00:47:16Z",
        "lease_until_utc": lease_until,
        "released_at_utc": released_at,
        "pull_request": pull_request,
        "head_sha": head_sha,
    }


def _input(*, claims: list[dict] | None = None) -> dict:
    return {
        "repository": "Oleksii-debug/12-6-ai.",
        "main_sha": MAIN,
        "generated_at_utc": "2026-09-29T00:49:00Z",
        "coverage": {
            "direct_collection_page_size": 100,
            "open_issue_pages_scanned": 16,
            "open_issue_last_nonempty_page_count": 4,
            "open_issue_terminal_empty_page_observed": True,
            "open_issue_count": 1504,
            "open_pr_pages_scanned": 6,
            "open_pr_last_nonempty_page_count": 6,
            "open_pr_terminal_empty_page_observed": True,
            "open_pr_count": 506,
        },
        "claims": [_claim()] if claims is None else claims,
        "scheduler_reservations": [
            {
                "reservation_id": "G3-R06-F10",
                "exclusive_key": "ai:bounded-training-execution-r06",
            }
        ],
        "queued_actions": 17,
        "in_progress_actions": 4,
    }


def _snapshot(*, claims: list[dict] | None = None) -> dict:
    return build_from_input(_input(claims=claims))


def test_snapshot_round_trip_validates_and_never_grants_mutation_authority() -> None:
    snapshot = _snapshot()
    assert snapshot["schema"] == CLAIM_SNAPSHOT_SCHEMA
    assert snapshot["risk_policy"]["schema"] == RISK_TIER_POLICY_SCHEMA
    assert snapshot["ci_pressure"]["classification"] == "GREEN"
    assert snapshot["claims"][0]["lease_state"] == "ACTIVE_WINDOW"
    assert snapshot["active_ownership_keys"] == {
        "PRODUCT|SWARM-PROTOCOL|CLAIM-SNAPSHOT-AND-RISK-TIERS|V1": 2289
    }
    assert snapshot["authority_boundary"][
        "live_exact_semantic_check_required_before_claim"
    ] is True
    assert snapshot["authority_boundary"][
        "live_exact_semantic_check_required_before_mutation"
    ] is True
    assert snapshot["authority_boundary"][
        "live_exact_semantic_check_required_before_run_once_action"
    ] is True
    validate_claim_snapshot(snapshot, now_utc=NOW)
    assert snapshot_grants_mutation_authority(snapshot) is False


def test_snapshot_hash_binds_every_authority_relevant_field() -> None:
    snapshot = _snapshot()
    original = snapshot["snapshot_sha256"]
    assert len(original) == 64

    tampered = copy.deepcopy(snapshot)
    tampered["main"]["sha"] = "0" * 40
    with pytest.raises(ValueError, match="snapshot content identity mismatch"):
        validate_claim_snapshot(tampered, now_utc=NOW)

    tampered = copy.deepcopy(snapshot)
    tampered["authority_boundary"]["snapshot_grants_mutation_authority"] = True
    tampered["snapshot_sha256"] = canonical_json_sha256(
        {key: value for key, value in tampered.items() if key != "snapshot_sha256"}
    )
    with pytest.raises(ValueError, match="authority boundary mismatch"):
        validate_claim_snapshot(tampered, now_utc=NOW)


def test_stale_and_far_future_snapshots_fail_closed() -> None:
    snapshot = _snapshot()
    with pytest.raises(ValueError, match="claim snapshot is stale"):
        validate_claim_snapshot(
            snapshot,
            now_utc="2026-09-29T01:00:00Z",
            max_age_seconds=DEFAULT_CLAIM_SNAPSHOT_MAX_AGE_SECONDS,
        )

    future = build_from_input({**_input(), "generated_at_utc": "2026-09-29T00:51:00Z"})
    with pytest.raises(ValueError, match="too far in the future"):
        validate_claim_snapshot(future, now_utc=NOW)


def test_snapshot_requires_exhaustive_direct_collection_sentinels() -> None:
    document = _input()
    document["coverage"]["open_issue_terminal_empty_page_observed"] = False
    with pytest.raises(ValueError, match="must be exact true"):
        build_from_input(document)

    document = _input()
    document["coverage"]["direct_collection_page_size"] = 99
    with pytest.raises(ValueError, match="must be exactly 100"):
        build_from_input(document)


def test_snapshot_coverage_rejects_impossible_page_geometry() -> None:
    document = _input()
    document["coverage"]["open_issue_pages_scanned"] = 1
    with pytest.raises(ValueError, match="open_issue coverage geometry mismatch"):
        build_from_input(document)

    document = _input()
    document["coverage"]["open_pr_last_nonempty_page_count"] = 101
    with pytest.raises(ValueError, match="last_nonempty_page_count"):
        build_from_input(document)

    document = _input()
    document["coverage"]["open_pr_pages_scanned"] = 0
    document["coverage"]["open_pr_last_nonempty_page_count"] = 0
    with pytest.raises(ValueError, match="open_pr coverage geometry mismatch"):
        build_from_input(document)


def test_snapshot_coverage_represents_zero_open_objects_exactly() -> None:
    document = _input()
    document["claims"] = []
    for prefix in ("open_issue", "open_pr"):
        document["coverage"][f"{prefix}_pages_scanned"] = 0
        document["coverage"][f"{prefix}_last_nonempty_page_count"] = 0
        document["coverage"][f"{prefix}_count"] = 0
    snapshot = build_from_input(document)
    validate_claim_snapshot(snapshot, now_utc=NOW)


def test_snapshot_claim_records_cannot_exceed_observed_open_issues() -> None:
    document = _input()
    document["coverage"]["open_issue_pages_scanned"] = 0
    document["coverage"]["open_issue_last_nonempty_page_count"] = 0
    document["coverage"]["open_issue_count"] = 0
    with pytest.raises(ValueError, match="claim records exceed observed open issue count"):
        build_from_input(document)


def test_validator_rejects_resealed_claim_count_above_observed_open_issues() -> None:
    snapshot = _snapshot()
    snapshot["coverage"]["open_issue_pages_scanned"] = 0
    snapshot["coverage"]["open_issue_last_nonempty_page_count"] = 0
    snapshot["coverage"]["open_issue_count"] = 0
    snapshot["snapshot_sha256"] = canonical_json_sha256(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )

    with pytest.raises(ValueError, match="claim records exceed observed open issue count"):
        validate_claim_snapshot(snapshot, now_utc=NOW)


def test_duplicate_active_ownership_keys_fail_closed() -> None:
    second = _claim(issue=2290)
    second["created_at_utc"] = "2026-09-29T00:48:00Z"
    with pytest.raises(ValueError, match="duplicate active ownership_key"):
        _snapshot(claims=[_claim(), second])


def test_released_or_terminal_record_does_not_occupy_active_index() -> None:
    released = _claim(
        issue=2200,
        ownership_key="PRODUCT|OLD|LINEAGE|V1",
        status="DONE_PASS_RELEASED",
        ownership_active=False,
        lease_until="2026-09-28T12:00:00Z",
        released_at="2026-09-28T12:01:00Z",
    )
    released["created_at_utc"] = "2026-09-28T06:00:00Z"
    snapshot = _snapshot(claims=[released, _claim()])
    assert snapshot["claims"][0]["lease_state"] == "RELEASED_OR_TERMINAL"
    assert "PRODUCT|OLD|LINEAGE|V1" not in snapshot["active_ownership_keys"]
    validate_claim_snapshot(snapshot, now_utc=NOW)


def test_expired_active_lease_is_not_silently_released() -> None:
    expired = _claim(lease_until="2026-09-29T00:48:00Z")
    snapshot = _snapshot(claims=[expired])
    assert snapshot["claims"][0]["lease_state"] == (
        "EXPIRED_REQUIRES_LIVE_STALE_TAKEOVER_CHECK"
    )
    assert snapshot["active_ownership_keys"][expired["ownership_key"]] == 2289
    validate_claim_snapshot(snapshot, now_utc=NOW)


def test_live_extended_statuses_do_not_control_ownership_implicitly() -> None:
    pending = _claim(status="CLAIM_PENDING_POSTCLAIM_DIRECT_ARBITRATION")
    repaired = _claim(
        issue=2290,
        ownership_key="PRODUCT|OTHER-LINEAGE|REPAIR|V1",
        status="ARBITRATION_WIN_PRODUCT_REPAIR",
    )
    repaired["created_at_utc"] = "2026-09-29T00:48:00Z"
    snapshot = _snapshot(claims=[pending, repaired])
    assert snapshot["active_ownership_keys"] == {
        pending["ownership_key"]: 2289,
        repaired["ownership_key"]: 2290,
    }
    validate_claim_snapshot(snapshot, now_utc=NOW)


def test_ownership_active_rejects_bool_int_alias() -> None:
    aliased = _claim()
    aliased["ownership_active"] = 1
    with pytest.raises(ValueError, match="ownership_active must be a boolean"):
        _snapshot(claims=[aliased])


def test_active_claim_requires_lease_and_pr_requires_exact_head() -> None:
    with pytest.raises(ValueError, match="active claim must include lease"):
        _snapshot(claims=[_claim(lease_until=None)])

    with pytest.raises(ValueError, match="head_sha requires pull_request"):
        _snapshot(claims=[_claim(head_sha="a" * 40)])

    with pytest.raises(ValueError, match="head_sha must be a lowercase 40-hex SHA"):
        _snapshot(claims=[_claim(pull_request=123, head_sha="A" * 40)])


def test_bool_int_aliases_fail_closed_in_numeric_fields() -> None:
    document = _input()
    document["queued_actions"] = False
    with pytest.raises(ValueError, match="queued_actions"):
        build_from_input(document)

    document = _input()
    document["coverage"]["open_issue_count"] = True
    with pytest.raises(ValueError, match="open_issue_count"):
        build_from_input(document)


def test_strict_json_rejects_duplicates_and_nonfinite_numbers() -> None:
    with pytest.raises(ValueError, match="duplicate JSON object member"):
        strict_json_loads('{"a":1,"a":2}')
    for text in ('{"a":NaN}', '{"a":Infinity}', '{"a":-Infinity}', '{"a":1e400}'):
        with pytest.raises(ValueError, match="non-finite JSON"):
            strict_json_loads(text)


def test_risk_tier_policy_is_closed_and_never_grants_authority() -> None:
    policy = risk_tier_policy_document()
    assert policy["schema"] == RISK_TIER_POLICY_SCHEMA
    assert policy["snapshot_never_grants_mutation_authority"] is True
    assert policy["existing_stricter_rules_remain_binding"] is True
    assert policy["automatic_integration_enabled"] is False
    assert policy["activation_requires_independent_qualification"] is True
    assert policy["d_nonauthority_auto_integration_requires_protected_main"] is True
    assert set(policy["tiers"]) == {
        "A_AUTHORITY",
        "B_RUNTIME",
        "C_EXECUTION",
        "D_NONAUTHORITY",
    }
    assert risk_tier_policy_identity_sha256() == canonical_json_sha256(policy)

    for tier in policy["tiers"]:
        requirements = risk_tier_requirements(
            tier,
            sensitive_authority_fields_changed=False,
        )
        assert requirements["grants_merge_authority"] is False
        assert requirements["grants_mutation_authority"] is False


def test_risk_tier_audit_requirements_match_2101_contract() -> None:
    assert risk_tier_requirements(
        "A_AUTHORITY", sensitive_authority_fields_changed=False
    )["different_worker_audit_required"] is True
    assert risk_tier_requirements(
        "B_RUNTIME", sensitive_authority_fields_changed=False
    )["different_worker_audit_required"] is True
    assert risk_tier_requirements(
        "C_EXECUTION", sensitive_authority_fields_changed=False
    )["different_worker_audit_required"] is False
    assert risk_tier_requirements(
        "C_EXECUTION", sensitive_authority_fields_changed=True
    )["different_worker_audit_required"] is True
    assert risk_tier_requirements(
        "D_NONAUTHORITY", sensitive_authority_fields_changed=False
    )["different_worker_audit_required"] is False
    with pytest.raises(ValueError, match="unknown risk tier"):
        risk_tier_requirements("E_UNKNOWN", sensitive_authority_fields_changed=False)


def test_protocol_config_binds_same_snapshot_and_risk_policy() -> None:
    config = json.loads(
        (ROOT / "configs/swarm/swarm300_protocol_v2.json").read_text("utf-8")
    )
    assert config["claim_snapshot"] == {
        "schema": CLAIM_SNAPSHOT_SCHEMA,
        "builder": "tools/build_swarm_claim_snapshot.py",
        "max_age_seconds": DEFAULT_CLAIM_SNAPSHOT_MAX_AGE_SECONDS,
        "normalized_input_required": True,
        "direct_collection_terminal_empty_sentinels_required": True,
        "duplicate_active_ownership_keys": "fail_closed",
        "candidate_discovery_only": True,
        "mutation_authority": False,
        "training_authority": False,
        "merge_authority": False,
        "live_exact_semantic_check_required_before_mutation": True,
        "scheduler_reservations": "identity_mirror_only_not_lock_authority",
        "ownership_active_explicit": True,
        "status_is_evidence_not_occupancy": True,
        "direct_collection_page_size": 100,
        "coverage_geometry_must_match_counts": True,
        "live_exact_semantic_check_required_before_claim": True,
        "live_exact_semantic_check_required_before_run_once_action": True,
    }

    risk_config = config["risk_tier_routing"]
    validate_risk_tier_policy_config(risk_config)
    semantics = dict(risk_config)
    del semantics["policy_identity_sha256"]
    assert semantics == risk_tier_policy_document()


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("activation_requires_independent_qualification",), False),
        (("d_nonauthority_auto_integration_requires_protected_main",), False),
        (("tiers", "A_AUTHORITY", "different_worker_audit"), "NOT_REQUIRED"),
        (("tiers", "B_RUNTIME", "automated_contract_tests_required"), False),
    ],
)
def test_risk_policy_config_mutations_cannot_reuse_old_identity(
    path: tuple[str, ...], replacement: object
) -> None:
    config = json.loads(
        (ROOT / "configs/swarm/swarm300_protocol_v2.json").read_text("utf-8")
    )
    risk_config = copy.deepcopy(config["risk_tier_routing"])
    cursor = risk_config
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement

    with pytest.raises(ValueError, match="risk tier policy semantics mismatch"):
        validate_risk_tier_policy_config(risk_config)


def test_builder_input_is_closed_world_and_output_is_create_only(tmp_path: Path) -> None:
    document = _input()
    extra = dict(document)
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="snapshot input keys mismatch"):
        build_from_input(extra)

    output = tmp_path / "claims-v1.json"
    snapshot = build_from_input(document)
    write_create_only(output, snapshot)
    loaded = strict_json_loads(output.read_text("utf-8"))
    validate_claim_snapshot(loaded, now_utc=NOW)
    with pytest.raises(FileExistsError):
        write_create_only(output, snapshot)
