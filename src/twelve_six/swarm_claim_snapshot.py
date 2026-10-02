from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from twelve_six.swarm_protocol_v2 import (
    canonical_lane_key,
    ci_pressure,
)

CLAIM_SNAPSHOT_SCHEMA = "12-6.swarm-claim-snapshot.v1"
RISK_TIER_POLICY_SCHEMA = "12-6.swarm-risk-tier-policy.v1"
DEFAULT_CLAIM_SNAPSHOT_MAX_AGE_SECONDS = 300
MAX_SNAPSHOT_FUTURE_SKEW_SECONDS = 5
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RISK_TIERS = (
    "A_AUTHORITY",
    "B_RUNTIME",
    "C_EXECUTION",
    "D_NONAUTHORITY",
)


def _normalize_key_part(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("key parts must be non-empty strings")
    tokens = re.findall(r"[A-Z0-9]+", value.upper())
    if not tokens:
        raise ValueError("key part has no ASCII alphanumeric identity")
    return "-".join(tokens)


def _parse_utc(value: str, name: str) -> datetime:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty ISO-8601 string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{name} must be valid ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{name} must be expressed in UTC")
    return parsed.astimezone(UTC)


def _canonical_utc_text(value: str, name: str) -> str:
    parsed = _parse_utc(value, name)
    return parsed.isoformat().replace("+00:00", "Z")


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object member: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant rejected: {value}")


def _strict_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number rejected: {value}")
    return number


def strict_json_loads(text: str) -> Any:
    if type(text) is not str:
        raise ValueError("JSON input must be text")
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_object_pairs,
        parse_constant=_reject_nonfinite_constant,
        parse_float=_strict_json_float,
    )


def canonical_json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical standards JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def risk_tier_policy_document() -> dict[str, Any]:
    return {
        "schema": RISK_TIER_POLICY_SCHEMA,
        "snapshot_never_grants_mutation_authority": True,
        "existing_stricter_rules_remain_binding": True,
        "automatic_integration_enabled": False,
        "activation_requires_independent_qualification": True,
        "d_nonauthority_auto_integration_requires_protected_main": True,
        "tiers": {
            "A_AUTHORITY": {
                "shared_exact_head_ci_required": True,
                "automated_contract_tests_required": True,
                "different_worker_audit": "REQUIRED_FRESH_EXACT_HEAD",
                "examples": [
                    "corpus_admission",
                    "decontamination_or_final_test_authority",
                    "tokenizer_promotion",
                    "positive_optimized_exposure",
                    "optimizer_start",
                    "scientific_checkpoint_promotion",
                ],
            },
            "B_RUNTIME": {
                "shared_exact_head_ci_required": True,
                "automated_contract_tests_required": True,
                "different_worker_audit": "REQUIRED_TARGETED_ADVERSARIAL",
                "examples": [
                    "trainer",
                    "checkpoint_or_recovery",
                    "dedup_or_matcher_semantics",
                ],
            },
            "C_EXECUTION": {
                "shared_exact_head_ci_required": True,
                "automated_contract_tests_required": True,
                "different_worker_audit": "REQUIRED_IF_SENSITIVE_AUTHORITY_FIELDS_CHANGE",
                "examples": [
                    "carrier",
                    "cli",
                    "transport",
                    "provider_glue_without_authority_semantics",
                ],
            },
            "D_NONAUTHORITY": {
                "shared_exact_head_ci_required": True,
                "automated_contract_tests_required": False,
                "different_worker_audit": "NOT_REQUIRED_BY_TIER_ONLY",
                "automatic_integration_preconditions": [
                    "protected_main_policy_active",
                    "explicit_protocol_activation",
                    "all_other_repository_merge_rules_satisfied",
                ],
                "examples": [
                    "docs",
                    "accessibility",
                    "formatting",
                    "non_authority_telemetry",
                ],
            },
        },
    }


def risk_tier_policy_identity_sha256() -> str:
    return canonical_json_sha256(risk_tier_policy_document())


def validate_risk_tier_policy_config(value: Any) -> None:
    if type(value) is not dict:
        raise ValueError("risk tier policy config must be an object")
    expected_policy = risk_tier_policy_document()
    expected_keys = set(expected_policy) | {"policy_identity_sha256"}
    _require_exact_keys(value, expected_keys, "risk tier policy config")

    identity = value["policy_identity_sha256"]
    if type(identity) is not str or _SHA256_RE.fullmatch(identity) is None:
        raise ValueError("risk tier policy identity must be lowercase 64-hex")
    if identity != risk_tier_policy_identity_sha256():
        raise ValueError("risk tier policy identity mismatch")

    semantics = dict(value)
    del semantics["policy_identity_sha256"]
    if semantics != expected_policy:
        raise ValueError("risk tier policy semantics mismatch")


def risk_tier_requirements(
    tier: str, *, sensitive_authority_fields_changed: bool
) -> dict[str, Any]:
    if type(tier) is not str or tier not in RISK_TIERS:
        raise ValueError("unknown risk tier")
    if type(sensitive_authority_fields_changed) is not bool:
        raise ValueError("sensitive_authority_fields_changed must be a boolean")

    policy = risk_tier_policy_document()["tiers"][tier]
    audit_rule = policy["different_worker_audit"]
    audit_required = audit_rule.startswith("REQUIRED") and (
        audit_rule != "REQUIRED_IF_SENSITIVE_AUTHORITY_FIELDS_CHANGE"
        or sensitive_authority_fields_changed
    )
    return {
        "tier": tier,
        "shared_exact_head_ci_required": policy["shared_exact_head_ci_required"],
        "automated_contract_tests_required": policy["automated_contract_tests_required"],
        "different_worker_audit_required": audit_required,
        "grants_merge_authority": False,
        "grants_mutation_authority": False,
    }


def snapshot_grants_mutation_authority(_snapshot: Mapping[str, Any]) -> bool:
    return False


def _require_exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_sha1(value: Any, name: str) -> str:
    if type(value) is not str or _SHA1_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 40-hex SHA")
    return value


def _require_nonempty_ascii(value: Any, name: str, *, max_length: int = 256) -> str:
    if type(value) is not str or not value or len(value) > max_length:
        raise ValueError(f"{name} must be a non-empty bounded string")
    if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
        raise ValueError(f"{name} must contain printable ASCII only")
    return value


def _validate_canonical_lane_key(value: Any) -> str:
    key = _require_nonempty_ascii(value, "lane_key", max_length=180)
    parts = key.split("|")
    if len(parts) != 4:
        raise ValueError("lane_key must contain exactly four canonical parts")
    canonical = canonical_lane_key(*parts)
    if canonical != key:
        raise ValueError("lane_key is not canonical")
    return key


def _validate_ownership_key(value: Any) -> str:
    key = _require_nonempty_ascii(value, "ownership_key")
    parts = key.split("|")
    if len(parts) < 2 or any(not part for part in parts):
        raise ValueError("ownership_key must contain canonical pipe-separated parts")
    if any(_normalize_key_part(part) != part for part in parts):
        raise ValueError("ownership_key is not canonical")
    return key


def _lease_state(record: dict[str, Any], generated_at: datetime) -> str:
    released_at = record["released_at_utc"]
    if released_at is not None or not record["ownership_active"]:
        return "RELEASED_OR_TERMINAL"
    lease_until = record["lease_until_utc"]
    if lease_until is None:
        return "MISSING_LEASE_FAIL_CLOSED"
    expiry = _parse_utc(lease_until, "lease_until_utc")
    if expiry < generated_at:
        return "EXPIRED_REQUIRES_LIVE_STALE_TAKEOVER_CHECK"
    return "ACTIVE_WINDOW"


def _normalize_claim_record(record: Any, generated_at: datetime) -> dict[str, Any]:
    if type(record) is not dict:
        raise ValueError("claim record must be an object")
    expected = {
        "issue_number",
        "ownership_key",
        "lane_key",
        "status",
        "ownership_active",
        "created_at_utc",
        "lease_until_utc",
        "released_at_utc",
        "pull_request",
        "head_sha",
    }
    _require_exact_keys(record, expected, "claim record")

    issue_number = _require_int(record["issue_number"], "claim issue_number", minimum=1)
    ownership_key = _validate_ownership_key(record["ownership_key"])
    lane_key = _validate_canonical_lane_key(record["lane_key"])
    status = _require_nonempty_ascii(record["status"], "claim status", max_length=128)
    ownership_active = record["ownership_active"]
    if type(ownership_active) is not bool:
        raise ValueError("ownership_active must be a boolean")

    created_at_text = _canonical_utc_text(record["created_at_utc"], "created_at_utc")
    created_at = _parse_utc(created_at_text, "created_at_utc")
    if created_at > generated_at:
        raise ValueError("claim created_at_utc is after snapshot generation")

    lease_until = record["lease_until_utc"]
    if lease_until is not None:
        lease_until = _canonical_utc_text(lease_until, "lease_until_utc")
        lease_parsed = _parse_utc(lease_until, "lease_until_utc")
        if lease_parsed < created_at:
            raise ValueError("claim lease expires before claim creation")
    elif ownership_active:
        raise ValueError("active claim must include lease_until_utc")

    released_at = record["released_at_utc"]
    if released_at is not None:
        released_at = _canonical_utc_text(released_at, "released_at_utc")
        released_parsed = _parse_utc(released_at, "released_at_utc")
        if released_parsed < created_at or released_parsed > generated_at:
            raise ValueError("released_at_utc is outside the claim lifetime")
    if ownership_active and released_at is not None:
        raise ValueError("active claim cannot carry released_at_utc")

    pull_request = record["pull_request"]
    head_sha = record["head_sha"]
    if pull_request is None and head_sha is not None:
        raise ValueError("head_sha requires pull_request")
    if pull_request is not None:
        _require_int(pull_request, "pull_request", minimum=1)
        _require_sha1(head_sha, "head_sha")

    normalized = {
        "issue_number": issue_number,
        "ownership_key": ownership_key,
        "lane_key": lane_key,
        "status": status,
        "ownership_active": ownership_active,
        "created_at_utc": created_at_text,
        "lease_until_utc": lease_until,
        "released_at_utc": released_at,
        "pull_request": pull_request,
        "head_sha": head_sha,
    }
    normalized["lease_state"] = _lease_state(normalized, generated_at)
    return normalized


def _normalize_coverage(coverage: Any) -> dict[str, Any]:
    if type(coverage) is not dict:
        raise ValueError("coverage must be an object")
    expected = {
        "direct_collection_page_size",
        "open_issue_pages_scanned",
        "open_issue_last_nonempty_page_count",
        "open_issue_terminal_empty_page_observed",
        "open_issue_count",
        "open_pr_pages_scanned",
        "open_pr_last_nonempty_page_count",
        "open_pr_terminal_empty_page_observed",
        "open_pr_count",
    }
    _require_exact_keys(coverage, expected, "coverage")

    page_size = _require_int(
        coverage["direct_collection_page_size"],
        "direct_collection_page_size",
        minimum=1,
    )
    if page_size != 100:
        raise ValueError("direct_collection_page_size must be exactly 100")

    result: dict[str, Any] = {"direct_collection_page_size": page_size}
    for prefix in ("open_issue", "open_pr"):
        pages = _require_int(
            coverage[f"{prefix}_pages_scanned"],
            f"{prefix}_pages_scanned",
            minimum=0,
        )
        last_count = _require_int(
            coverage[f"{prefix}_last_nonempty_page_count"],
            f"{prefix}_last_nonempty_page_count",
            minimum=0,
        )
        count = _require_int(coverage[f"{prefix}_count"], f"{prefix}_count")
        terminal = coverage[f"{prefix}_terminal_empty_page_observed"]
        if terminal is not True:
            raise ValueError(
                f"{prefix}_terminal_empty_page_observed must be exact true"
            )

        if pages == 0:
            if count != 0 or last_count != 0:
                raise ValueError(f"{prefix} coverage geometry mismatch")
        else:
            if last_count < 1 or last_count > page_size:
                raise ValueError(
                    f"{prefix}_last_nonempty_page_count must be between 1 and page size"
                )
            expected_count = (pages - 1) * page_size + last_count
            if count != expected_count:
                raise ValueError(f"{prefix} coverage geometry mismatch")

        result[f"{prefix}_pages_scanned"] = pages
        result[f"{prefix}_last_nonempty_page_count"] = last_count
        result[f"{prefix}_terminal_empty_page_observed"] = terminal
        result[f"{prefix}_count"] = count
    return result


def _normalize_scheduler_reservations(value: Any) -> list[dict[str, str]]:
    if type(value) is not list:
        raise ValueError("scheduler_reservations must be an array")
    normalized: list[dict[str, str]] = []
    identities: set[tuple[str, str]] = set()
    for item in value:
        if type(item) is not dict:
            raise ValueError("scheduler reservation must be an object")
        _require_exact_keys(item, {"reservation_id", "exclusive_key"}, "scheduler reservation")
        reservation_id = _require_nonempty_ascii(
            item["reservation_id"], "reservation_id", max_length=256
        )
        exclusive_key = _require_nonempty_ascii(
            item["exclusive_key"], "exclusive_key", max_length=256
        )
        identity = (reservation_id, exclusive_key)
        if identity in identities:
            raise ValueError("duplicate scheduler reservation identity")
        identities.add(identity)
        normalized.append({"reservation_id": reservation_id, "exclusive_key": exclusive_key})
    return sorted(normalized, key=lambda item: (item["exclusive_key"], item["reservation_id"]))


def build_claim_snapshot(
    *,
    repository: str,
    main_sha: str,
    generated_at_utc: str,
    coverage: dict[str, Any],
    claims: list[dict[str, Any]],
    scheduler_reservations: list[dict[str, str]],
    queued_actions: int,
    in_progress_actions: int,
) -> dict[str, Any]:
    repository_value = _require_nonempty_ascii(repository, "repository", max_length=200)
    main_sha_value = _require_sha1(main_sha, "main_sha")
    generated_at_text = _canonical_utc_text(generated_at_utc, "generated_at_utc")
    generated_at = _parse_utc(generated_at_text, "generated_at_utc")
    coverage_value = _normalize_coverage(coverage)
    queued = _require_int(queued_actions, "queued_actions")
    in_progress = _require_int(in_progress_actions, "in_progress_actions")

    if type(claims) is not list:
        raise ValueError("claims must be an array")
    normalized_claims = [_normalize_claim_record(record, generated_at) for record in claims]
    if len(normalized_claims) > coverage_value["open_issue_count"]:
        raise ValueError("claim records exceed observed open issue count")
    issue_numbers: set[int] = set()
    active_ownership_keys: dict[str, int] = {}
    for record in normalized_claims:
        issue_number = record["issue_number"]
        if issue_number in issue_numbers:
            raise ValueError("duplicate claim issue_number")
        issue_numbers.add(issue_number)
        if record["ownership_active"]:
            ownership_key = record["ownership_key"]
            if ownership_key in active_ownership_keys:
                raise ValueError("duplicate active ownership_key")
            active_ownership_keys[ownership_key] = issue_number

    normalized_claims.sort(key=lambda item: item["issue_number"])
    reservations = _normalize_scheduler_reservations(scheduler_reservations)
    policy_identity = risk_tier_policy_identity_sha256()
    payload: dict[str, Any] = {
        "schema": CLAIM_SNAPSHOT_SCHEMA,
        "repository": repository_value,
        "generated_at_utc": generated_at_text,
        "main": {"branch": "main", "sha": main_sha_value},
        "coverage": coverage_value,
        "claims": normalized_claims,
        "active_ownership_keys": dict(sorted(active_ownership_keys.items())),
        "scheduler_reservations": reservations,
        "ci_pressure": {
            "queued_actions": queued,
            "in_progress_actions": in_progress,
            "classification": ci_pressure(queued, in_progress),
        },
        "risk_policy": {
            "schema": RISK_TIER_POLICY_SCHEMA,
            "identity_sha256": policy_identity,
            "automatic_integration_enabled": False,
        },
        "authority_boundary": {
            "candidate_discovery_only": True,
            "live_exact_semantic_check_required_before_claim": True,
            "live_exact_semantic_check_required_before_mutation": True,
            "live_exact_semantic_check_required_before_run_once_action": True,
            "snapshot_grants_mutation_authority": False,
            "snapshot_grants_training_authority": False,
            "snapshot_grants_merge_authority": False,
        },
    }
    payload["snapshot_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_claim_snapshot(
    snapshot: dict[str, Any],
    *,
    now_utc: str,
    max_age_seconds: int = DEFAULT_CLAIM_SNAPSHOT_MAX_AGE_SECONDS,
) -> None:
    if type(snapshot) is not dict:
        raise ValueError("snapshot must be an object")
    expected_top = {
        "schema",
        "repository",
        "generated_at_utc",
        "main",
        "coverage",
        "claims",
        "active_ownership_keys",
        "scheduler_reservations",
        "ci_pressure",
        "risk_policy",
        "authority_boundary",
        "snapshot_sha256",
    }
    _require_exact_keys(snapshot, expected_top, "snapshot")
    if snapshot["schema"] != CLAIM_SNAPSHOT_SCHEMA:
        raise ValueError("snapshot schema mismatch")
    if type(max_age_seconds) is not int or max_age_seconds <= 0:
        raise ValueError("max_age_seconds must be a positive integer")

    now = _parse_utc(now_utc, "now_utc")
    generated = _parse_utc(snapshot["generated_at_utc"], "generated_at_utc")
    age_seconds = (now - generated).total_seconds()
    if age_seconds < -MAX_SNAPSHOT_FUTURE_SKEW_SECONDS:
        raise ValueError("snapshot generation time is too far in the future")
    if age_seconds > max_age_seconds:
        raise ValueError("claim snapshot is stale")

    main = snapshot["main"]
    if type(main) is not dict:
        raise ValueError("main must be an object")
    _require_exact_keys(main, {"branch", "sha"}, "main")
    if main["branch"] != "main":
        raise ValueError("snapshot main branch mismatch")
    _require_sha1(main["sha"], "main sha")
    _require_nonempty_ascii(snapshot["repository"], "repository", max_length=200)
    coverage_value = _normalize_coverage(snapshot["coverage"])

    if type(snapshot["claims"]) is not list:
        raise ValueError("claims must be an array")
    normalized_claims: list[dict[str, Any]] = []
    for stored_record in snapshot["claims"]:
        if type(stored_record) is not dict:
            raise ValueError("claim record must be an object")
        raw_record = dict(stored_record)
        stored_lease_state = raw_record.pop("lease_state", None)
        normalized_record = _normalize_claim_record(raw_record, generated)
        if stored_lease_state != normalized_record["lease_state"]:
            raise ValueError("claim lease_state mismatch")
        normalized_claims.append(normalized_record)
    if normalized_claims != snapshot["claims"]:
        raise ValueError("claim records are not canonical")
    if len(normalized_claims) > coverage_value["open_issue_count"]:
        raise ValueError("claim records exceed observed open issue count")
    issue_numbers = [record["issue_number"] for record in normalized_claims]
    if issue_numbers != sorted(issue_numbers) or len(issue_numbers) != len(set(issue_numbers)):
        raise ValueError("claim records must be unique and sorted by issue_number")

    expected_active: dict[str, int] = {}
    for record in normalized_claims:
        if record["ownership_active"]:
            ownership_key = record["ownership_key"]
            if ownership_key in expected_active:
                raise ValueError("duplicate active ownership_key")
            expected_active[ownership_key] = record["issue_number"]
    expected_active = dict(sorted(expected_active.items()))
    if snapshot["active_ownership_keys"] != expected_active:
        raise ValueError("active ownership index mismatch")

    reservations = _normalize_scheduler_reservations(snapshot["scheduler_reservations"])
    if reservations != snapshot["scheduler_reservations"]:
        raise ValueError("scheduler reservations are not canonical")

    pressure = snapshot["ci_pressure"]
    if type(pressure) is not dict:
        raise ValueError("ci_pressure must be an object")
    _require_exact_keys(
        pressure,
        {"queued_actions", "in_progress_actions", "classification"},
        "ci_pressure",
    )
    queued = _require_int(pressure["queued_actions"], "queued_actions")
    in_progress = _require_int(pressure["in_progress_actions"], "in_progress_actions")
    if pressure["classification"] != ci_pressure(queued, in_progress):
        raise ValueError("CI pressure classification mismatch")

    risk_policy = snapshot["risk_policy"]
    if type(risk_policy) is not dict:
        raise ValueError("risk_policy must be an object")
    _require_exact_keys(
        risk_policy,
        {"schema", "identity_sha256", "automatic_integration_enabled"},
        "risk_policy",
    )
    if risk_policy["schema"] != RISK_TIER_POLICY_SCHEMA:
        raise ValueError("risk policy schema mismatch")
    if risk_policy["identity_sha256"] != risk_tier_policy_identity_sha256():
        raise ValueError("risk policy identity mismatch")
    if risk_policy["automatic_integration_enabled"] is not False:
        raise ValueError("automatic integration must remain disabled")

    boundary = snapshot["authority_boundary"]
    expected_boundary = {
        "candidate_discovery_only": True,
        "live_exact_semantic_check_required_before_claim": True,
        "live_exact_semantic_check_required_before_mutation": True,
        "live_exact_semantic_check_required_before_run_once_action": True,
        "snapshot_grants_mutation_authority": False,
        "snapshot_grants_training_authority": False,
        "snapshot_grants_merge_authority": False,
    }
    if type(boundary) is not dict or boundary != expected_boundary:
        raise ValueError("snapshot authority boundary mismatch")

    snapshot_sha = snapshot["snapshot_sha256"]
    if type(snapshot_sha) is not str or _SHA256_RE.fullmatch(snapshot_sha) is None:
        raise ValueError("snapshot_sha256 must be lowercase 64-hex")
    unhashed = dict(snapshot)
    del unhashed["snapshot_sha256"]
    if canonical_json_sha256(unhashed) != snapshot_sha:
        raise ValueError("snapshot content identity mismatch")
