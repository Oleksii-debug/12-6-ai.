from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from twelve_six.data.rights_firewall import (
    ALLOW,
    DENY,
    PURPOSES,
    EvaluationReservation,
    PurposeDecisions,
    PurposeRightsAuthority,
    PurposeRightsFirewall,
    SourceObjectIdentity,
)

CONFIG_PATH = Path("configs/data/next100_060_purpose_rights_firewall_v1.json")
EXPECTED_SCHEMA = "12-6.next100-060-purpose-rights-firewall.v1"
EXPECTED_WORKER = "NEXT100-060-EVAL-RIGHTS-FIREWALL"
EXPECTED_REPOSITORY = "Oleksii-debug/12-6-ai."


class ValidationError(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _load() -> dict[str, Any]:
    try:
        value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("unable to read firewall config") from exc
    if not isinstance(value, dict):
        raise ValidationError("firewall config must be a JSON object")
    return value


def _self_hash(config: dict[str, Any]) -> str:
    unsigned = dict(config)
    unsigned.pop("authority_identity_sha256", None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _decisions(value: object) -> PurposeDecisions:
    if not isinstance(value, dict):
        raise ValidationError("decision profile must be an object")
    _expect(set(value) == set(PURPOSES), "decision profile must contain all six purposes")
    return PurposeDecisions(
        training=value["training"],
        tokenizer_fitting=value["tokenizer-fitting"],
        selection_validation=value["selection-validation"],
        final_test=value["final-test"],
        redistribution=value["redistribution"],
        analysis=value["analysis"],
    )


def _authority(item: dict[str, Any], profiles: dict[str, Any]) -> PurposeRightsAuthority:
    profile_name = item.get("decision_profile")
    _expect(isinstance(profile_name, str), "object decision profile is missing")
    _expect(profile_name in profiles, f"unknown decision profile: {profile_name}")
    reservation_data = item.get("reservation")
    reservation = None
    if reservation_data is not None:
        _expect(isinstance(reservation_data, dict), "reservation must be an object")
        purposes = reservation_data.get("purposes")
        _expect(isinstance(purposes, list), "reservation purposes must be a list")
        reservation = EvaluationReservation(
            reserved_purposes=tuple(purposes),
            reserved_at_utc=str(reservation_data.get("reserved_at_utc", "")),
            reservation_commit_sha=str(reservation_data.get("commit", "")),
        )

    blob = item.get("git_blob_sha1")
    _expect(blob is None or isinstance(blob, str), "git blob identity must be string or null")
    source = SourceObjectIdentity(
        source_id=str(item.get("source_id", "")),
        source_family=str(item.get("family", "")),
        upstream_revision=str(item.get("revision", "")),
        object_path=str(item.get("locator", "")),
        content_sha256=str(item.get("content_sha256", "")),
        git_blob_sha1=blob,
    )
    return PurposeRightsAuthority(
        authority_id=str(item.get("authority_id", "")),
        authority_commit_sha=str(item.get("authority_commit", "")),
        issued_at_utc=str(item.get("issued_at_utc", "")),
        source_object=source,
        upstream_license_id=str(item.get("license", "")),
        rights_evidence_sha256=str(item.get("rights_evidence_sha256", "")),
        project_decision_ref=f"{EXPECTED_WORKER}:{profile_name}",
        decisions=_decisions(profiles[profile_name]),
        reservation=reservation,
    )


def validate() -> dict[str, Any]:
    config = _load()
    _expect(config.get("schema_version") == EXPECTED_SCHEMA, "schema drift")
    _expect(config.get("worker_id") == EXPECTED_WORKER, "worker drift")
    _expect(config.get("repository") == EXPECTED_REPOSITORY, "repository drift")
    _expect(config.get("execution_profile") == "LOCAL_FREE", "LOCAL_FREE required")
    _expect(config.get("training_executed") is False, "training must remain false")
    _expect(tuple(config.get("purpose_dimensions", ())) == PURPOSES, "purpose vector drift")
    _expect(config.get("decision_vocabulary") == [ALLOW, DENY], "decision vocabulary drift")
    _expect(config.get("authority_identity_sha256") == _self_hash(config), "self-hash mismatch")

    hard_rules = config.get("hard_rules")
    _expect(isinstance(hard_rules, dict), "hard rules missing")
    required_rules = {
        "broad_license_does_not_imply_project_purpose",
        "all_six_decisions_required",
        "exact_object_identity_required",
        "evaluation_requires_reservation_timestamp_and_commit",
        "selection_reservation_irreversible_against_training_and_tokenizer",
        "final_test_reservation_irreversible_against_training_tokenizer_selection",
        "final_test_may_not_influence_selection",
        "rights_change_requires_successor_authority",
        "successor_names_immediate_predecessor",
    }
    _expect(required_rules <= hard_rules.keys(), "required hard rules missing")
    _expect(all(hard_rules[key] is True for key in required_rules), "hard rule weakened")

    profiles = config.get("decision_profiles")
    _expect(isinstance(profiles, dict), "decision profiles missing")
    parsed_profiles = {name: _decisions(value) for name, value in profiles.items()}
    _expect(set(parsed_profiles) == {"TRAIN_TOKENIZER", "SELECTION", "FINAL_TEST"}, "profile drift")

    exact = config.get("exact_object_decisions")
    _expect(isinstance(exact, list) and len(exact) == 12, "expected 12 exact object decisions")
    firewall = PurposeRightsFirewall()
    authorities = []
    for item in exact:
        _expect(isinstance(item, dict), "exact object decision must be an object")
        authority = _authority(item, profiles)
        firewall.register(authority)
        authorities.append(authority)

    training = [a for a in authorities if a.decisions.training == ALLOW]
    selection = [a for a in authorities if a.decisions.selection_validation == ALLOW]
    final_test = [a for a in authorities if a.decisions.final_test == ALLOW]
    _expect(len(training) == 1, "training proof object count drift")
    _expect(len(selection) == 10, "selection object count drift")
    _expect(len(final_test) == 1, "final-test object count drift")
    _expect(all(a.reservation is not None for a in selection + final_test), "reservation missing")

    access = config.get("final_test_access")
    _expect(isinstance(access, dict), "final-test access boundary missing")
    _expect(access.get("payload_read_by_next100_060") is False, "final payload was read")
    _expect(access.get("outcomes_read_by_next100_060") is False, "final outcomes were read")
    _expect(
        access.get("outcomes_may_influence_selection") is False,
        "final outcomes may influence selection",
    )

    terminal = config.get("terminal_enforcement_inputs")
    _expect(isinstance(terminal, list), "terminal inputs missing")
    terminal_ids = {item.get("authority") for item in terminal if isinstance(item, dict)}
    _expect("EVAL-303-SELECTION-VALIDATION-COMPOSITE" in terminal_ids, "EVAL-303 missing")
    _expect("NEXT100-045-CODE-STARLETTE" in terminal_ids, "Starlette terminal input missing")

    observed = config.get("observed_concurrent_rights_authorities")
    _expect(isinstance(observed, list) and len(observed) >= 15, "concurrency intake incomplete")
    for item in observed:
        _expect(isinstance(item, dict), "concurrency row must be an object")
        if item.get("workflow_state") == "QUEUED":
            _expect(item.get("positive_enforcement") is False, "queued positive input admitted")

    successor = config.get("successor_contract")
    _expect(isinstance(successor, dict), "successor contract missing")
    _expect(
        successor.get("rights_state_change_requires_successor") is True,
        "successor rule weakened",
    )
    _expect(
        successor.get("selection_and_final_test_reservations_are_irreversible") is True,
        "reservation irreversibility weakened",
    )
    return config


def main() -> int:
    config = validate()
    print(
        "NEXT100-060 PASS "
        f"identity={config['authority_identity_sha256']} "
        f"objects={len(config['exact_object_decisions'])} "
        f"concurrent={len(config['observed_concurrent_rights_authorities'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
