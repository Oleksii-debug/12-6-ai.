#!/usr/bin/env python3
"""Fail-closed validator for the D03 eCFR acquisition probe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "12-6.d03-ecfr-acquisition-probe.v1"
STATUS = "DISCOVERY_ONLY_ZERO_CREDIT"
BASE_SHA = "a73ab38026cb7849f478cc13ad58b93534a76e2f"
SOURCE_ID = "en.us.ecfr.regulations"
FAMILY_ID = "us.federal-regulations.ecfr"
TITLES_ENDPOINT = "https://www.ecfr.gov/api/versioner-import/v1/titles"
FULL_TITLE_TEMPLATE = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml"

REQUIRED_EXCLUSIONS = (
    "incorporated_by_reference_material",
    "contractor_or_private_authorship",
    "transferred_copyright_material",
    "third_party_tables_images_media",
    "provenance_ambiguous_payload",
)

REQUIRED_SUCCESSOR_GATES = (
    "PIN_EXACT_POINT_IN_TIME_TITLE_OBJECTS",
    "TWO_BYTE_IDENTICAL_ACQUISITIONS",
    "EXACT_SHA256_AND_BYTE_LEDGER",
    "DETERMINISTIC_XML_TEXT_EXTRACTION",
    "RIGHTS_AND_PROVENANCE_CLASSIFICATION",
    "QUALITY_LANGUAGE_PRIVACY_FILTER",
    "GLOBAL_EXACT_NEAR_FRAGMENT_LINEAGE_DEDUP",
    "EVALUATION_RESERVATION_DECONTAMINATION",
    "BALANCE_AND_FAMILY_CAP_RETEST",
    "CLUSTER_SAFE_SPLIT",
    "DETERMINISTIC_TOKENIZE_PACK",
    "TWO_CLEAN_BYTE_IDENTICAL_BUILDS",
    "POST_PACK_UNIQUE_CAUSAL_LOSS_LEDGER",
    "TOKENIZER_FIT_AUTHORITY",
    "D05_CHECKPOINT_REQUALIFICATION",
    "EXPLICIT_COMPUTE_AND_TRAINING_AUTHORIZATION",
)

ZERO_CLAIMS = {
    "research_corpus_v1_released": False,
    "training_eligible": False,
    "tokenizer_fit_authorized": False,
    "model_training_executed": False,
    "optimizer_updates": 0,
    "training_authorized_bytes": 0,
    "authorized_unique_loss_positions": 0,
    "paid_compute_authorized": False,
    "learned_20m_claim": False,
    "learned_100m_claim": False,
}


class ProbeValidationError(ValueError):
    """Raised when the discovery-only authority is weakened or overclaimed."""


def _fail(message: str) -> None:
    raise ProbeValidationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _canonical_identity(config: dict[str, Any]) -> str:
    payload = copy.deepcopy(config)
    payload.pop("contract_identity_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_probe(config: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(config, dict), "probe must be a JSON object")
    _require(config.get("schema") == SCHEMA, "unexpected schema")
    _require(config.get("status") == STATUS, "probe status must remain discovery-only")

    identity = config.get("contract_identity_sha256")
    _require(
        isinstance(identity, str)
        and len(identity) == 64
        and all(ch in "0123456789abcdef" for ch in identity),
        "contract identity must be lowercase SHA-256",
    )
    _require(identity == _canonical_identity(config), "contract identity mismatch")

    base = config.get("base_authority")
    _require(isinstance(base, dict), "base_authority missing")
    _require(base.get("repository") == "Oleksii-debug/12-6-ai.", "repository drift")
    _require(base.get("git_sha") == BASE_SHA, "base SHA drift")
    _require(base.get("issue") == 672, "ownership issue drift")

    source = config.get("source")
    _require(isinstance(source, dict), "source missing")
    _require(source.get("source_id") == SOURCE_ID, "source identity drift")
    _require(source.get("family_id") == FAMILY_ID, "family identity drift")
    _require(source.get("stratum") == "en", "eCFR probe must remain EN stratum")
    _require(
        source.get("publisher")
        == "Office of the Federal Register / Government Publishing Office",
        "publisher drift",
    )
    _require(source.get("portal") == "https://www.ecfr.gov", "portal drift")
    _require(source.get("titles_endpoint") == TITLES_ENDPOINT, "titles endpoint drift")
    _require(
        source.get("historical_full_title_endpoint_template") == FULL_TITLE_TEMPLATE,
        "historical full-title endpoint drift",
    )
    _require(source.get("api_key_required") is False, "unexpected API-key requirement")
    _require(source.get("point_in_time_required") is True, "point-in-time pinning required")
    _require(
        source.get("current_or_latest_endpoint_authority")
        == "DISCOVERY_ONLY_NOT_MATERIALIZATION_AUTHORITY",
        "mutable current/latest endpoint cannot become materialization authority",
    )
    _require(
        source.get("family_accounting")
        == "ONE_CONSERVATIVE_FAMILY_UNTIL_LINEAGE_AUDIT",
        "family accounting weakened",
    )
    _require(
        source.get("title_or_agency_multiplication_allowed") is False,
        "titles/agencies cannot inflate family count",
    )

    observation = config.get("discovery_observation")
    _require(isinstance(observation, dict), "discovery observation missing")
    _require(observation.get("authority") == "DISCOVERY_ONLY", "observation overpromoted")
    _require(observation.get("import_in_progress") is False, "observation was mid-import")
    title_numbers = observation.get("title_numbers")
    _require(isinstance(title_numbers, dict), "title observation missing")
    _require(
        title_numbers == {"minimum": 1, "maximum": 50, "reserved": [35]},
        "title vector drift",
    )
    _require(observation.get("non_reserved_titles_observed") == 49, "title count drift")

    rights = config.get("rights")
    _require(isinstance(rights, dict), "rights block missing")
    _require(rights.get("decision") == "REVIEW_REQUIRED_ZERO_CREDIT", "rights overpromoted")
    _require(rights.get("us_code_anchor") == "17_USC_105", "rights anchor drift")
    _require(
        rights.get("public_availability_is_rights_authority") is False,
        "public availability is not rights authority",
    )
    _require(
        rights.get("blanket_us_government_work_assumption_allowed") is False,
        "blanket federal-hosting rights assumption is forbidden",
    )
    _require(
        rights.get("government_can_hold_transferred_copyrights") is True,
        "transferred-copyright caveat removed",
    )
    _require(rights.get("foreign_rights_not_inferred") is True, "foreign rights overclaimed")
    _require(
        tuple(rights.get("must_exclude_or_separately_clear", ())) == REQUIRED_EXCLUSIONS,
        "required rights/provenance exclusions drift",
    )
    _require(
        rights.get("rights_review_required_before_training_eligibility") is True,
        "rights review must precede training eligibility",
    )

    materialization = config.get("materialization")
    _require(isinstance(materialization, dict), "materialization block missing")
    _require(materialization.get("executed") is False, "probe cannot claim acquisition execution")
    _require(
        materialization.get("selected_title_dates") == [],
        "probe cannot pin unmaterialized titles",
    )
    for field in (
        "object_count",
        "raw_bytes",
        "normalized_bytes",
        "canonical_capacity_credit_bytes",
        "family_credit",
    ):
        _require(materialization.get(field) == 0, f"{field} must remain zero at probe")
    for field in (
        "two_byte_identical_acquisitions_required",
        "sha256_and_byte_ledger_required",
        "deterministic_xml_parse_required",
        "exact_historical_date_and_title_required",
        "redirect_and_content_type_validation_required",
    ):
        _require(materialization.get(field) is True, f"{field} must remain required")

    _require(
        tuple(config.get("required_successor_gates", ())) == REQUIRED_SUCCESSOR_GATES,
        "successor gate order/surface drift",
    )
    gate_state = config.get("gate_state")
    _require(isinstance(gate_state, dict) and gate_state, "gate_state missing")
    _require(set(gate_state.values()) == {"NOT_RUN"}, "all scientific data gates must be NOT_RUN")

    claims = config.get("claims")
    _require(isinstance(claims, dict), "claims missing")
    _require(claims == ZERO_CLAIMS, "probe cannot authorize corpus/training/compute claims")

    execution = config.get("execution")
    _require(isinstance(execution, dict), "execution missing")
    _require(execution.get("class") == "LOCAL_FREE", "execution must remain LOCAL_FREE")
    _require(execution.get("bulk_download_executed") is False, "bulk download not executed")
    _require(execution.get("final_test_accessed") is False, "final-test access forbidden")
    _require(
        execution.get("dedicated_actions_workflow_added") is False,
        "do not add another dedicated workflow while CI queue is saturated",
    )

    return {
        "valid": True,
        "status": STATUS,
        "contract_identity_sha256": identity,
        "source_id": SOURCE_ID,
        "family_id": FAMILY_ID,
        "canonical_capacity_credit_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "training_authorized": False,
        "paid_compute_authorized": False,
        "next_gate": REQUIRED_SUCCESSOR_GATES[0],
    }


def load_and_validate(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeValidationError(f"cannot read probe: {exc}") from exc
    return validate_probe(data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("configs/data/d03_ecfr_acquisition_probe_v1.json"),
    )
    args = parser.parse_args()
    try:
        result = load_and_validate(args.config)
    except ProbeValidationError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
