from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA = "d03-hplt3-ukr-cyrl-source-audit-v1"
EXPECTED_SOURCE_ID = "HPLT-3.0-ukr_Cyrl"
EXPECTED_RELEASE = "HPLT Monolingual Datasets 3.0"
EXPECTED_LANGUAGE = "ukr_Cyrl"
EXPECTED_DATASET_CARD_COMMIT = "3394d6ba8dae4da834e3b11771daf95028a960b1"
EXPECTED_MAP_URL = "https://data.hplt-project.org/three/sorted/ukr_Cyrl.map"
EXPECTED_MD5_URL = "https://data.hplt-project.org/three/sorted/ukr_Cyrl.md5"
BLOCKED_VERDICT = "BLOCKED_SOURCE_RIGHTS_AND_IMMUTABLE_ACQUISITION"
REQUIRED_PROJECT_GATES = (
    "source_level_rights",
    "privacy_review",
    "project_global_dedup",
    "evaluation_decontamination",
    "deterministic_split",
    "unique_loss_ledger",
)


class HPLT3ContractError(ValueError):
    """Raised when the HPLT 3.0 Ukrainian source-audit contract is unsafe."""


def canonical_payload(contract: dict[str, Any]) -> bytes:
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def contract_sha256(contract: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_payload(contract)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HPLT3ContractError(message)


def _require_zero_credit(credit: dict[str, Any]) -> None:
    expected_zero = {
        "downloaded_bytes": 0,
        "training_authorized_bytes": 0,
        "unique_corpus_tokens": 0,
        "unique_causal_loss_positions": 0,
        "optimizer_updates": 0,
    }
    for field, expected in expected_zero.items():
        _require(credit.get(field) == expected, f"{field} must remain {expected}")
    _require(credit.get("tokenizer_fit_permitted") is False, "tokenizer fit must remain disabled")
    _require(credit.get("bulk_download_performed") is False, "bulk download must remain false")
    _require(credit.get("final_test_payload_read") is False, "final-test payload must remain sealed")
    _require(credit.get("paid_compute_used") is False, "paid compute must remain false")


def _require_fail_closed_rights(rights: dict[str, Any]) -> None:
    _require(rights.get("packaging_license") == "CC0-1.0", "packaging license drift")
    _require(
        rights.get("packaging_license_scope") == "PACKAGING_ONLY",
        "CC0 must be scoped to packaging only",
    )
    _require(rights.get("underlying_text_owned_by_hplt") is False, "underlying text ownership overclaim")
    _require(
        rights.get("underlying_text_training_rights") == "UNRESOLVED_SOURCE_LEVEL_REVIEW_REQUIRED",
        "underlying text rights must remain unresolved",
    )
    _require(
        rights.get("package_license_implies_training_authority") is False,
        "package license may not imply training authority",
    )
    _require(rights.get("canonical_training_authorized") is False, "training authorization overclaim")


def _require_upstream(upstream: dict[str, Any]) -> None:
    _require(upstream.get("release") == EXPECTED_RELEASE, "release identity drift")
    _require(upstream.get("language_script") == EXPECTED_LANGUAGE, "language/script drift")
    _require(
        upstream.get("dataset_card_commit") == EXPECTED_DATASET_CARD_COMMIT,
        "immutable dataset-card commit drift",
    )
    commit = upstream.get("dataset_card_commit", "")
    _require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "bad upstream commit")
    _require(upstream.get("map_url") == EXPECTED_MAP_URL, "map URL drift")
    _require(upstream.get("md5_url") == EXPECTED_MD5_URL, "MD5 manifest URL drift")
    _require(upstream.get("crawl_sources") == ["Common Crawl", "Internet Archive"], "crawl-source drift")
    _require(upstream.get("crawl_years") == [2012, 2024], "crawl-year boundary drift")
    _require(upstream.get("global_dedup_reported_for_ukrainian") is True, "upstream dedup fact missing")
    _require(upstream.get("provenance_metadata_reported") is True, "upstream provenance fact missing")
    _require(upstream.get("pii_annotations_reported") is True, "upstream PII fact missing")


def _require_acquisition_boundary(acquisition: dict[str, Any]) -> None:
    _require(acquisition.get("map_snapshot_sha256") is None, "map snapshot must be unresolved in audit-only package")
    _require(acquisition.get("md5_snapshot_sha256") is None, "MD5 snapshot must be unresolved in audit-only package")
    _require(acquisition.get("shard_sha256_manifest") is None, "shard SHA-256 manifest must be unresolved")
    _require(acquisition.get("immutable_acquisition_identity") is False, "immutable acquisition identity overclaim")
    _require(
        acquisition.get("md5_is_integrity_input_not_project_identity") is True,
        "MD5 must not be treated as the final project identity",
    )
    _require(
        acquisition.get("successor_must_rehash_shards_sha256") is True,
        "successor SHA-256 rehash requirement missing",
    )


def _require_project_gates(gates: dict[str, Any]) -> None:
    _require(set(gates) == set(REQUIRED_PROJECT_GATES), "project gate set drift")
    for gate in REQUIRED_PROJECT_GATES:
        _require(gates.get(gate) == "BLOCKED", f"{gate} must remain BLOCKED in audit-only package")


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    _require(contract.get("schema") == CONTRACT_SCHEMA, "schema mismatch")
    _require(contract.get("source_id") == EXPECTED_SOURCE_ID, "source identity mismatch")
    _require(contract.get("project_state") == "DISCOVERED_NOT_ADMITTED", "project state overclaim")
    _require(contract.get("admission_verdict") == BLOCKED_VERDICT, "admission verdict must fail closed")

    _require_upstream(contract.get("upstream", {}))
    _require_fail_closed_rights(contract.get("rights", {}))
    _require_acquisition_boundary(contract.get("acquisition", {}))
    _require_project_gates(contract.get("project_gates", {}))
    _require_zero_credit(contract.get("credit", {}))

    expected_hash = contract_sha256(contract)
    _require(contract.get("contract_sha256") == expected_hash, "contract SHA-256 mismatch")
    return {
        "schema": CONTRACT_SCHEMA,
        "source_id": EXPECTED_SOURCE_ID,
        "verdict": BLOCKED_VERDICT,
        "contract_sha256": expected_hash,
        "training_authorized_bytes": 0,
        "unique_causal_loss_positions": 0,
    }


def load_and_validate(path: str | Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(contract, dict), "contract root must be an object")
    return validate_contract(contract)
