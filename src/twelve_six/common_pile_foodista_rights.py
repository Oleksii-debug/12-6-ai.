"""Fail-closed Foodista source-rights policy authority.

This authority deliberately does not convert sitewide/license metadata into corpus credit.
It freezes the current Foodista evidence and blocks canonical admission while the declared
automated-Web-crawl origin lacks record-level original-rights provenance.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = ROOT / "configs/data/d03_common_pile_foodista_source_rights_v1.json"
DEFAULT_REGISTRY = ROOT / "configs/data/common_pile_source_rights_v1.json"

SCHEMA_VERSION = "12-6.d03-common-pile-foodista-source-rights.v1"
AUTHORITY_ID = "D03-COMMON-PILE-FOODISTA-SOURCE-RIGHTS-V1"
STATUS = "SOURCE_POLICY_REVIEW_BLOCKED_ZERO_CREDIT"
POLICY_IDENTITY = "8f2f981ce4ed2c6d1010569cf686b053a15735d9650e902768368aa689432c1c"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

ROOT_KEYS = {
    "schema_version",
    "authority_id",
    "status",
    "execution_profile",
    "project_authority",
    "parent_registry",
    "upstream_collector",
    "product_contract",
    "primary_evidence",
    "risk_assessment",
    "decision",
    "truth_boundary",
    "policy_identity_sha256",
}

EXPECTED_PARENT = {
    "path": "configs/data/common_pile_source_rights_v1.json",
    "blob_sha1": "7b4d6828288672bf25c551e85a5d7f7399e8ef0f",
    "registry_id": "COMMON-PILE-SOURCE-RIGHTS-V1",
    "source_key": "foodista",
    "hf_dataset": "common-pile/foodista",
    "rights_basis_class": "SITEWIDE_OPEN_LICENSE",
    "rights_signal": "CC-BY",
    "required_parent_review_status": "REVIEW_REQUIRED",
}

EXPECTED_UPSTREAM = {
    "repository": "https://github.com/r-three/common-pile",
    "audited_commit": "9457f04a14cb2355ab00023420369d46ffd4a395",
    "collector_path": "sources/food",
    "readme_path": "sources/food/README.md",
    "readme_blob_sha1": "155d861c68194b68a18df1fcfa3c32e9105c645d",
    "preprocess_path": "sources/food/preprocess.py",
    "preprocess_blob_sha1": "e557449432851e4d18a76a76f23f9eedf1a6b830",
    "to_dolma_path": "sources/food/to_dolma.py",
    "to_dolma_blob_sha1": "7823c9dd05e2a0f4fb4e302df701332f430f16c4",
}

EXPECTED_PRODUCT = {
    "product_pr": 1174,
    "product_issue": 1168,
    "product_head_sha": "0b94ff93fc7860833ed24faa40a7d3e185740547",
    "config_path": "configs/data/d03_common_pile_foodista_bounded_v1.json",
    "config_blob_sha1": "9d960fa62ceaa5d2cf1804ccd3f52b5beaef2836",
    "dataset": "common-pile/foodista",
    "dataset_revision": "04d1b7a6562c6d6459426d2a3b88184b8a98f7b6",
    "shard_path": "v0/documents/00000_foodista.jsonl.gz",
    "shard_sha256": "286b801bc826f161efad892af5529b201f91f42c5a932e3962d74d24037fe087",
    "shard_compressed_bytes": 6760466,
    "required_source_label": "foodista",
    "required_metadata_license": (
        "Creative Commons - Attribution - https://creativecommons.org/licenses/by/3.0/"
    ),
    "required_metadata_hosts": ["www.foodista.com", "foodista.com"],
}

EXPECTED_EVIDENCE_URLS = [
    "https://www.foodista.com/static/help",
    "https://www.foodista.com/blog/2009/01/29/secret-recipes",
    "https://huggingface.co/datasets/common-pile/foodista/blob/main/README.md",
]

EXPECTED_RISK = {
    "sitewide_cc_attribution_signal_observed": True,
    "exact_payload_metadata_cc_by_3_observed": True,
    "foodista_origin_host_observed": True,
    "automated_web_crawl_declared_by_primary_source": True,
    "dataset_card_license_laundering_warning_observed": True,
    "record_level_original_rights_provenance_available": False,
    "sitewide_license_alone_resolves_third_party_origin_rights": False,
    "package_metadata_alone_is_training_authority": False,
}

EXPECTED_DECISION = {
    "source_scope_qualified": False,
    "decision_class": "PROJECT_POLICY_BLOCKED_ORIGINAL_RIGHTS_PROVENANCE_UNRESOLVED",
    "legal_conclusion_claimed": False,
    "canonical_payload_admission_permitted": False,
    "zero_credit_mechanics_may_continue": True,
    "required_resolution": (
        "Provide record-level original-rights provenance or equivalent primary-source grant "
        "that resolves the automated-Web-crawl origin risk for every retained record."
    ),
    "policy_revalidation_required_on_product_or_rights_drift": True,
    "canonical_training_authorized": False,
    "corpus_credit_bytes": 0,
    "authorized_loss_positions": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates_authorized": False,
    "evaluation_eligible": False,
    "final_test_accessed": False,
}

EXPECTED_TRUTH = {
    "real_data_executed_by_this_package": False,
    "canonical_corpus_admitted": False,
    "training_authorized_bytes": 0,
    "authorized_optimized_target_exposure": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_payload_accessed": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights_used": False,
    "external_llm_or_api_used_for_data_or_intelligence": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def _canonical_policy_hash(policy: dict[str, Any]) -> str:
    payload = deepcopy(policy)
    payload.pop("policy_identity_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_zero(value: Any, name: str) -> None:
    _require(type(value) is int and value == 0, f"{name} must be strict integer zero")


def _strict_bool(value: Any, expected: bool, name: str) -> None:
    _require(type(value) is bool and value is expected, f"{name} must be {expected!r}")


def _exact_mapping(actual: Any, expected: dict[str, Any], name: str) -> None:
    _require(type(actual) is dict, f"{name} must be an object")
    _require(actual == expected, f"{name} drift")


def validate_policy(policy: dict[str, Any], registry_bytes: bytes) -> None:
    """Validate exact policy identities and preserve the zero-credit block."""

    _require(type(policy) is dict, "policy must be an object")
    _require(set(policy) == ROOT_KEYS, "policy root field drift")
    _require(policy["schema_version"] == SCHEMA_VERSION, "schema version drift")
    _require(policy["authority_id"] == AUTHORITY_ID, "authority id drift")
    _require(policy["status"] == STATUS, "status drift")
    _require(policy["execution_profile"] == "LOCAL_FREE", "execution profile drift")
    _require(
        policy["policy_identity_sha256"] == POLICY_IDENTITY,
        "declared policy identity drift",
    )
    _require(_canonical_policy_hash(policy) == POLICY_IDENTITY, "policy content drift")

    project = policy["project_authority"]
    _require(type(project) is dict, "project_authority must be an object")
    _require(project.get("swarm_protocol") == "SWARM-300-V2", "swarm protocol drift")
    _require(
        type(project.get("control_issue")) is int and project["control_issue"] == 723,
        "control issue drift",
    )
    _require(
        type(project.get("worker_issue")) is int and project["worker_issue"] == 1217,
        "worker issue drift",
    )
    _require(
        project.get("base_main_sha") == "7c14db5f6e43d369bf859cb5c6803024c929943c",
        "base main drift",
    )

    _exact_mapping(policy["parent_registry"], EXPECTED_PARENT, "parent_registry")
    _exact_mapping(policy["upstream_collector"], EXPECTED_UPSTREAM, "upstream_collector")
    _exact_mapping(policy["product_contract"], EXPECTED_PRODUCT, "product_contract")

    _require(HEX40.fullmatch(EXPECTED_PARENT["blob_sha1"]) is not None, "registry sha shape")
    _require(
        _git_blob_sha1(registry_bytes) == EXPECTED_PARENT["blob_sha1"],
        "checked-in generic registry blob drift",
    )
    registry = json.loads(registry_bytes.decode("utf-8"))
    _require(registry.get("registry_id") == EXPECTED_PARENT["registry_id"], "registry id drift")
    rows = [row for row in registry.get("sources", []) if row.get("key") == "foodista"]
    _require(len(rows) == 1, "Foodista registry row must be unique")
    row = rows[0]
    _require(row.get("hf_dataset") == "common-pile/foodista", "Foodista dataset drift")
    _require(row.get("rights_basis_class") == "SITEWIDE_OPEN_LICENSE", "rights class drift")
    _require(row.get("license_or_status_signals") == ["CC-BY"], "rights signal drift")
    _require(row.get("project_review_status") == "REVIEW_REQUIRED", "review status drift")
    _strict_bool(row.get("canonical_training_authorized"), False, "registry training authority")
    _strict_zero(row.get("credited_bytes"), "registry credited_bytes")
    _strict_zero(row.get("authorized_loss_positions"), "registry authorized_loss_positions")
    _strict_bool(row.get("final_test_excluded"), True, "registry final_test_excluded")

    product = policy["product_contract"]
    _require(HEX40.fullmatch(product["product_head_sha"]) is not None, "product head shape")
    _require(HEX40.fullmatch(product["config_blob_sha1"]) is not None, "product config sha shape")
    _require(HEX40.fullmatch(product["dataset_revision"]) is not None, "dataset revision shape")
    _require(HEX64.fullmatch(product["shard_sha256"]) is not None, "shard sha shape")
    _require(type(product["shard_compressed_bytes"]) is int, "shard bytes type")
    _require(product["shard_compressed_bytes"] > 0, "shard bytes must be positive")

    upstream = policy["upstream_collector"]
    for field in (
        "audited_commit",
        "readme_blob_sha1",
        "preprocess_blob_sha1",
        "to_dolma_blob_sha1",
    ):
        _require(HEX40.fullmatch(upstream[field]) is not None, f"{field} shape drift")

    evidence = policy["primary_evidence"]
    _require(type(evidence) is list and len(evidence) == 3, "primary evidence set drift")
    _require(
        [item.get("url") for item in evidence] == EXPECTED_EVIDENCE_URLS,
        "primary evidence URL drift",
    )
    for index, item in enumerate(evidence):
        _require(type(item) is dict, f"primary_evidence[{index}] must be object")
        _require(bool(item.get("authority")), f"primary_evidence[{index}] authority missing")
        _require(bool(item.get("supported_fact")), f"primary_evidence[{index}] supported fact missing")
        _require(
            bool(item.get("countervailing_fact")),
            f"primary_evidence[{index}] countervailing fact missing",
        )
        _require(bool(item.get("accessed_utc")), f"primary_evidence[{index}] access time missing")

    risk = policy["risk_assessment"]
    _require(type(risk) is dict and set(risk) == set(EXPECTED_RISK), "risk field drift")
    for key, expected in EXPECTED_RISK.items():
        _strict_bool(risk[key], expected, f"risk_assessment.{key}")

    decision = policy["decision"]
    _require(
        type(decision) is dict and set(decision) == set(EXPECTED_DECISION),
        "decision field drift",
    )
    for key, expected in EXPECTED_DECISION.items():
        if type(expected) is bool:
            _strict_bool(decision[key], expected, f"decision.{key}")
        elif type(expected) is int:
            _strict_zero(decision[key], f"decision.{key}")
        else:
            _require(decision[key] == expected, f"decision.{key} drift")

    truth = policy["truth_boundary"]
    _require(type(truth) is dict and set(truth) == set(EXPECTED_TRUTH), "truth boundary field drift")
    for key, expected in EXPECTED_TRUTH.items():
        if type(expected) is bool:
            _strict_bool(truth[key], expected, f"truth_boundary.{key}")
        else:
            _strict_zero(truth[key], f"truth_boundary.{key}")


def load_and_validate(
    policy_path: Path = DEFAULT_POLICY,
    registry_path: Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    validate_policy(policy, registry_path.read_bytes())
    return policy


def main() -> int:
    policy = load_and_validate()
    print(
        json.dumps(
            {
                "authority_id": policy["authority_id"],
                "status": policy["status"],
                "source_scope_qualified": policy["decision"]["source_scope_qualified"],
                "canonical_payload_admission_permitted": policy["decision"][
                    "canonical_payload_admission_permitted"
                ],
                "training_authorized_bytes": policy["truth_boundary"][
                    "training_authorized_bytes"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
