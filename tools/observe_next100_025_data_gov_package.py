#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import next100_025_data_gov_registry_snapshot as core


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def safe_resource(resource: dict) -> dict:
    return {
        "id": resource.get("id"),
        "name": resource.get("name"),
        "format": resource.get("format"),
        "url": resource.get("url"),
        "created": resource.get("created"),
        "last_modified": resource.get("last_modified"),
        "size": resource.get("size"),
        "hash": resource.get("hash"),
    }


def zero_credit_boundary() -> dict[str, object]:
    return {
        "candidate_snapshot_only": True,
        "canonical_corpus_admitted": False,
        "family_credit": False,
        "source_capacity_bytes_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_optimized_target_exposure": 0,
        "evaluation_authorized_bytes": 0,
        "global_dedup_complete": False,
        "reserved_evaluation_decontamination_complete": False,
        "post_composition_quality_privacy_complete": False,
        "balance_family_caps_complete": False,
        "cluster_safe_split_complete": False,
        "deterministic_packing_complete": False,
        "postpack_unique_loss_ledger_complete": False,
        "tokenizer_fit_authorized": False,
        "optimizer_updates": 0,
        "model_training_executed": False,
        "final_test_payload_accessed": False,
        "paid_compute_used": False,
    }


def observe(package: dict, cfg: dict, rights_sha256: str) -> dict:
    organization = package.get("organization") or {}
    publisher = organization.get("title") if isinstance(organization, dict) else None
    license_title = str(package.get("license_title") or "")
    configured_id = cfg["dataset"]["dataset_id"]
    observed_id = package.get("id")

    if publisher != cfg["dataset"]["publisher"]:
        status = "BLOCK_PUBLISHER_DRIFT"
    elif "creative commons attribution" not in license_title.casefold():
        status = "BLOCK_LICENSE_DRIFT"
    elif observed_id != configured_id:
        status = "RETEST_PACKAGE_IDENTITY_DRIFT"
    else:
        status = "OBSERVED_EXPECTED_PACKAGE_IDENTITY"

    resources = package.get("resources") or []
    resource_metadata = [
        safe_resource(resource)
        for resource in resources
        if isinstance(resource, dict)
    ]
    core_observation = {
        "schema_version": "12-6.next100-025-data-gov-package-observation.v1",
        "worker": cfg["worker"],
        "status": status,
        "configured_dataset_identifier": configured_id,
        "configured_dataset_page": cfg["dataset"]["dataset_page"],
        "requested_package_api": cfg["dataset"]["package_api"],
        "observed_ckan_package_id": observed_id,
        "observed_ckan_name": package.get("name"),
        "observed_title": package.get("title"),
        "observed_metadata_modified": package.get("metadata_modified"),
        "observed_publisher": publisher,
        "observed_license_title": license_title,
        "rights_evidence_sha256": rights_sha256,
        "resource_count": len(resource_metadata),
        "resources": resource_metadata,
        "claim_boundary": zero_credit_boundary(),
        "next": (
            "Do not mutate the configured identity from observation alone. First reconcile "
            "public dataset identifier, CKAN package identity, publisher, license, and "
            "selected resource identity. A later LOCKED source snapshot remains zero-credit "
            "and must still pass current global dedup, fresh reserved-evaluation "
            "decontamination, quality/privacy, balance/family caps, cluster-safe split, "
            "deterministic packing/two-clean-build, and positive unique-loss accounting "
            "before any training authorization."
        ),
    }
    digest = hashlib.sha256(canonical_json(core_observation)).hexdigest()
    return {**core_observation, "observation_sha256": digest}


def validate_static_contract(cfg: dict) -> str:
    if cfg["local_free_only"] is not True:
        raise RuntimeError("LOCAL_FREE gate is not true")
    core.validate_mode(cfg.get("mode"))
    api = urlparse(cfg["dataset"]["package_api"])
    if api.scheme != "https" or api.hostname not in {"data.gov.ua", "www.data.gov.ua"}:
        raise RuntimeError("package API escaped data.gov.ua boundary")
    rights_path = Path(cfg["rights_evidence"]["path"])
    rights_bytes = rights_path.read_bytes()
    rights_sha = hashlib.sha256(rights_bytes).hexdigest()
    if rights_sha != cfg["rights_evidence"]["sha256"]:
        raise RuntimeError("rights evidence SHA mismatch")
    uses = cfg["rights"]["uses"]
    for purpose in ("acquisition", "storage", "analysis", "model_training"):
        if uses.get(purpose) != "ALLOWED":
            raise RuntimeError(f"required right changed: {purpose}")
    if uses.get("redistribution") != "ALLOWED_WITH_ATTRIBUTION":
        raise RuntimeError("redistribution attribution boundary changed")
    if uses.get("evaluation") != "NOT_ADMITTED":
        raise RuntimeError("evaluation boundary weakened")
    return rights_sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--package-json",
        help="Optional saved CKAN package_show response for deterministic offline replay.",
    )
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rights_sha = validate_static_contract(cfg)
    if args.package_json:
        payload = Path(args.package_json).read_bytes()
    else:
        payload = core.fetch(cfg["dataset"]["package_api"], 2_000_000)
    response = core.load_json_bytes(payload)
    if not isinstance(response, dict) or response.get("success") is not True:
        raise RuntimeError("CKAN package_show did not succeed")
    package = response.get("result")
    if not isinstance(package, dict):
        raise RuntimeError("CKAN package result missing")

    observation = observe(package, cfg, rights_sha)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(observation, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(observation, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
