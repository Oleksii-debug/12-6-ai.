from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.d03_rada_normalization_contract import (
    NormalizationContractError,
    bind_manifest_to_contract,
    canonical_config_sha256,
    validate_production_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/d03_rada_bulk_normalization_v1.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_exact_production_config_is_accepted_and_identified() -> None:
    config = _config()
    validate_production_config(config)
    identity_a = canonical_config_sha256(config)
    identity_b = canonical_config_sha256(copy.deepcopy(config))
    assert identity_a == identity_b
    assert len(identity_a) == 64


@pytest.mark.parametrize(
    ("field", "key", "replacement"),
    [
        ("parent_probe", "head_sha", "0" * 40),
        ("parent_probe", "probe_worker_id", "OTHER-WORKER"),
        ("parent_probe", "probe_config_identity_sha256", "0" * 64),
        ("parent_probe", "source_family_identity_sha256", "0" * 64),
    ],
)
def test_parent_authority_mutations_fail_closed(
    field: str, key: str, replacement: object
) -> None:
    config = _config()
    config[field][key] = replacement
    with pytest.raises(NormalizationContractError, match="parent_probe drifted"):
        validate_production_config(config)


def test_hidden_tag_removal_fails_closed() -> None:
    config = _config()
    config["normalization"]["hidden_tags"].remove("script")
    with pytest.raises(NormalizationContractError, match="normalization drifted"):
        validate_production_config(config)


def test_block_tag_reordering_or_removal_fails_closed() -> None:
    for mutate in ("reverse", "remove"):
        config = _config()
        if mutate == "reverse":
            config["normalization"]["block_tags"].reverse()
        else:
            config["normalization"]["block_tags"].remove("p")
        with pytest.raises(NormalizationContractError, match="normalization drifted"):
            validate_production_config(config)


def test_output_record_contract_mutation_fails_closed() -> None:
    config = _config()
    config["output_contract"]["jsonl_record_fields"].remove("source_encoding")
    with pytest.raises(NormalizationContractError, match="output_contract drifted"):
        validate_production_config(config)


def test_downstream_gate_removal_fails_closed() -> None:
    config = _config()
    config["downstream_required"].remove("PRIVACY_PII_FILTER")
    with pytest.raises(NormalizationContractError, match="downstream_required drifted"):
        validate_production_config(config)


def test_truth_boundary_safe_result_mutation_fails_closed() -> None:
    config = _config()
    config["claim_boundary"]["safe_result"] = "CORPUS_READY"
    with pytest.raises(NormalizationContractError, match="claim_boundary drifted"):
        validate_production_config(config)


def test_extra_top_level_policy_surface_fails_closed() -> None:
    config = _config()
    config["best_effort"] = True
    with pytest.raises(NormalizationContractError, match="top-level"):
        validate_production_config(config)


def test_manifest_binding_replaces_stale_identity_with_contract_bound_identity() -> None:
    contract_sha = canonical_config_sha256(_config())
    manifest = {
        "schema_version": "12-6.d03-rada-bulk-normalization-manifest.v1",
        "manifest_identity_sha256": "0" * 64,
        "training_authorized_bytes": 0,
    }
    strengthened = bind_manifest_to_contract(
        manifest,
        normalization_contract_sha256=contract_sha,
    )
    assert strengthened["normalization_contract_sha256"] == contract_sha
    assert strengthened["training_authorized_bytes"] == 0
    assert strengthened["manifest_identity_sha256"] != "0" * 64

    expected_payload = dict(strengthened)
    expected_identity = expected_payload.pop("manifest_identity_sha256")
    canonical = json.dumps(
        expected_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert expected_identity == hashlib.sha256(canonical).hexdigest()


def test_manifest_binding_rejects_malformed_contract_identity() -> None:
    with pytest.raises(NormalizationContractError, match="must be SHA-256"):
        bind_manifest_to_contract({}, normalization_contract_sha256="not-a-sha")
