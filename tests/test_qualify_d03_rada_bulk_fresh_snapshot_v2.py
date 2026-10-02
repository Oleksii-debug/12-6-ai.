from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools/qualify_d03_rada_bulk_fresh_snapshot_v2.py"
SPEC = importlib.util.spec_from_file_location(
    "qualify_d03_rada_bulk_fresh_snapshot_v2", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def _config() -> dict[str, object]:
    path = Path(__file__).parents[1] / "configs/data/d03_rada_bulk_fresh_snapshot_v2.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _entry(name: str, size: int, digest: str) -> dict[str, object]:
    return {
        "path": f"zak/perv/text/{name}",
        "basename": name,
        "raw_bytes": size,
        "raw_sha256": digest,
        "crc32": "1234abcd",
    }


def _report(*, http_etag: str = "same") -> dict[str, object]:
    config = _config()
    entries = [
        _entry("d1.htm", 3, "1" * 64),
        _entry("d2.htm", 4, "2" * 64),
    ]
    identity = hashlib.sha256()
    for entry in entries:
        identity.update(str(entry["basename"]).encode())
        identity.update(b"\0")
        identity.update(str(entry["raw_bytes"]).encode("ascii"))
        identity.update(b"\0")
        identity.update(str(entry["raw_sha256"]).encode("ascii"))
        identity.update(b"\n")
    probe = config["probe_authority"]
    source = config["source"]
    current = config["expected_current_observation"]
    return {
        "schema_version": mod.PROBE_SCHEMA,
        "worker_id": probe["worker_id"],
        "config_identity_sha256": probe["config_identity_sha256"],
        "parent_authority": {
            "head_sha": probe["parent_head_sha"],
            "registry_identity_sha256": probe["parent_registry_identity_sha256"],
        },
        "source_family": source["family_id"],
        "source_dataset_id": source["dataset_id"],
        "archive": {
            "url": source["archive_url"],
            "bytes": current["archive_bytes"],
            "md5": "a" * 32,
            "sha256": current["archive_sha256"],
        },
        "inventory": {
            "canonical_entry_count": 2,
            "canonical_raw_bytes": 7,
            "ignored_file_count": 0,
            "total_zip_uncompressed_bytes": 7,
            "entry_identity_sha256": identity.hexdigest(),
            "entries": entries,
        },
        "gates": {
            "exact_archive_identity": "OBSERVED_UNPINNED",
            "safe_zip_inventory": "PASS",
            "discovery_capacity_threshold": "FAIL_BELOW_MINIMUM",
            "canonical_normalization": "NOT_RUN",
            "quality": "NOT_RUN",
            "privacy": "NOT_RUN",
            "global_cross_source_dedup": "NOT_RUN",
            "evaluation_decontamination": "NOT_RUN",
            "balance_diversity": "NOT_RUN",
            "corpus_materialization": "NOT_RUN",
            "unique_loss_ledger": "NOT_RUN",
        },
        "training_authorized_bytes": 0,
        "corpus_admitted": False,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "paid_compute_used": False,
        "safe_result": (
            "CURRENT_UPSTREAM_OBSERVED_BELOW_DISCOVERY_MINIMUM_"
            "SUCCESSOR_TRIAGE_REQUIRED"
        ),
        "http_response": {"etag": http_etag},
        "discovery_observation_revalidated": False,
    }


def _raw(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def test_two_clean_reports_qualify_same_snapshot_without_training_credit() -> None:
    config = _config()
    first = _report(http_etag="first")
    second = _report(http_etag="second")
    result = mod.qualify_two_clean_probes(
        config,
        first,
        second,
        probe_a_bytes=_raw(first),
        probe_b_bytes=_raw(second),
    )
    assert result["status"] == "QUALIFIED_EXACT_MUTABLE_SOURCE_SNAPSHOT_ZERO_CREDIT"
    assert result["two_fresh_probes_semantically_identical"] is True
    assert result["historical_qp_authority_preserved"] is True
    assert result["source_snapshot_repin_required"] is True
    assert result["rights_recheck_required"] is True
    assert result["canonical_capacity_credited"] == 0
    assert result["authorized_optimized_target_exposure"] == 0
    assert result["tokenizer_fit_authorized"] is False
    assert result["training_executed"] is False


def test_evidence_identity_self_hashes_core() -> None:
    config = _config()
    first = _report()
    result = mod.qualify_two_clean_probes(
        config,
        first,
        copy.deepcopy(first),
        probe_a_bytes=_raw(first),
        probe_b_bytes=_raw(first),
    )
    identity = result.pop("evidence_identity_sha256")
    assert hashlib.sha256(mod._canonical(result)).hexdigest() == identity


def test_rejects_archive_sha_drift() -> None:
    config = _config()
    first = _report()
    first["archive"]["sha256"] = "f" * 64
    with pytest.raises(mod.FreshSnapshotQualificationError, match="archive SHA-256 drift"):
        mod.qualify_two_clean_probes(
            config,
            first,
            _report(),
            probe_a_bytes=_raw(first),
            probe_b_bytes=_raw(_report()),
        )


def test_rejects_two_probe_semantic_divergence() -> None:
    config = _config()
    first = _report()
    second = _report()
    second["gates"]["discovery_capacity_threshold"] = "PASS"
    second["safe_result"] = "CURRENT_UPSTREAM_OBSERVED_SUCCESSOR_PIN_REQUIRED"
    with pytest.raises(
        mod.FreshSnapshotQualificationError,
        match="did not converge semantically",
    ):
        mod.qualify_two_clean_probes(
            config,
            first,
            second,
            probe_a_bytes=_raw(first),
            probe_b_bytes=_raw(second),
        )


def test_rejects_inventory_entry_identity_tamper() -> None:
    report = _report()
    report["inventory"]["entry_identity_sha256"] = "0" * 64
    with pytest.raises(mod.FreshSnapshotQualificationError, match="entry identity mismatch"):
        mod._validate_report(report, _config(), label="fixture")


def test_rejects_downstream_gate_promotion() -> None:
    report = _report()
    report["gates"]["canonical_normalization"] = "PASS"
    with pytest.raises(mod.FreshSnapshotQualificationError, match="downstream gate ran"):
        mod._validate_report(report, _config(), label="fixture")


def test_rejects_training_byte_bool_alias() -> None:
    report = _report()
    report["training_authorized_bytes"] = False
    with pytest.raises(mod.FreshSnapshotQualificationError, match="training bytes widened"):
        mod._validate_report(report, _config(), label="fixture")


def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers() -> None:
    with pytest.raises(mod.FreshSnapshotQualificationError, match="duplicate JSON key"):
        mod._strict_json_bytes(b'{"a":1,"a":2}', label="fixture")
    with pytest.raises(mod.FreshSnapshotQualificationError, match="non-standard JSON constant"):
        mod._strict_json_bytes(b'{"a":NaN}', label="fixture")
    with pytest.raises(mod.FreshSnapshotQualificationError, match="non-finite JSON number"):
        mod._strict_json_bytes(b'{"a":1e400}', label="fixture")


def test_config_refuses_rewriting_historical_snapshot_as_fresh() -> None:
    config = _config()
    config["expected_current_observation"]["archive_sha256"] = config[
        "predecessor_authority"
    ]["archive_sha256"]
    with pytest.raises(mod.FreshSnapshotQualificationError, match="must not rewrite"):
        mod._validate_config(config)


def test_report_schema_is_closed_world() -> None:
    report = _report()
    report["unexpected"] = True
    with pytest.raises(mod.FreshSnapshotQualificationError, match="schema drift"):
        mod._validate_report(report, _config(), label="fixture")
