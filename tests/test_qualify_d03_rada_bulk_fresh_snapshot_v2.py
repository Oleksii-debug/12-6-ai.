from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "tools/qualify_d03_rada_bulk_fresh_snapshot_v2.py"
SPEC = importlib.util.spec_from_file_location(
    "qualify_d03_rada_bulk_fresh_snapshot_v2", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def _config() -> dict[str, object]:
    return json.loads(
        (ROOT / "configs/data/d03_rada_bulk_fresh_snapshot_v2.json").read_text(
            encoding="utf-8"
        )
    )


def _rights() -> dict[str, object]:
    return json.loads(
        (ROOT / "configs/data/d03_rada_bulk_fresh_snapshot_rights_v2.json").read_text(
            encoding="utf-8"
        )
    )


def _entry(name: str, size: int, digest: str) -> dict[str, object]:
    return {
        "path": f"zak/perv/text/{name}",
        "basename": name,
        "raw_bytes": size,
        "raw_sha256": digest,
        "crc32": "1234abcd",
    }


def _report(
    *,
    http_etag: str = "same",
    archive_bytes: int = 46_768_257,
    archive_sha256: str = "3" * 64,
) -> dict[str, object]:
    config = _config()
    entries = [
        _entry("d1.htm", 3, "1" * 64),
        _entry("d2.htm", 4, "2" * 64),
    ]
    identity = hashlib.sha256()
    for entry in entries:
        identity.update(str(entry["basename"]).encode())
        identity.update(b"\\0")
        identity.update(str(entry["raw_bytes"]).encode("ascii"))
        identity.update(b"\\0")
        identity.update(str(entry["raw_sha256"]).encode("ascii"))
        identity.update(b"\\n")
    probe = config["probe_authority"]
    source = config["source"]
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
            "bytes": archive_bytes,
            "md5": "a" * 32,
            "sha256": archive_sha256,
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


def _qualify(
    first: dict[str, object] | None = None,
    second: dict[str, object] | None = None,
) -> dict[str, object]:
    a = first or _report()
    b = second or copy.deepcopy(a)
    return mod.qualify_two_clean_probes(
        _config(),
        _rights(),
        a,
        b,
        probe_a_bytes=_raw(a),
        probe_b_bytes=_raw(b),
    )


def test_two_clean_live_reports_establish_exact_identity_without_training_credit() -> None:
    first = _report(http_etag="first")
    second = _report(http_etag="second")
    result = _qualify(first, second)

    assert result["status"] == "QUALIFIED_EXACT_MUTABLE_SOURCE_SNAPSHOT_ZERO_CREDIT"
    assert result["archive_bytes"] == 46_768_257
    assert result["archive_sha256"] == "3" * 64
    assert result["exact_identity_established_by_two_clean_live_reads"] is True
    assert result["prior_mutable_observation_treated_as_noncurrent"] is True
    assert result["historical_qp_authority_preserved"] is True
    assert result["artifact_retention_rights_recheck_status"] == (
        "PASS_ATTRIBUTED_RETENTION_ONLY"
    )
    assert result["corpus_training_rights_recheck_required"] is True
    assert result["canonical_capacity_credited"] == 0
    assert result["training_authorized_bytes"] == 0
    assert result["training_executed"] is False


def test_live_identity_is_not_hard_pinned_to_yesterdays_mutable_observation() -> None:
    result = _qualify(
        _report(archive_bytes=99, archive_sha256="4" * 64),
        _report(archive_bytes=99, archive_sha256="4" * 64),
    )
    assert result["archive_bytes"] == 99
    assert result["archive_sha256"] == "4" * 64
    assert result["prior_mutable_observation_sha256"] == (
        "08d38fd32f64550597985bdcdf63a42bbda5d9bec260445de20cdb435c600fcc"
    )


def test_evidence_identity_self_hashes_core() -> None:
    result = _qualify()
    identity = result.pop("evidence_identity_sha256")
    assert hashlib.sha256(mod._canonical(result)).hexdigest() == identity


def test_two_probe_archive_identity_divergence_fails() -> None:
    first = _report(archive_sha256="3" * 64)
    second = _report(archive_sha256="4" * 64)
    with pytest.raises(
        mod.FreshSnapshotQualificationError,
        match="did not converge semantically",
    ):
        _qualify(first, second)


def test_inventory_entry_identity_tamper_fails() -> None:
    report = _report()
    report["inventory"]["entry_identity_sha256"] = "0" * 64
    with pytest.raises(mod.FreshSnapshotQualificationError, match="entry identity mismatch"):
        mod._validate_report(report, _config(), label="fixture")


def test_downstream_gate_promotion_fails() -> None:
    report = _report()
    report["gates"]["canonical_normalization"] = "PASS"
    with pytest.raises(mod.FreshSnapshotQualificationError, match="downstream gate ran"):
        mod._validate_report(report, _config(), label="fixture")


def test_training_byte_bool_alias_fails() -> None:
    report = _report()
    report["training_authorized_bytes"] = False
    with pytest.raises(mod.FreshSnapshotQualificationError, match="training bytes widened"):
        mod._validate_report(report, _config(), label="fixture")


def test_strict_json_rejects_duplicate_and_nonfinite_numbers() -> None:
    with pytest.raises(mod.FreshSnapshotQualificationError, match="duplicate JSON key"):
        mod._strict_json_bytes(b'{"a":1,"a":2}', label="fixture")
    with pytest.raises(mod.FreshSnapshotQualificationError, match="non-standard JSON constant"):
        mod._strict_json_bytes(b'{"a":NaN}', label="fixture")
    with pytest.raises(mod.FreshSnapshotQualificationError, match="non-finite JSON number"):
        mod._strict_json_bytes(b'{"a":1e400}', label="fixture")


def test_capture_config_marks_prior_mutable_observation_noncurrent() -> None:
    config = _config()
    assert config["prior_mutable_observation"]["status"] == (
        "SUPERSEDED_MUTABLE_OBSERVATION_DO_NOT_PIN_AS_CURRENT"
    )
    mod._validate_config(config)


def test_rights_policy_is_hash_bound_and_retention_only() -> None:
    rights = _rights()
    assert mod._validate_rights(rights) == mod.RIGHTS_IDENTITY
    assert rights["project_decision"]["scope"] == (
        "ARTIFACT_RETENTION_AND_REPRODUCIBILITY_ONLY"
    )
    assert rights["project_decision"]["bulk_corpus_admission_granted"] is False
    assert rights["project_decision"]["training_authority_granted"] is False
    assert rights["truth_boundary"]["training_authorized_bytes"] == 0


def test_rights_policy_tamper_fails_closed() -> None:
    rights = _rights()
    rights["project_decision"]["bulk_corpus_admission_granted"] = True
    with pytest.raises(mod.FreshSnapshotQualificationError, match="project decision drift"):
        mod._validate_rights(rights)


def test_attribution_binds_exact_captured_archive_and_zero_credit_scope() -> None:
    report = _report()
    text = mod.attribution_text(_rights(), report["archive"])
    assert "Апарат Верховної Ради України" in text
    assert "Captured archive bytes: 46768257" in text
    assert f"Captured archive SHA-256: {'3' * 64}" in text
    assert "Creative Commons Attribution 4.0 International" in text
    assert "grants no corpus or training credit" in text


def test_report_schema_is_closed_world() -> None:
    report = _report()
    report["unexpected"] = True
    with pytest.raises(mod.FreshSnapshotQualificationError, match="schema drift"):
        mod._validate_report(report, _config(), label="fixture")
