from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

import tools.pin_d03_rada_bulk_observation as mod

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = json.loads(
    (ROOT / "configs/data/d03_rada_bulk_observation_pin_v1.json").read_text(
        encoding="utf-8"
    )
)


def _archive_bytes(files: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return output.getvalue()


def _fixture() -> tuple[bytes, dict, bytes, dict]:
    files = {
        "d1.htm": b"<p>one</p>",
        "d2.htm": b"<p>two</p>",
        "readme.txt": b"ignored",
    }
    archive = _archive_bytes(files)
    entries = []
    for name in ("d1.htm", "d2.htm"):
        raw = files[name]
        entries.append(
            {
                "basename": name,
                "path": name,
                "raw_bytes": len(raw),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "crc32": "00000000",
            }
        )

    identity = hashlib.sha256()
    raw_total = 0
    for entry in entries:
        identity.update(entry["basename"].encode("utf-8"))
        identity.update(b"\0")
        identity.update(str(entry["raw_bytes"]).encode("ascii"))
        identity.update(b"\0")
        identity.update(entry["raw_sha256"].encode("ascii"))
        identity.update(b"\n")
        raw_total += entry["raw_bytes"]

    archive_md5 = hashlib.md5(archive, usedforsecurity=False).hexdigest()
    archive_sha256 = hashlib.sha256(archive).hexdigest()
    probe = {
        "schema_version": "12-6.d03-rada-bulk-source-probe-report.v1",
        "worker_id": "D03-RADA-BULK-SOURCE-PROBE-20260826",
        "config_identity_sha256": "a" * 64,
        "source_family": "ua.rada.open-data.laws-texts",
        "discovery_observation_revalidated": False,
        "safe_result": (
            "CURRENT_UPSTREAM_OBSERVED_BELOW_DISCOVERY_MINIMUM_"
            "SUCCESSOR_TRIAGE_REQUIRED"
        ),
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
        "archive": {
            "bytes": len(archive),
            "md5": archive_md5,
            "sha256": archive_sha256,
        },
        "inventory": {
            "canonical_entry_count": 2,
            "canonical_raw_bytes": raw_total,
            "entry_identity_sha256": identity.hexdigest(),
            "entries": entries,
        },
    }
    probe_bytes = (
        json.dumps(probe, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    config = {
        "schema_version": mod.CONFIG_SCHEMA,
        "worker_id": "TEST-D03-RADA-PIN",
        "local_free_only": True,
        "parent_probe": {
            "pr": 830,
            "head_sha": "b" * 40,
            "workflow_run_id": 1,
            "artifact_id": 2,
            "artifact_digest": "sha256:" + "c" * 64,
            "probe_report_sha256": hashlib.sha256(probe_bytes).hexdigest(),
            "probe_config_identity_sha256": "a" * 64,
        },
        "exact_snapshot": {
            "archive_bytes": len(archive),
            "archive_md5": archive_md5,
            "archive_sha256": archive_sha256,
            "canonical_entry_count": 2,
            "canonical_raw_bytes": raw_total,
            "entry_identity_sha256": identity.hexdigest(),
        },
        "expected_observation": {
            "source_family": "ua.rada.open-data.laws-texts",
            "exact_archive_identity": "OBSERVED_UNPINNED",
            "safe_zip_inventory": "PASS",
            "discovery_capacity_threshold": "FAIL_BELOW_MINIMUM",
            "safe_result": (
                "CURRENT_UPSTREAM_OBSERVED_BELOW_DISCOVERY_MINIMUM_"
                "SUCCESSOR_TRIAGE_REQUIRED"
            ),
            "discovery_observation_revalidated": False,
        },
        "pin_policy": {
            "purpose": "NORMALIZATION_INPUT_ONLY",
            "recompute_entry_identity": True,
            "verify_every_canonical_entry_against_archive": True,
            "preserve_failed_capacity_threshold": True,
            "derived_exact_archive_identity": "PASS_SUCCESSOR_EXACT_ARTIFACT_PIN",
            "derived_safe_result": (
                "SUCCESSOR_PINNED_EXACT_ARCHIVE_NORMALIZATION_ONLY_"
                "CAPACITY_STILL_UNCREDITED"
            ),
        },
        "claim_boundary": {
            "training_authorized_bytes": 0,
            "corpus_admitted": False,
            "normalized_capacity_credited": False,
            "tokenizer_fit_authorized": False,
            "model_training_executed": False,
            "paid_compute_used": False,
            "research_corpus_v1_released": False,
            "learned_20m_claimed": False,
        },
    }
    return archive, probe, probe_bytes, config


def _bind_fixture_authority(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    monkeypatch.setattr(mod, "EXPECTED_PARENT", copy.deepcopy(config["parent_probe"]))


def _rehash_probe(probe: dict, config: dict) -> bytes:
    probe_bytes = (
        json.dumps(probe, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    config["parent_probe"]["probe_report_sha256"] = hashlib.sha256(
        probe_bytes
    ).hexdigest()
    return probe_bytes


def test_production_config_binds_terminal_current_observation() -> None:
    for field, expected in mod.EXPECTED_PARENT.items():
        assert PRODUCTION_CONFIG["parent_probe"][field] == expected
    assert PRODUCTION_CONFIG["expected_observation"][
        "discovery_observation_revalidated"
    ] is False
    assert PRODUCTION_CONFIG["expected_observation"][
        "discovery_capacity_threshold"
    ] == "FAIL_BELOW_MINIMUM"
    assert PRODUCTION_CONFIG["claim_boundary"]["training_authorized_bytes"] == 0


def test_pin_preserves_failed_capacity_and_zero_credit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, probe, probe_bytes, config = _fixture()
    _bind_fixture_authority(monkeypatch, config)
    result = mod.pin_observation(archive, probe, probe_bytes, config)
    assert result["gates"]["exact_archive_identity"] == mod.DERIVED_ARCHIVE_GATE
    assert result["gates"]["discovery_capacity_threshold"] == "FAIL_BELOW_MINIMUM"
    assert result["training_authorized_bytes"] == 0
    assert result["successor_pin"]["normalized_capacity_credited"] == 0
    assert result["successor_pin"]["source_observation_revalidated"] is False


def test_pin_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    archive, probe, probe_bytes, config = _fixture()
    _bind_fixture_authority(monkeypatch, config)
    first = mod.pin_observation(archive, probe, probe_bytes, config)
    second = mod.pin_observation(archive, probe, probe_bytes, config)
    assert mod._canonical_json_bytes(first) == mod._canonical_json_bytes(second)


def test_report_sha_tamper_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    archive, probe, probe_bytes, config = _fixture()
    _bind_fixture_authority(monkeypatch, config)
    config["parent_probe"]["probe_report_sha256"] = "0" * 64
    with pytest.raises(mod.ObservationPinError):
        mod.pin_observation(archive, probe, probe_bytes, config)


def test_archive_tamper_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    archive, probe, probe_bytes, config = _fixture()
    _bind_fixture_authority(monkeypatch, config)
    with pytest.raises(mod.ObservationPinError):
        mod.pin_observation(archive + b"x", probe, probe_bytes, config)


def test_inventory_identity_tamper_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    archive, probe, _, config = _fixture()
    probe = copy.deepcopy(probe)
    config = copy.deepcopy(config)
    _bind_fixture_authority(monkeypatch, config)
    probe["inventory"]["entry_identity_sha256"] = "0" * 64
    probe_bytes = _rehash_probe(probe, config)
    with pytest.raises(mod.ObservationPinError):
        mod.pin_observation(archive, probe, probe_bytes, config)


def test_capacity_promotion_attack_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    archive, probe, _, config = _fixture()
    probe = copy.deepcopy(probe)
    config = copy.deepcopy(config)
    _bind_fixture_authority(monkeypatch, config)
    probe["gates"]["discovery_capacity_threshold"] = "PASS"
    probe_bytes = _rehash_probe(probe, config)
    with pytest.raises(mod.ObservationPinError):
        mod.pin_observation(archive, probe, probe_bytes, config)


def test_revalidation_lie_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    archive, probe, _, config = _fixture()
    probe = copy.deepcopy(probe)
    config = copy.deepcopy(config)
    _bind_fixture_authority(monkeypatch, config)
    probe["discovery_observation_revalidated"] = True
    probe_bytes = _rehash_probe(probe, config)
    with pytest.raises(mod.ObservationPinError):
        mod.pin_observation(archive, probe, probe_bytes, config)


def test_training_credit_attack_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    archive, probe, _, config = _fixture()
    probe = copy.deepcopy(probe)
    config = copy.deepcopy(config)
    _bind_fixture_authority(monkeypatch, config)
    probe["training_authorized_bytes"] = 1
    probe_bytes = _rehash_probe(probe, config)
    with pytest.raises(mod.ObservationPinError):
        mod.pin_observation(archive, probe, probe_bytes, config)


def test_hidden_canonical_archive_member_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _, probe, _, config = _fixture()
    probe = copy.deepcopy(probe)
    config = copy.deepcopy(config)
    _bind_fixture_authority(monkeypatch, config)
    changed_archive = _archive_bytes(
        {
            "d1.htm": b"<p>one</p>",
            "d2.htm": b"<p>two</p>",
            "d3.htm": b"unreported",
        }
    )
    archive_md5 = hashlib.md5(changed_archive, usedforsecurity=False).hexdigest()
    archive_sha256 = hashlib.sha256(changed_archive).hexdigest()
    probe["archive"] = {
        "bytes": len(changed_archive),
        "md5": archive_md5,
        "sha256": archive_sha256,
    }
    config["exact_snapshot"]["archive_bytes"] = len(changed_archive)
    config["exact_snapshot"]["archive_md5"] = archive_md5
    config["exact_snapshot"]["archive_sha256"] = archive_sha256
    probe_bytes = _rehash_probe(probe, config)
    with pytest.raises(mod.ObservationPinError):
        mod.pin_observation(changed_archive, probe, probe_bytes, config)
