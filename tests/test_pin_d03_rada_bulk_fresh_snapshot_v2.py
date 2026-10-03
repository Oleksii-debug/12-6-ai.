from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "tools" / "pin_d03_rada_bulk_fresh_snapshot_v2.py"
SPEC = importlib.util.spec_from_file_location("test_rada_pin_module", PIN)
assert SPEC is not None and SPEC.loader is not None
sys.path.insert(0, str(PIN.parent))
try:
    pin = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(pin)
finally:
    sys.path.remove(str(PIN.parent))


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _raw(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fixture() -> tuple[bytes, dict, dict, dict, dict, dict, bytes]:
    config = _json("configs/data/d03_rada_bulk_fresh_snapshot_v2.json")
    rights = _json("configs/data/d03_rada_bulk_fresh_snapshot_rights_v2.json")
    source = {"d1.htm": b"one", "d2.htm": b"four"}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, value in sorted(source.items()):
            zf.writestr("zak/perv/text/" + name, value)
    archive = stream.getvalue()
    entries = []
    digest = hashlib.sha256()
    for name, value in sorted(source.items()):
        sha = hashlib.sha256(value).hexdigest()
        entries.append({
            "path": "zak/perv/text/" + name,
            "basename": name,
            "raw_bytes": len(value),
            "raw_sha256": sha,
            "crc32": f"{zlib.crc32(value):08x}",
        })
        digest.update(name.encode() + b"\0" + str(len(value)).encode() + b"\0")
        digest.update(sha.encode() + b"\n")
    report = {
        "schema_version": pin.PROBE_SCHEMA if hasattr(pin, "PROBE_SCHEMA") else (
            "12-6.d03-rada-bulk-source-probe-report.v1"
        ),
        "worker_id": config["probe_authority"]["worker_id"],
        "config_identity_sha256": config["probe_authority"]["config_identity_sha256"],
        "parent_authority": {
            "head_sha": config["probe_authority"]["parent_head_sha"],
            "registry_identity_sha256": (
                config["probe_authority"]["parent_registry_identity_sha256"]
            ),
        },
        "source_family": config["source"]["family_id"],
        "source_dataset_id": config["source"]["dataset_id"],
        "archive": {
            "url": config["source"]["archive_url"],
            "bytes": len(archive),
            "md5": hashlib.md5(archive, usedforsecurity=False).hexdigest(),
            "sha256": hashlib.sha256(archive).hexdigest(),
        },
        "inventory": {
            "canonical_entry_count": 2,
            "canonical_raw_bytes": sum(map(len, source.values())),
            "ignored_file_count": 0,
            "total_zip_uncompressed_bytes": sum(map(len, source.values())),
            "entry_identity_sha256": digest.hexdigest(),
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
        "http_response": {"etag": "first"},
        "discovery_observation_revalidated": False,
    }
    second = copy.deepcopy(report)
    second["http_response"] = {"etag": "second"}
    qualification = pin.qualify_two_clean_probes(
        config, rights, report, second,
        probe_a_bytes=_raw(report), probe_b_bytes=_raw(second),
    )
    attribution = pin.attribution_text(rights, report["archive"]).encode("utf-8")
    return archive, report, second, qualification, config, rights, attribution


def _pin(archive, first, second, qualification, config, rights, attribution, **extra):
    return pin.pin_capture(
        archive, first, qualification, attribution, config, rights,
        execution_head_sha="a" * 40,
        probe_b=second,
        probe_a_bytes=extra.get("raw_a", _raw(first)),
        probe_b_bytes=extra.get("raw_b", _raw(second)),
    )


def test_real_zip_two_distinct_reports_pin_normalization_only_zero_credit() -> None:
    archive, first, second, qual, config, rights, attr = _fixture()
    result = _pin(archive, first, second, qual, config, rights, attr)
    assert result["status"] == pin.STATUS
    assert result["archive_sha256"] == hashlib.sha256(archive).hexdigest()
    assert result["source_qualification_identity_sha256"] == (
        qual["evidence_identity_sha256"]
    )
    assert result["historical_qp_authority_preserved"] is True
    assert result["training_authorized_bytes"] == 0
    assert result["authorized_optimized_target_exposure"] == 0
    assert result["tokenizer_fit_authorized"] is False
    assert result["training_executed"] is False


def test_forged_resealed_second_report_digest_never_creates_pin() -> None:
    archive, first, second, qual, config, rights, attr = _fixture()
    forged = copy.deepcopy(qual)
    forged["probe_b_file_sha256"] = "f" * 64
    unsigned = dict(forged)
    unsigned.pop("evidence_identity_sha256")
    forged["evidence_identity_sha256"] = hashlib.sha256(
        pin._canonical(unsigned)
    ).hexdigest()
    with pytest.raises(pin.FreshSnapshotPinError, match="two original raw probe"):
        _pin(archive, first, second, forged, config, rights, attr)


def test_second_report_change_fails_even_if_first_and_zip_are_valid() -> None:
    archive, first, second, qual, config, rights, attr = _fixture()
    second["archive"]["sha256"] = "f" * 64
    with pytest.raises(pin.FreshSnapshotPinError):
        _pin(archive, first, second, qual, config, rights, attr)


def test_probe_bytes_must_match_the_parsed_objects() -> None:
    archive, first, second, qual, config, rights, attr = _fixture()
    with pytest.raises(pin.FreshSnapshotPinError, match="probe A raw bytes/object"):
        _pin(
            archive, first, second, qual, config, rights, attr,
            raw_a=_raw(second),
        )


def test_pin_strict_input_rejects_duplicate_and_nonfinite_json(tmp_path: Path) -> None:
    for name, raw in (
        ("duplicate", b'{"archive":{"sha256":"a","sha256":"b"}}'),
        ("nan", b'{"value":NaN}'),
        ("overflow", b'{"value":1e400}'),
    ):
        path = tmp_path / (name + ".json")
        path.write_bytes(raw)
        with pytest.raises(pin.FreshSnapshotPinError, match="cannot load"):
            pin._load_json(path, name)


def test_cli_requires_second_probe_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        str(PIN), "--archive", "a", "--probe-report", "a",
        "--qualification", "a", "--attribution", "a",
        "--config", "a", "--rights-policy", "a",
        "--execution-head-sha", "a" * 40, "--output", "out",
    ])
    with pytest.raises(SystemExit) as exc:
        pin.main()
    assert exc.value.code == 2


def test_cli_two_probe_success_publishes_only_a_fresh_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    archive, first, second, qual, config, rights, attr = _fixture()
    inputs = {
        "archive.zip": archive,
        "a.json": _raw(first),
        "b.json": _raw(second),
        "qualification.json": _raw(qual),
        "ATTRIBUTION.txt": attr,
        "config.json": _raw(config),
        "rights.json": _raw(rights),
    }
    for name, raw in inputs.items():
        (tmp_path / name).write_bytes(raw)
    output = tmp_path / "pin.json"
    monkeypatch.setattr(sys, "argv", [
        str(PIN), "--archive", str(tmp_path / "archive.zip"),
        "--probe-report", str(tmp_path / "a.json"),
        "--probe-b", str(tmp_path / "b.json"),
        "--qualification", str(tmp_path / "qualification.json"),
        "--attribution", str(tmp_path / "ATTRIBUTION.txt"),
        "--config", str(tmp_path / "config.json"),
        "--rights-policy", str(tmp_path / "rights.json"),
        "--execution-head-sha", "a" * 40,
        "--output", str(output),
    ])
    assert pin.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == pin.STATUS
    assert json.loads(output.read_text(encoding="utf-8"))["training_authorized_bytes"] == 0
    with pytest.raises(pin.FreshSnapshotPinError, match="refusing to overwrite"):
        pin.main()
