from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "tools" / "materialize_d03_common_pile_foodista_v1.py"
spec = importlib.util.spec_from_file_location("foodista_intake", MODULE)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def raw_cfg() -> dict:
    return json.loads(mod.CONFIG.read_text(encoding="utf-8"))


def row(*, record_id: int = 1, text: str | None = None) -> dict:
    if text is None:
        text = (
            "Seasonal tomato soup is a simple home recipe with fresh tomatoes, herbs, "
            "olive oil, onion, garlic, and careful simmering. " * 6
        )
    return {
        "id": record_id,
        "text": text,
        "source": "foodista",
        "added": "2024-05-23T00:42:18.321809",
        "created": "2010-01-01T00:00:00",
        "metadata": {
            "authors": ["Fixture Author"],
            "license": mod.EXPECTED_LICENSE,
            "provenance": f"foodista-dolma-0000.json.gz:{record_id + 1}",
            "url": f"https://www.foodista.com/recipe/fixture-{record_id}",
        },
    }


def test_config_is_exact_and_zero_credit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_validate_parent_rights", lambda cfg: "a" * 64)
    cfg = mod.load_config()
    assert cfg["source"]["revision"] == mod.SOURCE_REVISION
    assert cfg["source"]["sha256"] == mod.SOURCE_SHA256
    assert cfg["source"]["compressed_bytes"] == mod.SOURCE_COMPRESSED_BYTES
    assert cfg["collector_authority"]["readme_git_blob_sha1"] == (
        "155d861c68194b68a18df1fcfa3c32e9105c645d"
    )
    assert cfg["claim_boundary"]["training_authorized_bytes"] == 0
    assert cfg["claim_boundary"]["training_eligible"] is False
    assert cfg["claim_boundary"]["evaluation_eligible"] is False


def test_config_truth_boundary_bool_alias_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mod, "_validate_parent_rights", lambda cfg: "a" * 64)
    cfg = raw_cfg()
    cfg["claim_boundary"]["training_authorized_bytes"] = False
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(mod.FoodistaIntakeError, match="integer zero"):
        mod.load_config(path)


def test_accepted_row_emits_no_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_validate_parent_rights", lambda cfg: "a" * 64)
    cfg = mod.load_config()
    accepted, reason, normalized = mod.assess_row(row(), cfg)
    assert accepted is True
    assert reason == "accepted"
    assert normalized
    chosen, reasons, scanned = mod.select_rows([row()], cfg)
    assert scanned == 1
    assert reasons == {"accepted": 1}
    payload = json.loads(mod.candidate_payload(chosen).decode("utf-8"))
    assert payload["source_key"] == "foodista"
    assert "metadata" not in payload
    assert "Fixture Author" not in payload.values()
    assert all("foodista.com" not in str(value) for value in payload.values())
    assert payload["training_eligible"] is False
    assert payload["evaluation_eligible"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", "other"),
        ("license", "CC-BY"),
        ("url", "https://evil.example/recipe/1"),
        ("url", "https://www.foodista.com:bad/recipe/1"),
        ("provenance", "forged.json.gz:1"),
    ],
)
def test_source_rights_or_origin_drift_rejected(
    monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    monkeypatch.setattr(mod, "_validate_parent_rights", lambda cfg: "a" * 64)
    cfg = mod.load_config()
    candidate = row()
    if field == "source":
        candidate[field] = value
    else:
        candidate["metadata"][field] = value
    ok, reason, _ = mod.assess_row(candidate, cfg)
    assert ok is False
    assert reason in {"source_label_mismatch", "metadata_rights_or_origin_mismatch"}


def test_metadata_unknown_field_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_validate_parent_rights", lambda cfg: "a" * 64)
    cfg = mod.load_config()
    candidate = row()
    candidate["metadata"]["extra"] = "x"
    with pytest.raises(mod.FoodistaIntakeError, match="metadata field drift"):
        mod.assess_row(candidate, cfg)


def test_bool_record_id_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_validate_parent_rights", lambda cfg: "a" * 64)
    cfg = mod.load_config()
    candidate = row()
    candidate["id"] = True
    with pytest.raises(mod.FoodistaIntakeError, match="record id invalid"):
        mod.assess_row(candidate, cfg)


@pytest.mark.parametrize(
    ("needle", "reason"),
    [
        (" contact owner@example.com for details ", "email"),
        (" server 192.168.1.10 should never enter training ", "ipv4"),
        (" password=SuperSecret123 must be rejected ", "secret_marker"),
        (" hidden\x00control must be rejected ", "control_character"),
    ],
)
def test_privacy_markers_rejected(
    monkeypatch: pytest.MonkeyPatch, needle: str, reason: str
) -> None:
    monkeypatch.setattr(mod, "_validate_parent_rights", lambda cfg: "a" * 64)
    cfg = mod.load_config()
    candidate = row(text=row()["text"] + needle)
    ok, observed, _ = mod.assess_row(candidate, cfg)
    assert ok is False
    assert observed == reason


def test_duplicate_id_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_validate_parent_rights", lambda cfg: "a" * 64)
    cfg = mod.load_config()
    with pytest.raises(mod.FoodistaIntakeError, match="duplicate source record id"):
        mod.select_rows([row(record_id=1), row(record_id=1, text=row()["text"] + " extra")], cfg)


def test_normalized_duplicate_is_accounted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_validate_parent_rights", lambda cfg: "a" * 64)
    cfg = mod.load_config()
    a = row(record_id=1)
    b = row(record_id=2, text="  " + a["text"].replace(" ", "  ") + "  ")
    accepted, reasons, scanned = mod.select_rows([a, b], cfg)
    assert scanned == 2
    assert len(accepted) == 1
    assert reasons["accepted"] == 1
    assert reasons["exact_normalized_duplicate"] == 1


def test_exact_source_size_or_hash_mismatch_fails(tmp_path: Path) -> None:
    path = tmp_path / "fake.gz"
    path.write_bytes(b"not the source")
    with pytest.raises(mod.FoodistaIntakeError):
        mod.verify_exact_source(path)


def test_two_same_byte_runs_are_identical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mod, "_validate_parent_rights", lambda cfg: "b" * 64)
    monkeypatch.setattr(mod, "verify_exact_source", lambda path: mod.SOURCE_COMPRESSED_BYTES)
    rows = [row(record_id=1), row(record_id=2, text=row()["text"] + " Second article.")]
    shard = tmp_path / "source.jsonl.gz"
    with gzip.open(shard, "wb") as handle:
        for item in rows:
            handle.write(json.dumps(item, separators=(",", ":")).encode("utf-8") + b"\n")

    reports = []
    payloads = []
    for suffix in ("a", "b"):
        output = tmp_path / f"candidate-{suffix}.jsonl"
        report = tmp_path / f"report-{suffix}.json"
        reports.append(mod.run(shard, output, report, download=False))
        payloads.append(output.read_bytes())
    assert payloads[0] == payloads[1]
    assert reports[0] == reports[1]
    assert reports[0]["retained_records"] == 2
    assert reports[0]["durable_report_contains_source_text"] is False
    assert reports[0]["training_authorized_bytes"] == 0
