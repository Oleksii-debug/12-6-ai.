from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
CONFIG = ROOT / "configs/data/next100_025_derzhgeocadastre_open_registry_v1.json"
SPEC = importlib.util.spec_from_file_location(
    "derzh_snapshot",
    TOOLS / "next100_025_data_gov_registry_snapshot.py",
)
assert SPEC and SPEC.loader
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


class _FakeResponse:
    def __init__(self, final_url: str, payload: bytes = b"payload") -> None:
        self.final_url = final_url
        self.payload = payload
        self.read_called = False

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def geturl(self) -> str:
        return self.final_url

    def read(self, limit: int) -> bytes:
        self.read_called = True
        return self.payload[:limit]


def _config() -> dict:
    return {
        "family": {"family_id": "ua.data-gov.derzhgeocadastre.dataset-register"},
        "dataset": {"dataset_id": "dataset-id"},
        "rights": {
            "dataset_license_label": "Creative Commons Attribution",
        },
    }


def _item() -> dict:
    return {
        "normalized_sha256": "1" * 64,
        "text": "назва: Реєстр наборів даних України\n",
    }


def _live_drift_resources() -> list[dict]:
    return [
        {
            "id": "current-csv",
            "name": "register",
            "format": ".csv",
            "url": "https://data.gov.ua/dataset/x/resource/current-csv/download/register.csv",
            "last_modified": "2026-09-03T11:38:00",
        },
        {
            "id": "schema-csv",
            "name": "Структура набора даних",
            "format": ".csv",
            "url": "https://data.gov.ua/dataset/x/resource/schema-csv/download/schema.csv",
            "last_modified": "2026-09-03T11:38:00",
        },
        {
            "id": "archived-json",
            "name": "Архівний - Реєстр наборів даних, які перебувають у володінні розпорядника інформації",
            "format": "JSON",
            "url": "https://data.gov.ua/dataset/x/resource/archived-json/download/register.json",
            "last_modified": "2025-01-01T00:00:00",
        },
    ]


def test_locked_snapshot_is_identity_lock_not_training_authority() -> None:
    boundary = snapshot.current_main_claim_boundary("LOCKED")

    assert boundary["source_snapshot_identity_locked"] is True
    assert boundary["source_specific_training_rights_compatible"] is True
    assert boundary["candidate_snapshot_only"] is True
    assert boundary["canonical_corpus_admitted"] is False
    assert boundary["family_credit"] is False
    assert boundary["source_capacity_bytes_credited"] == 0
    assert boundary["training_authorized_bytes"] == 0
    assert boundary["authorized_optimized_target_exposure"] == 0
    assert boundary["global_dedup_complete"] is False
    assert boundary["reserved_evaluation_decontamination_complete"] is False
    assert boundary["cluster_safe_split_complete"] is False
    assert boundary["deterministic_packing_complete"] is False
    assert boundary["postpack_unique_loss_ledger_complete"] is False
    assert boundary["tokenizer_fit_authorized"] is False
    assert boundary["optimizer_updates"] == 0
    assert boundary["model_training_executed"] is False
    assert snapshot.snapshot_status("LOCKED") == "SOURCE_SNAPSHOT_LOCKED_ZERO_CREDIT"


def test_locked_candidate_row_cannot_become_training_or_evaluation_eligible() -> None:
    row = snapshot.build_candidate_row(
        cfg=_config(),
        resource={"id": "resource-id"},
        resource_url="https://data.gov.ua/resource.json",
        raw_hash="2" * 64,
        item=_item(),
    )

    assert row["artifact_role"] == "SOURCE_CANDIDATE_ONLY"
    assert row["source_training_rights_compatible"] is True
    assert row["training_eligible"] is False
    assert row["evaluation_eligible"] is False


def test_probe_and_locked_modes_have_same_zero_credit_training_boundary() -> None:
    probe = snapshot.current_main_claim_boundary("PROBE")
    locked = snapshot.current_main_claim_boundary("LOCKED")

    for key in (
        "source_capacity_bytes_credited",
        "training_authorized_bytes",
        "authorized_optimized_target_exposure",
        "evaluation_authorized_bytes",
        "optimizer_updates",
    ):
        assert probe[key] == locked[key] == 0

    for key in (
        "canonical_corpus_admitted",
        "family_credit",
        "global_dedup_complete",
        "reserved_evaluation_decontamination_complete",
        "post_composition_quality_privacy_complete",
        "balance_family_caps_complete",
        "cluster_safe_split_complete",
        "deterministic_packing_complete",
        "postpack_unique_loss_ledger_complete",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "paid_compute_used",
    ):
        assert probe[key] is locked[key] is False


def test_unknown_mode_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="unsupported snapshot mode"):
        snapshot.current_main_claim_boundary("TRAIN")


def test_direct_mode_based_training_eligibility_regression_is_absent() -> None:
    source = (TOOLS / "next100_025_data_gov_registry_snapshot.py").read_text(
        encoding="utf-8"
    )
    assert '"training_eligible": cfg["mode"] == "LOCKED"' not in source
    assert '"training_eligible": False' in source


def test_current_csv_register_is_admissible_but_schema_and_archived_json_are_not() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    package = {"resources": _live_drift_resources()}

    selected = snapshot.pick_resource(package, cfg)

    assert selected["id"] == "current-csv"
    assert selected["format"] == ".csv"


def test_locked_expected_id_cannot_bypass_archive_admissibility() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg["mode"] = "LOCKED"
    cfg["resource_selection"]["expected_resource_id"] = "archived-json"
    package = {"resources": _live_drift_resources()}

    with pytest.raises(RuntimeError, match="locked resource id is not admissible"):
        snapshot.pick_resource(package, cfg)


def test_locked_expected_id_cannot_bypass_schema_exclusion() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    cfg["mode"] = "LOCKED"
    cfg["resource_selection"]["expected_resource_id"] = "schema-csv"
    package = {"resources": _live_drift_resources()}

    with pytest.raises(RuntimeError, match="locked resource id is not admissible"):
        snapshot.pick_resource(package, cfg)


def test_fetch_rejects_final_redirect_outside_data_gov(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse("https://example.org/redirected.csv")
    monkeypatch.setattr(snapshot.urllib.request, "urlopen", lambda req, timeout: response)

    with pytest.raises(RuntimeError, match="final response URL escaped data.gov.ua boundary"):
        snapshot.fetch("https://data.gov.ua/resource.csv", 100)

    assert response.read_called is False


def test_fetch_accepts_same_origin_final_url(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse("https://www.data.gov.ua/redirected.csv", b"ok")
    monkeypatch.setattr(snapshot.urllib.request, "urlopen", lambda req, timeout: response)

    assert snapshot.fetch("https://data.gov.ua/resource.csv", 100) == b"ok"
    assert response.read_called is True


def test_csv_records_support_utf8_bom_and_comma_delimiter() -> None:
    payload = (
        "\ufeffname,description,email\r\n"
        "Набір даних,Опис державного набору,private@example.gov.ua\r\n"
        "Реєстр,Опис реєстру,other@example.gov.ua\r\n"
    ).encode("utf-8")

    records = snapshot.load_csv_records(payload)

    assert records == [
        {
            "name": "Набір даних",
            "description": "Опис державного набору",
            "email": "private@example.gov.ua",
        },
        {
            "name": "Реєстр",
            "description": "Опис реєстру",
            "email": "other@example.gov.ua",
        },
    ]


def test_csv_records_support_cp1251_semicolon_delimiter() -> None:
    payload = (
        "name;description;format\r\n"
        "Реєстр;Опис набору даних;CSV\r\n"
        "Набір;Опис державних даних;JSON\r\n"
    ).encode("cp1251")

    records = snapshot.load_csv_records(payload)

    assert records[0]["name"] == "Реєстр"
    assert records[0]["description"] == "Опис набору даних"
    assert records[1]["format"] == "JSON"


def test_csv_records_fail_closed_on_duplicate_headers() -> None:
    payload = "name,name\nРеєстр,Інша назва\n".encode()

    with pytest.raises(RuntimeError, match="CSV headers are blank or duplicated"):
        snapshot.load_csv_records(payload)


def test_csv_records_fail_closed_on_ragged_extra_columns() -> None:
    payload = "name,description\nРеєстр,Опис,EXTRA\n".encode()

    with pytest.raises(RuntimeError, match="CSV row has unexpected extra columns"):
        snapshot.load_csv_records(payload)


def test_csv_records_fail_closed_on_ragged_missing_columns() -> None:
    payload = "name,description,format\nРеєстр,Опис\n".encode()

    with pytest.raises(RuntimeError, match="CSV row has missing columns"):
        snapshot.load_csv_records(payload)
