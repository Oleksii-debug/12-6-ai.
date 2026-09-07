from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "tools" / "materialize_d03_derzhgeocadastre_candidate.py"
spec = importlib.util.spec_from_file_location("derzh_candidate", MODULE)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def package_bytes(url: str = "https://data.gov.ua/dataset/resource.json") -> bytes:
    package = {
        "success": True,
        "result": {
            "name": mod.DATASET_ID,
            "title": mod.TITLE,
            "license_title": "Creative Commons Attribution 4.0",
            "organization": {"title": mod.PUBLISHER},
            "resources": [
                {
                    "id": "resource-1",
                    "name": "Реєстр JSON",
                    "format": "JSON",
                    "url": url,
                    "last_modified": "2026-09-01T00:00:00",
                }
            ],
        },
    }
    return json.dumps(package, ensure_ascii=False).encode()


def resource_bytes() -> bytes:
    rows = []
    stems = (
        "державні дані інформація набір реєстр українські оновлення розпорядник "
        "публічні нормативні послуги призначення категорія"
    )
    for index in range(12):
        unique = " ".join(f"унікальні{index}слова{item}" for item in range(95))
        rows.append(
            {
                "title": f"Реєстр державних наборів {index}",
                "description": f"{stems}. {unique}.",
                "category": f"Категорія української інформації {index}",
                "contact_email": f"person{index}@example.org",
                "phone": "+380 67 123 45 67",
            }
        )
    return json.dumps({"items": rows}, ensure_ascii=False).encode()


def load() -> dict:
    return mod.load_config()


def test_checked_in_config_is_zero_credit_and_binds_incumbent() -> None:
    cfg = load()
    assert cfg["incumbent_authority"]["head_sha"] == mod.INCUMBENT_HEAD
    assert cfg["claim_boundary"]["training_authorized_bytes"] == 0
    assert cfg["claim_boundary"]["training_eligible"] is False
    assert cfg["rights"]["evaluation"] == "NOT_ADMITTED"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("claim_boundary", "training_authorized_bytes"), 1),
        (("claim_boundary", "training_eligible"), True),
        (("claim_boundary", "candidate_only"), False),
        (("rights", "evaluation"), "ALLOWED"),
        (("quality", "min_accepted_records"), 1),
        (("resource_selection", "max_download_bytes"), 50_000_000),
    ],
)
def test_policy_mutations_fail_closed(
    tmp_path: Path, path: tuple[str, str], value: object
) -> None:
    cfg = load()
    cfg[path[0]][path[1]] = value
    config_path = tmp_path / "mutated.json"
    config_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(mod.CandidateError):
        mod.load_config(config_path)


def test_safe_text_excludes_contact_and_pii_scalars() -> None:
    cfg = load()
    row = {
        "title": "Український державний реєстр даних",
        "description": "Адміністративна інформація про державні набори.",
        "contact_email": "secret@example.org",
        "address": "м. Київ, вул. Прикладна, 1",
        "phone": "+380 67 123 45 67",
        "other": "123456789012345",
    }
    text, stats = mod.safe_record_text(row, cfg)
    assert "secret@example.org" not in text
    assert "+380" not in text
    assert "Прикладна" not in text
    assert stats["excluded_key"] >= 3


def test_unsafe_resource_host_fails_closed() -> None:
    cfg = load()
    with pytest.raises(mod.CandidateError, match="resource host drift"):
        mod.materialize_from_bytes(
            package_bytes("https://evil.example/resource.json"),
            resource_bytes(),
            cfg,
        )


def test_probe_materialization_is_deterministic_and_zero_credit() -> None:
    cfg = load()
    candidate_a, report_a = mod.materialize_from_bytes(package_bytes(), resource_bytes(), cfg)
    candidate_b, report_b = mod.materialize_from_bytes(package_bytes(), resource_bytes(), cfg)
    assert candidate_a == candidate_b
    assert report_a == report_b
    assert report_a["source_lock_state"] == "PROBE_LOCK_REQUIRED"
    assert report_a["training_authorized"] is False
    assert report_a["claim_boundary"]["training_authorized_bytes"] == 0
    rows = [json.loads(line) for line in candidate_a.splitlines()]
    assert len(rows) == 12
    assert all(row["training_eligible"] is False for row in rows)
    assert all(row["evaluation_eligible"] is False for row in rows)
    assert all("contact_email" not in row["text"] for row in rows)


def test_locked_source_identity_still_cannot_authorize_training() -> None:
    cfg = load()
    _, probe = mod.materialize_from_bytes(package_bytes(), resource_bytes(), cfg)
    locked = copy.deepcopy(cfg)
    locked["resource_selection"]["lock"] = probe["resource_identity"]
    candidate, report = mod.materialize_from_bytes(package_bytes(), resource_bytes(), locked)
    assert report["source_lock_state"] == "LOCKED_ZERO_CREDIT"
    assert report["global_dedup_completed"] is False
    assert report["reserved_evaluation_decontamination_completed"] is False
    assert report["training_authorized"] is False
    rows = [json.loads(line) for line in candidate.splitlines()]
    assert all(row["training_eligible"] is False for row in rows)


def test_partial_source_lock_is_rejected() -> None:
    cfg = load()
    cfg["resource_selection"]["lock"]["resource_id"] = "resource-1"
    with pytest.raises(mod.CandidateError, match="partial source lock"):
        mod.materialize_from_bytes(package_bytes(), resource_bytes(), cfg)


def test_lock_substitution_is_rejected() -> None:
    cfg = load()
    _, probe = mod.materialize_from_bytes(package_bytes(), resource_bytes(), cfg)
    locked = copy.deepcopy(cfg)
    locked["resource_selection"]["lock"] = probe["resource_identity"]
    locked["resource_selection"]["lock"]["raw_sha256"] = "0" * 64
    with pytest.raises(mod.CandidateError, match="locked source identity mismatch"):
        mod.materialize_from_bytes(package_bytes(), resource_bytes(), locked)


def test_evaluation_and_training_markers_never_come_from_source_record() -> None:
    cfg = load()
    raw = json.loads(resource_bytes())
    raw["items"][0]["training_eligible"] = True
    raw["items"][0]["evaluation_eligible"] = True
    candidate, report = mod.materialize_from_bytes(
        package_bytes(),
        json.dumps(raw, ensure_ascii=False).encode(),
        cfg,
    )
    rows = [json.loads(line) for line in candidate.splitlines()]
    assert all(row["training_eligible"] is False for row in rows)
    assert all(row["evaluation_eligible"] is False for row in rows)
    assert report["evaluation_authority"] == "NOT_ADMITTED"


def test_missing_downstream_gate_fails_config_validation(tmp_path: Path) -> None:
    cfg = load()
    cfg["required_downstream_gates"].pop()
    path = tmp_path / "weakened.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(mod.CandidateError, match="downstream gates"):
        mod.load_config(path)
