from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "scan_d03_rada_trees_secondary_plaintext.py"
SPEC = importlib.util.spec_from_file_location("rada_secondary_plaintext", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def production_config() -> dict:
    return json.loads(MOD.CONFIG.read_text(encoding="utf-8"))


def synthetic_listing() -> list[dict]:
    return [
        {"path": "texts/a.txt", "size_bytes": 12},
        {"path": "texts/a.xtag", "size_bytes": 120},
        {"path": "texts/b.txt", "size_bytes": 13},
        {"path": "texts/b.xtag", "size_bytes": 130},
    ]


def bind_small_listing(monkeypatch: pytest.MonkeyPatch, cfg: dict, listing: list[dict]) -> None:
    monkeypatch.setattr(MOD, "MEMBER_COUNT", 4)
    monkeypatch.setattr(MOD, "UNCOMPRESSED_BYTES", 275)
    monkeypatch.setattr(MOD, "TXT_COUNT", 2)
    monkeypatch.setattr(MOD, "TXT_BYTES", 25)
    monkeypatch.setattr(MOD, "XTAG_COUNT", 2)
    monkeypatch.setattr(MOD, "XTAG_BYTES", 250)
    monkeypatch.setattr(MOD, "LISTING_SHA256", MOD.listing_identity(listing, cfg))


def test_production_config_is_exact_and_zero_credit() -> None:
    cfg = MOD.load_config()
    assert cfg["parent_authority"]["scientific_run_id"] == 34158830442
    assert cfg["source"]["content_sha256"] == MOD.SOURCE_SHA256
    assert cfg["terminal_listing"]["plain_text_member_count"] == 4391
    assert cfg["terminal_listing"]["plain_text_listing_bytes"] == 879031855
    assert cfg["selective_extract_policy"]["max_selected_uncompressed_bytes"] == 2_000_000_000
    assert cfg["claim_boundary"]["training_authorized_bytes"] == 0
    assert cfg["claim_boundary"]["tokenizer_fit_authorized"] is False


def test_exact_parent_listing_selects_only_plaintext(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = production_config()
    listing = synthetic_listing()
    bind_small_listing(monkeypatch, cfg, listing)

    selected = MOD.select_plaintext_listing(listing, cfg)

    assert selected == [
        {"path": "texts/a.txt", "size_bytes": 12},
        {"path": "texts/b.txt", "size_bytes": 13},
    ]


def test_listing_identity_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = production_config()
    listing = synthetic_listing()
    bind_small_listing(monkeypatch, cfg, listing)
    listing[0]["size_bytes"] += 1
    monkeypatch.setattr(MOD, "UNCOMPRESSED_BYTES", 276)
    monkeypatch.setattr(MOD, "TXT_BYTES", 26)

    with pytest.raises(MOD.PlaintextScanError, match="listing identity mismatch"):
        MOD.select_plaintext_listing(listing, cfg)


def test_unexpected_suffix_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = production_config()
    listing = synthetic_listing()
    listing[-1] = {"path": "texts/b.json", "size_bytes": 130}
    bind_small_listing(monkeypatch, cfg, listing)

    with pytest.raises(MOD.PlaintextScanError, match="unexpected suffix"):
        MOD.select_plaintext_listing(listing, cfg)


def test_plaintext_path_must_stay_under_texts_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = production_config()
    listing = synthetic_listing()
    listing[0] = {"path": "other/a.txt", "size_bytes": 12}
    bind_small_listing(monkeypatch, cfg, listing)

    with pytest.raises(MOD.PlaintextScanError, match="path prefix mismatch"):
        MOD.select_plaintext_listing(listing, cfg)


def test_selected_extraction_envelope_stays_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = production_config()
    listing = synthetic_listing()
    bind_small_listing(monkeypatch, cfg, listing)
    cfg["selective_extract_policy"]["max_selected_uncompressed_bytes"] = 24

    with pytest.raises(MOD.PlaintextScanError, match="extraction envelope"):
        MOD.select_plaintext_listing(listing, cfg)


def test_extracted_tree_is_hashed_and_classified_without_text_emission(tmp_path: Path) -> None:
    cfg = production_config()
    root = tmp_path / "selected"
    texts = root / "texts"
    texts.mkdir(parents=True)
    payload_a = (
        "Український парламент ухвалив закон і розглянув державне питання.\n" * 8
    ).encode("utf-8")
    payload_b = (
        "Верховна Рада України провела пленарне засідання та обговорення.\n" * 8
    ).encode("utf-8")
    (texts / "a.txt").write_bytes(payload_a)
    (texts / "b.txt").write_bytes(payload_b)
    selected = [
        {"path": "texts/a.txt", "size_bytes": len(payload_a)},
        {"path": "texts/b.txt", "size_bytes": len(payload_b)},
    ]

    classified = MOD.classify_selected_tree(root, selected, cfg)

    assert len(classified) == 2
    assert {item["classification"] for item in classified} == {"PLAIN_TEXT_CANDIDATE"}
    assert all(len(item["sha256"]) == 64 for item in classified)
    assert all(item["text_emitted"] is False for item in classified)
    assert all("text" not in item for item in classified)


def test_claim_boundary_cannot_be_promoted(tmp_path: Path) -> None:
    cfg = production_config()
    cfg["claim_boundary"]["training_authorized_bytes"] = 1
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")

    with pytest.raises(MOD.PlaintextScanError, match="zero boundary drift"):
        MOD.load_config(path)
