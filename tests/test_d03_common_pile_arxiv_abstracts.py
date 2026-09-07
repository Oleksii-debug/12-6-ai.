from __future__ import annotations

import copy
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = (
    Path(__file__).parents[1] / "tools" / "materialize_d03_common_pile_arxiv_abstracts.py"
)
spec = importlib.util.spec_from_file_location("arxiv_candidate", MODULE)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def load() -> dict:
    return mod.load_config()


def make_record(record_id: str, text: str | None = None) -> dict:
    if text is None:
        text = (
            "This scientific abstract presents a reproducible analysis of language models, "
            "optimization methods, evaluation protocols, statistical evidence, and robust "
            "experimental results for a carefully controlled research setting. "
            f"The unique record identifier for this synthetic fixture is {record_id}."
        )
    return {
        "id": record_id,
        "text": text,
        "source": mod.SOURCE_FIELD,
        "created": "2024-01-01T00:00:00",
        "added": "2024-08-12T18:16:26",
        "metadata": {
            "license": (
                "Creative Commons Zero - Public Domain - "
                "https://creativecommons.org/publicdomain/zero/1.0/"
            ),
            "full_text_license": "None",
            "authors": "Synthetic Author",
        },
    }


def write_gzip(path: Path, records: list[dict]) -> None:
    with gzip.open(path, "wb") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True).encode() + b"\n")


def test_checked_in_config_binds_exact_source_and_zero_credit() -> None:
    cfg = load()
    assert cfg["source"]["sha256"] == mod.SOURCE_SHA256
    assert cfg["source"]["bytes"] == mod.SOURCE_BYTES
    assert cfg["claim_boundary"]["training_authorized_bytes"] == 0
    assert cfg["claim_boundary"]["training_eligible"] is False
    assert cfg["claim_boundary"]["evaluation_eligible"] is False


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("claim_boundary", "training_authorized_bytes", 1),
        ("claim_boundary", "training_eligible", True),
        ("claim_boundary", "candidate_only", False),
        ("privacy", "reject_email", False),
        ("selection", "max_candidate_normalized_utf8_bytes", 10_000_000),
        ("record_contract", "metadata_license_prefix", "anything"),
    ],
)
def test_policy_mutation_fails_closed(
    tmp_path: Path, section: str, key: str, value: object
) -> None:
    cfg = load()
    cfg[section][key] = value
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(mod.CandidateError):
        mod.load_config(path)


def test_rights_registry_cannot_self_authorize_training(tmp_path: Path) -> None:
    cfg = load()
    registry_path = Path(cfg["rights_authority"]["registry_path"])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for source in registry["sources"]:
        if source["key"] == "arxiv_abstracts":
            source["canonical_training_authorized"] = True
    mutated = tmp_path / "rights.json"
    mutated.write_text(json.dumps(registry), encoding="utf-8")
    cfg["rights_authority"]["registry_path"] = str(mutated)
    with pytest.raises(mod.CandidateError, match="self-authorized"):
        mod._validate_rights_registry(cfg)


def test_valid_record_emits_only_zero_credit_candidate() -> None:
    cfg = load()
    ok, reason, candidate = mod.assess_record(make_record("0704.0001"), cfg)
    assert ok is True
    assert reason == "accepted"
    assert candidate is not None
    assert candidate["training_eligible"] is False
    assert candidate["evaluation_eligible"] is False
    assert candidate["rights_basis"] == "CC0_ARXIV_METADATA_ABSTRACT"


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda row: row.update(source="other"), "source field drift"),
        (
            lambda row: row["metadata"].update(license="arXiv non-exclusive"),
            "CC0 metadata license missing",
        ),
        (lambda row: row.update(extra="drift"), "record field drift"),
    ],
)
def test_source_and_license_drift_fail_closed(mutator, match: str) -> None:
    cfg = load()
    row = make_record("0704.0002")
    mutator(row)
    with pytest.raises(mod.CandidateError, match=match):
        mod.assess_record(row, cfg)


@pytest.mark.parametrize(
    ("suffix", "reason"),
    [
        (" Contact researcher@example.org for details.", "email"),
        (" Phone +1 212 555 1234 for details.", "phone"),
        ("\x00 hidden control", "control_character"),
    ],
)
def test_privacy_like_abstracts_are_rejected(suffix: str, reason: str) -> None:
    cfg = load()
    row = make_record("0704.0003")
    row["text"] += suffix
    ok, observed, candidate = mod.assess_record(row, cfg)
    assert ok is False
    assert observed == reason
    assert candidate is None


def test_non_english_record_is_rejected() -> None:
    cfg = load()
    text = (
        "Українське наукове дослідження описує результати, методологію, експерименти, "
        "оцінювання, статистичні показники та відтворюваний аналіз для мовної моделі. "
        "Цей текст навмисно є українським і не повинен потрапити до англійської сім'ї."
    )
    ok, reason, _ = mod.assess_record(make_record("0704.0004", text), cfg)
    assert ok is False
    assert reason in {"low_latin_ratio", "high_cyrillic_ratio"}


def test_materialization_is_deterministic_and_stays_zero_credit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load()
    records = [make_record(f"0704.{index:04d}") for index in range(1, 10)]
    source = tmp_path / "source.jsonl.gz"
    write_gzip(source, records)

    monkeypatch.setattr(
        mod,
        "_source_file_identity",
        lambda _: (mod.SOURCE_BYTES, mod.SOURCE_SHA256),
    )
    monkeypatch.setattr(mod, "MIN_CANDIDATE_BYTES", 500)
    monkeypatch.setattr(mod, "MAX_CANDIDATE_BYTES", 10_000)
    monkeypatch.setattr(mod, "MAX_SCANNED_BYTES", 50_000)

    candidate_a = tmp_path / "candidate-a.jsonl"
    report_a = tmp_path / "report-a.json"
    candidate_b = tmp_path / "candidate-b.jsonl"
    report_b = tmp_path / "report-b.json"
    first = mod.materialize(source, candidate_a, report_a, cfg)
    second = mod.materialize(source, candidate_b, report_b, cfg)
    assert first == second
    assert candidate_a.read_bytes() == candidate_b.read_bytes()
    assert report_a.read_bytes() == report_b.read_bytes()
    assert first["training_authorized_bytes"] == 0
    assert first["training_eligible"] is False
    assert first["global_dedup_completed"] is False
    assert first["reserved_evaluation_decontamination_completed"] is False
    rows = [json.loads(line) for line in candidate_a.read_bytes().splitlines()]
    assert rows
    assert all(row["training_eligible"] is False for row in rows)


def test_duplicate_source_id_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load()
    source = tmp_path / "source.jsonl.gz"
    write_gzip(source, [make_record("0704.0010"), make_record("0704.0010")])
    monkeypatch.setattr(
        mod,
        "_source_file_identity",
        lambda _: (mod.SOURCE_BYTES, mod.SOURCE_SHA256),
    )
    monkeypatch.setattr(mod, "MIN_CANDIDATE_BYTES", 1)
    monkeypatch.setattr(mod, "MAX_CANDIDATE_BYTES", 10_000)
    monkeypatch.setattr(mod, "MAX_SCANNED_BYTES", 50_000)
    with pytest.raises(mod.CandidateError, match="duplicate source record id"):
        mod.materialize(
            source,
            tmp_path / "candidate.jsonl",
            tmp_path / "report.json",
            cfg,
        )


def test_exact_normalized_duplicate_is_discounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load()
    first = make_record("0704.0011")
    second = make_record("0704.0012", first["text"])
    third = make_record("0704.0013")
    source = tmp_path / "source.jsonl.gz"
    write_gzip(source, [first, second, third])
    monkeypatch.setattr(
        mod,
        "_source_file_identity",
        lambda _: (mod.SOURCE_BYTES, mod.SOURCE_SHA256),
    )
    monkeypatch.setattr(mod, "MIN_CANDIDATE_BYTES", 1)
    monkeypatch.setattr(mod, "MAX_CANDIDATE_BYTES", 10_000)
    monkeypatch.setattr(mod, "MAX_SCANNED_BYTES", 50_000)
    report = mod.materialize(
        source,
        tmp_path / "candidate.jsonl",
        tmp_path / "report.json",
        cfg,
    )
    assert report["disposition_counts"]["exact_normalized_duplicate"] == 1
    assert report["retained_records"] == 2


def test_missing_downstream_gate_fails_closed(tmp_path: Path) -> None:
    cfg = load()
    cfg["required_downstream_gates"].pop()
    path = tmp_path / "weakened.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(mod.CandidateError, match="downstream gates"):
        mod.load_config(path)
