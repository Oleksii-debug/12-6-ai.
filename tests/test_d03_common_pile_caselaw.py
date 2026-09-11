from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "tools" / "materialize_d03_common_pile_caselaw.py"
spec = importlib.util.spec_from_file_location("caselaw_candidate", MODULE)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def load() -> dict:
    return mod.load_config()


def make_record(record_id: str, text: str | None = None, source: str | None = None) -> dict:
    if text is None:
        text = (
            "This court opinion explains the procedural history, governing legal standard, "
            "record evidence, judicial reasoning, and final disposition in a public case. "
            "The synthetic fixture contains ordinary English prose to exercise deterministic "
            "normalization and candidate selection without private information. "
            f"Record {record_id}."
        )
    return {
        "id": record_id,
        "source": source or "Caselaw Access Project",
        "added": "2024-08-24T03:29:51.129235",
        "created": "2024-08-24T03:29:51.129683",
        "metadata": {
            "author": "PER CURIAM",
            "license": "Public Domain",
            "url": "https://static.case.law/",
        },
        "text": text,
    }


def write_gzip(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True).encode() + b"\n")


def source_paths(tmp_path: Path, first: list[dict], second: list[dict]) -> list[Path]:
    paths = [
        tmp_path / "cap_00044.jsonl.gz",
        tmp_path / "cap_00043.jsonl.gz",
    ]
    write_gzip(paths[0], first)
    write_gzip(paths[1], second)
    return paths


def fake_identities(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {
        item["file"]: (item["bytes"], item["sha256"]) for item in mod.SOURCE_OBJECTS
    }
    monkeypatch.setattr(mod, "_source_file_identity", lambda path: expected[path.name])


def relax_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "MIN_CANDIDATE_BYTES", 1)
    monkeypatch.setattr(mod, "MAX_CANDIDATE_BYTES", 100_000)
    monkeypatch.setattr(mod, "MAX_SCANNED_BYTES", 100_000)


def test_checked_in_config_binds_exact_ordered_source_vector_and_zero_credit() -> None:
    cfg = load()
    assert cfg["source"]["ordering_policy"] == mod.SOURCE_ORDERING_POLICY
    assert [row["file"] for row in cfg["source"]["objects"]] == [
        "cap_00044.jsonl.gz",
        "cap_00043.jsonl.gz",
    ]
    assert cfg["source"]["objects"][0]["bytes"] == 9_441_419
    assert cfg["source"]["objects"][1]["bytes"] == 10_476_512
    assert cfg["source"]["objects"][0]["sha256"] == mod.SOURCE_OBJECTS[0]["sha256"]
    assert cfg["source"]["objects"][1]["sha256"] == mod.SOURCE_OBJECTS[1]["sha256"]
    assert cfg["claim_boundary"]["training_authorized_bytes"] == 0
    assert cfg["claim_boundary"]["source_level_project_review_complete"] is False
    assert cfg["claim_boundary"]["training_eligible"] is False


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("claim_boundary", "training_authorized_bytes", 1),
        ("claim_boundary", "training_eligible", True),
        ("claim_boundary", "candidate_only", False),
        ("claim_boundary", "source_level_project_review_complete", True),
        ("privacy", "reject_email", False),
        ("selection", "max_candidate_normalized_utf8_bytes", 10_000_000),
        ("record_contract", "metadata_license_exact", "anything"),
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


def test_source_order_mutation_fails_closed(tmp_path: Path) -> None:
    cfg = load()
    cfg["source"]["objects"].reverse()
    path = tmp_path / "reordered.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(mod.CandidateError, match="source identity/order drift"):
        mod.load_config(path)


def test_rights_registry_cannot_self_authorize_training(tmp_path: Path) -> None:
    cfg = load()
    registry_path = Path(cfg["rights_authority"]["registry_path"])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for source in registry["sources"]:
        if source["key"] == "caselaw_access_project":
            source["canonical_training_authorized"] = True
    mutated = tmp_path / "rights.json"
    mutated.write_text(json.dumps(registry), encoding="utf-8")
    cfg["rights_authority"]["registry_path"] = str(mutated)
    with pytest.raises(mod.CandidateError, match="self-authorized"):
        mod._validate_rights_registry(cfg)


def test_valid_record_is_candidate_only() -> None:
    ok, reason, candidate = mod.assess_record(make_record("case/0001.html"), load())
    assert ok is True
    assert reason == "accepted"
    assert candidate is not None
    assert candidate["training_eligible"] is False
    assert candidate["evaluation_eligible"] is False
    assert candidate["rights_basis"].endswith("REVIEW_REQUIRED")


@pytest.mark.parametrize("source", ["Caselaw Access Project", "Court Listener", "CourtListener"])
def test_known_source_labels_are_accepted(source: str) -> None:
    ok, reason, _ = mod.assess_record(make_record(f"{source}/1", source=source), load())
    assert (ok, reason) == (True, "accepted")


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda row: row.update(source="unknown"), "source field drift"),
        (lambda row: row["metadata"].update(license="CC-BY"), "public-domain metadata"),
        (lambda row: row.update(extra="drift"), "record field drift"),
    ],
)
def test_source_license_and_schema_drift_fail_closed(mutator, match: str) -> None:
    row = make_record("case/0002.html")
    mutator(row)
    with pytest.raises(mod.CandidateError, match=match):
        mod.assess_record(row, load())


@pytest.mark.parametrize(
    ("suffix", "reason"),
    [
        (" Contact researcher@example.org for details.", "email"),
        (" Phone +1 212 555 1234 for details.", "phone"),
        (" SSN 123-45-6789 appears here.", "us_ssn_shape"),
        ("\x00 hidden control", "control_character"),
    ],
)
def test_privacy_like_records_are_rejected(suffix: str, reason: str) -> None:
    row = make_record("case/0003.html")
    row["text"] += suffix
    ok, observed, candidate = mod.assess_record(row, load())
    assert ok is False
    assert observed == reason
    assert candidate is None


def test_non_english_record_is_rejected() -> None:
    text = (
        "Український судовий документ описує процесуальну історію, докази, правову норму, "
        "аргументацію сторін, мотиви рішення та остаточний висновок суду. Цей синтетичний "
        "приклад навмисно не є англійським і не повинен потрапити до англійської сім'ї."
    )
    ok, reason, _ = mod.assess_record(make_record("case/0004.html", text), load())
    assert ok is False
    assert reason in {"low_latin_ratio", "high_cyrillic_ratio"}


def test_materialization_is_deterministic_across_ordered_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load()
    paths = source_paths(
        tmp_path,
        [make_record(f"case/a-{index}") for index in range(5)],
        [make_record(f"case/b-{index}") for index in range(5)],
    )
    fake_identities(monkeypatch)
    relax_capacity(monkeypatch)
    candidate_a, candidate_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    report_a, report_b = tmp_path / "a.json", tmp_path / "b.json"
    first = mod.materialize(paths, candidate_a, report_a, cfg)
    second = mod.materialize(paths, candidate_b, report_b, cfg)
    assert first == second
    assert candidate_a.read_bytes() == candidate_b.read_bytes()
    assert report_a.read_bytes() == report_b.read_bytes()
    assert first["training_authorized_bytes"] == 0
    assert first["source_level_project_review_complete"] is False
    assert first["global_dedup_completed"] is False
    assert first["reserved_evaluation_decontamination_completed"] is False
    assert first["source_objects"][0]["file"] == "cap_00044.jsonl.gz"
    assert first["source_objects"][1]["file"] == "cap_00043.jsonl.gz"
    rows = [json.loads(line) for line in candidate_a.read_bytes().splitlines()]
    assert rows and all(row["training_eligible"] is False for row in rows)
    assert all("metadata" not in row for row in rows)


def test_reversed_runtime_source_order_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = source_paths(tmp_path, [make_record("case/a")], [make_record("case/b")])
    fake_identities(monkeypatch)
    with pytest.raises(mod.CandidateError, match="source order/file drift"):
        mod.materialize(
            list(reversed(paths)),
            tmp_path / "candidate.jsonl",
            tmp_path / "report.json",
            load(),
        )


def test_duplicate_source_id_across_shards_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = source_paths(
        tmp_path,
        [make_record("case/dup")],
        [make_record("case/dup")],
    )
    cfg = load()
    fake_identities(monkeypatch)
    relax_capacity(monkeypatch)
    with pytest.raises(mod.CandidateError, match="duplicate source record id"):
        mod.materialize(paths, tmp_path / "candidate.jsonl", tmp_path / "report.json", cfg)


def test_exact_normalized_duplicate_across_shards_is_discounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = make_record("case/a")
    second = make_record("case/b", first["text"])
    third = make_record("case/c")
    paths = source_paths(tmp_path, [first], [second, third])
    cfg = load()
    fake_identities(monkeypatch)
    relax_capacity(monkeypatch)
    report = mod.materialize(
        paths, tmp_path / "candidate.jsonl", tmp_path / "report.json", cfg
    )
    assert report["disposition_counts"]["exact_normalized_duplicate"] == 1
    assert report["retained_records"] == 2


def test_wrong_source_identity_fails_before_decode(tmp_path: Path) -> None:
    cfg = load()
    paths = [
        tmp_path / "cap_00044.jsonl.gz",
        tmp_path / "cap_00043.jsonl.gz",
    ]
    for path in paths:
        path.write_bytes(b"not the source")
    with pytest.raises(mod.CandidateError, match="source byte count mismatch"):
        mod.materialize(paths, tmp_path / "candidate.jsonl", tmp_path / "report.json", cfg)


def test_oversize_json_line_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = source_paths(
        tmp_path,
        [make_record("case/huge", "A" * 5000)],
        [make_record("case/other")],
    )
    cfg = load()
    fake_identities(monkeypatch)
    relax_capacity(monkeypatch)
    monkeypatch.setattr(mod, "MAX_LINE_BYTES", 1000)
    with pytest.raises(mod.CandidateError, match="safety envelope"):
        mod.materialize(paths, tmp_path / "candidate.jsonl", tmp_path / "report.json", cfg)


def test_scan_budget_stops_before_crossing_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = source_paths(
        tmp_path,
        [make_record("case/a")],
        [make_record("case/b"), make_record("case/c")],
    )
    cfg = load()
    fake_identities(monkeypatch)
    monkeypatch.setattr(mod, "MIN_CANDIDATE_BYTES", 1)
    monkeypatch.setattr(mod, "MAX_CANDIDATE_BYTES", 100_000)
    with gzip.open(paths[0], "rb") as first_handle:
        first_size = len(first_handle.read())
    with gzip.open(paths[1], "rb") as second_handle:
        second_first_line = second_handle.readline()
    monkeypatch.setattr(mod, "MAX_SCANNED_BYTES", first_size + len(second_first_line) - 1)
    report = mod.materialize(
        paths, tmp_path / "candidate.jsonl", tmp_path / "report.json", cfg
    )
    assert report["scan_budget_reached"] is True
    assert report["decompressed_bytes_scanned"] == first_size
    assert report["next_scan_bytes_if_budget_exceeded"] > mod.MAX_SCANNED_BYTES
    assert report["per_source"]["cap_00043.jsonl.gz"]["rows"] == 0


def test_missing_downstream_gate_fails_closed(tmp_path: Path) -> None:
    cfg = load()
    cfg["required_downstream_gates"].pop()
    path = tmp_path / "weakened.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(mod.CandidateError, match="downstream gates"):
        mod.load_config(path)
