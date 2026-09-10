from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/materialize_d03_rada_trees_quality_windows.py"
SPEC = importlib.util.spec_from_file_location("d03_rada_quality_windows", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakePrivacyResult:
    def __init__(
        self,
        text: str,
        action: str = "ALLOW",
        detectors: dict[str, int] | None = None,
    ) -> None:
        raw = text.encode("utf-8")
        self._evidence = {
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "input_bytes": len(raw),
            "action": action,
            "detector_counts": detectors or {},
        }

    def evidence(self) -> dict[str, Any]:
        return dict(self._evidence)


def record(record_id: str = "r1", text: str = "абвгд") -> dict[str, Any]:
    raw = text.encode("utf-8")
    return {
        "record_id": record_id,
        "source_family": MODULE.EXPECTED_FAMILY,
        "source_dataset": MODULE.EXPECTED_DATASET,
        "source_revision": MODULE.EXPECTED_REVISION,
        "source_archive": MODULE.EXPECTED_ARCHIVE,
        "source_path": f"texts/{record_id}.txt",
        "session_date": "2020-01-01",
        "source_payload_sha256": "b" * 64,
        "source_payload_bytes": len(raw),
        "decoded_encoding": "utf-8",
        "decoded_text_utf8_sha256": sha(text),
        "decoded_text_utf8_bytes": len(raw),
        "rights_scope_status": MODULE.EXPECTED_RIGHTS_STATUS,
        "attribution_required": True,
        "language_quality_privacy_complete": False,
        "global_dedup_complete": False,
        "reserved_evaluation_decontamination_complete": False,
        "training_eligible": False,
        "evaluation_eligible": False,
        "text": text,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(MODULE.canonical_bytes(row) for row in rows))


def quality_partial(record_id: str, text: str, mode: str) -> dict[str, Any]:
    assert mode == "uk" and len(text) == 5
    spans = [(0, 0, 2, True), (1, 2, 3, False), (2, 3, 5, True)]
    windows: list[dict[str, Any]] = []
    retained = rejected = 0
    for index, start, end, accepted in spans:
        payload = text[start:end]
        nbytes = len(payload.encode("utf-8"))
        retained += nbytes if accepted else 0
        rejected += 0 if accepted else nbytes
        windows.append(
            {
                "index": index,
                "start_char": start,
                "end_char": end,
                "utf8_bytes": nbytes,
                "authoritative": True,
                "decision": {"accepted": accepted},
            }
        )
    return {
        "record_id": record_id,
        "mode": mode,
        "authoritative_unit": "BOUNDED_NATURAL_LANGUAGE_WINDOW",
        "family_eviction_authority": False,
        "status": "RETAIN_PARTIAL",
        "retained_utf8_bytes": retained,
        "rejected_utf8_bytes": rejected,
        "windows": windows,
    }


def quality_policy() -> dict[str, Any]:
    return {"policy_sha256": "e" * 64}


def privacy_policy() -> dict[str, Any]:
    return {"policy_sha256": "d" * 64}


def privacy_allow(text: str) -> FakePrivacyResult:
    return FakePrivacyResult(text)


def mechanics(quality=quality_partial, privacy=privacy_allow):
    return quality, quality_policy, privacy, privacy_policy


def run(candidate: Path, output: Path, report: Path) -> dict[str, Any]:
    return MODULE.materialize(
        candidate,
        output,
        report,
        expected_candidate_sha256=file_sha(candidate),
    )


def test_partial_materializes_accepted_windows_then_scans_exact_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, output, report = (
        tmp_path / "candidate.jsonl",
        tmp_path / "out.jsonl",
        tmp_path / "report.json",
    )
    write_jsonl(candidate, [record()])
    scanned: list[str] = []

    def privacy(text: str) -> FakePrivacyResult:
        scanned.append(text)
        return FakePrivacyResult(text)

    monkeypatch.setattr(MODULE, "_load_mechanics", lambda: mechanics(privacy=privacy))
    result = run(candidate, output, report)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert scanned == ["аб", "гд"]
    assert [row["text"] for row in rows] == ["аб", "гд"]
    assert [row["quality_window_index"] for row in rows] == [0, 2]
    assert [row["decoded_text_utf8_sha256"] for row in rows] == [sha("аб"), sha("гд")]
    assert all(row["quality_parent_decoded_text_utf8_sha256"] == sha("абвгд") for row in rows)
    assert all(row["quality_granularity_policy_sha256"] == "e" * 64 for row in rows)
    assert all(row["privacy_policy_sha256"] == "d" * 64 for row in rows)
    assert all(row["training_eligible"] is False for row in rows)
    assert result["quality"]["retained_units_before_privacy"] == 2
    assert result["privacy"]["allowed_units"] == 2
    assert result["claim_boundary"]["training_authorized_bytes"] == 0
    assert result["claim_boundary"]["unique_causal_loss_positions_authorized"] == 0


def test_privacy_non_allow_unit_is_held_without_payload_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.jsonl"
    output, report = tmp_path / "out.jsonl", tmp_path / "report.json"
    write_jsonl(candidate, [record()])

    def privacy(text: str) -> FakePrivacyResult:
        if text == "гд":
            return FakePrivacyResult(text, "REDACT", {"email": 1})
        return FakePrivacyResult(text)

    monkeypatch.setattr(MODULE, "_load_mechanics", lambda: mechanics(privacy=privacy))
    result = run(candidate, output, report)
    assert [json.loads(line)["text"] for line in output.read_text().splitlines()] == ["аб"]
    assert result["privacy"]["actions"] == {"ALLOW": 1, "REDACT": 1}
    assert result["privacy"]["held_units"] == 1
    assert result["privacy"]["detector_counts"] == {"email": 1}
    assert (
        result["privacy"]["allowed_utf8_bytes"]
        + result["privacy"]["held_utf8_bytes"]
        == result["quality"]["retained_utf8_bytes"]
    )
    assert "гд" not in json.dumps(json.loads(report.read_text()), ensure_ascii=False)


def test_retain_all_is_single_byte_identical_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "Парламентський текст"
    candidate = tmp_path / "candidate.jsonl"
    output, report = tmp_path / "out.jsonl", tmp_path / "report.json"
    write_jsonl(candidate, [record(text=text)])

    def quality(record_id: str, payload: str, mode: str) -> dict[str, Any]:
        return {
            "record_id": record_id,
            "mode": mode,
            "authoritative_unit": "DOCUMENT",
            "family_eviction_authority": False,
            "status": "RETAIN_ALL",
            "retained_utf8_bytes": len(payload.encode()),
            "rejected_utf8_bytes": 0,
            "windows": [],
        }

    monkeypatch.setattr(MODULE, "_load_mechanics", lambda: mechanics(quality=quality))
    run(candidate, output, report)
    row = json.loads(output.read_text())
    assert row["record_id"] == "r1" and row["text"] == text
    assert row["quality_window_index"] is None
    assert row["privacy_scan_input_sha256"] == sha(text)


def test_reject_document_never_reaches_privacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.jsonl"
    output, report = tmp_path / "out.jsonl", tmp_path / "report.json"
    write_jsonl(candidate, [record()])

    def quality(record_id: str, payload: str, mode: str) -> dict[str, Any]:
        return {
            "record_id": record_id,
            "mode": mode,
            "authoritative_unit": "DOCUMENT",
            "family_eviction_authority": False,
            "status": "REJECT_DOCUMENT",
            "retained_utf8_bytes": 0,
            "rejected_utf8_bytes": len(payload.encode()),
            "windows": [],
        }

    def privacy(_: str) -> FakePrivacyResult:
        raise AssertionError("privacy must not scan quality-rejected payload")

    monkeypatch.setattr(MODULE, "_load_mechanics", lambda: mechanics(quality, privacy))
    result = run(candidate, output, report)
    assert output.read_bytes() == b""
    assert result["quality"]["statuses"] == {"REJECT_DOCUMENT": 1}


def test_quality_partition_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(candidate, [record()])

    def bad_quality(record_id: str, text: str, mode: str) -> dict[str, Any]:
        value = quality_partial(record_id, text, mode)
        value["windows"][1]["start_char"] = 1
        return value

    monkeypatch.setattr(MODULE, "_load_mechanics", lambda: mechanics(quality=bad_quality))
    with pytest.raises(MODULE.MaterializationError, match="window partition drift"):
        run(candidate, tmp_path / "out.jsonl", tmp_path / "report.json")


def test_quality_authority_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(candidate, [record()])

    def bad_quality(record_id: str, text: str, mode: str) -> dict[str, Any]:
        value = quality_partial(record_id, text, mode)
        value["family_eviction_authority"] = True
        return value

    monkeypatch.setattr(MODULE, "_load_mechanics", lambda: mechanics(quality=bad_quality))
    with pytest.raises(MODULE.MaterializationError, match="family-eviction authority drift"):
        run(candidate, tmp_path / "out.jsonl", tmp_path / "report.json")


def test_privacy_evidence_must_bind_exact_emitted_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(candidate, [record()])

    class WrongPrivacy:
        def evidence(self) -> dict[str, Any]:
            return {
                "input_sha256": "0" * 64,
                "input_bytes": 1,
                "action": "ALLOW",
                "detector_counts": {},
            }

    monkeypatch.setattr(
        MODULE,
        "_load_mechanics",
        lambda: mechanics(privacy=lambda _: WrongPrivacy()),
    )
    with pytest.raises(MODULE.MaterializationError, match="privacy input SHA binding drift"):
        run(candidate, tmp_path / "out.jsonl", tmp_path / "report.json")


def test_nested_privacy_payload_evidence_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(candidate, [record()])

    class UnsafePrivacy(FakePrivacyResult):
        def evidence(self) -> dict[str, Any]:
            value = super().evidence()
            value["nested"] = {"matched_value": "secret"}
            return value

    monkeypatch.setattr(MODULE, "_load_mechanics", lambda: mechanics(privacy=UnsafePrivacy))
    with pytest.raises(MODULE.MaterializationError, match="forbidden payload field"):
        run(candidate, tmp_path / "out.jsonl", tmp_path / "report.json")


def test_boolean_privacy_detector_count_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(candidate, [record()])
    privacy = lambda text: FakePrivacyResult(text, "ALLOW", {"email": True})
    monkeypatch.setattr(MODULE, "_load_mechanics", lambda: mechanics(privacy=privacy))
    with pytest.raises(MODULE.MaterializationError, match="detector count malformed"):
        run(candidate, tmp_path / "out.jsonl", tmp_path / "report.json")


def test_input_text_must_match_upstream_decoded_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.jsonl"
    row = record()
    row["decoded_text_utf8_sha256"] = "0" * 64
    write_jsonl(candidate, [row])
    monkeypatch.setattr(MODULE, "_load_mechanics", lambda: mechanics())
    with pytest.raises(MODULE.MaterializationError, match="decoded text SHA drift"):
        run(candidate, tmp_path / "out.jsonl", tmp_path / "report.json")


def test_duplicate_record_id_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(candidate, [record(), record()])
    monkeypatch.setattr(MODULE, "_load_mechanics", lambda: mechanics())
    with pytest.raises(MODULE.MaterializationError, match="duplicate record_id"):
        run(candidate, tmp_path / "out.jsonl", tmp_path / "report.json")


def test_wrong_expected_candidate_sha_fails_before_mechanics_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(candidate, [record()])

    def must_not_load() -> tuple[Any, ...]:
        raise AssertionError("mechanics must not load before input authority matches")

    monkeypatch.setattr(MODULE, "_load_mechanics", must_not_load)
    with pytest.raises(MODULE.MaterializationError, match="does not match expected"):
        MODULE.materialize(
            candidate,
            tmp_path / "out.jsonl",
            tmp_path / "report.json",
            expected_candidate_sha256="0" * 64,
        )


def test_output_alias_and_symlink_are_rejected_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.jsonl"
    report = tmp_path / "report.json"
    write_jsonl(candidate, [record()])
    before = candidate.read_bytes()
    monkeypatch.setattr(MODULE, "_load_mechanics", lambda: mechanics())
    with pytest.raises(MODULE.MaterializationError, match="paths must be distinct"):
        MODULE.materialize(
            candidate,
            candidate,
            report,
            expected_candidate_sha256=file_sha(candidate),
        )
    assert candidate.read_bytes() == before

    target = tmp_path / "target.jsonl"
    target.write_text("sentinel")
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    with pytest.raises(MODULE.MaterializationError, match="must not be a symlink"):
        MODULE.materialize(candidate, link, report, expected_candidate_sha256=file_sha(candidate))
    assert target.read_text() == "sentinel"


def test_existing_output_is_never_silently_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.jsonl"
    output, report = tmp_path / "out.jsonl", tmp_path / "report.json"
    write_jsonl(candidate, [record()])
    output.write_text("sentinel")
    monkeypatch.setattr(MODULE, "_load_mechanics", lambda: mechanics())
    with pytest.raises(MODULE.MaterializationError, match="already exists"):
        MODULE.materialize(candidate, output, report, expected_candidate_sha256=file_sha(candidate))
    assert output.read_text() == "sentinel"


def test_report_and_output_are_deterministic_and_self_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate.jsonl"
    write_jsonl(candidate, [record()])
    monkeypatch.setattr(MODULE, "_load_mechanics", lambda: mechanics())
    out1, rep1 = tmp_path / "out1.jsonl", tmp_path / "rep1.json"
    out2, rep2 = tmp_path / "out2.jsonl", tmp_path / "rep2.json"
    first, second = run(candidate, out1, rep1), run(candidate, out2, rep2)
    assert out1.read_bytes() == out2.read_bytes()
    assert rep1.read_bytes() == rep2.read_bytes()
    assert first == second
    core = dict(first)
    claimed = core.pop("report_sha256")
    assert claimed == MODULE.sha256_bytes(MODULE.canonical_bytes(core))


def test_exact_branch_composes_incumbent_quality_and_privacy_apis() -> None:
    apply_quality, quality_policy_fn, scan_privacy, privacy_policy_fn = MODULE._load_mechanics()
    text = (
        "Верховна Рада України розглянула питання порядку денного та ухвалила рішення. "
        "Доповідач представив матеріали, після чого відбулося обговорення. "
    ) * 12
    quality = apply_quality("integration-probe", text, "uk")
    assert quality["retained_utf8_bytes"] + quality["rejected_utf8_bytes"] == len(text.encode())
    assert scan_privacy("Звичайний парламентський текст.").evidence()["action"] == "ALLOW"
    assert len(MODULE._policy_sha(quality_policy_fn, "quality granularity")) == 64
    assert len(MODULE._policy_sha(privacy_policy_fn, "privacy")) == 64
