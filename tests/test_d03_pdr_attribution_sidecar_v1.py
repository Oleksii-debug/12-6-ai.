from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "pdr_attribution",
    TOOLS / "build_d03_pdr_attribution_sidecar_v1.py",
)
assert SPEC and SPEC.loader
pdr = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pdr
SPEC.loader.exec_module(pdr)


def _candidate(record_id: str = "essay-a", text: str = "A stable candidate text.") -> dict:
    encoded = text.encode("utf-8")
    return {
        "artifact_role": "SOURCE_CANDIDATE_ONLY",
        "record_id": record_id,
        "source_id": pdr.SOURCE_FAMILY,
        "source_dataset": pdr.SOURCE_DATASET,
        "source_revision": pdr.SOURCE_REVISION,
        "origin_url": f"https://publicdomainreview.org/essay/{record_id}/",
        "license": pdr.EXPECTED_LICENSE,
        "attribution_required": True,
        "language": "en",
        "normalized_sha256": hashlib.sha256(encoded).hexdigest(),
        "normalized_utf8_bytes": len(encoded),
        "text": text,
        "training_eligible": False,
        "evaluation_eligible": False,
    }


def _raw(
    record_id: str = "essay-a",
    *,
    author: object = "Ada Example",
    source_type: str = "essay",
) -> dict:
    return {
        "id": record_id,
        "text": "Raw source text with the original byline.",
        "source": pdr.SOURCE_VALUE,
        "date": "2024-01-01T00:00:00",
        "author": author,
        "type": source_type,
        "added": "2026-01-01T00:00:00",
        "metadata": {
            "license": pdr.EXPECTED_LICENSE,
            "url": f"https://publicdomainreview.org/essay/{record_id}/",
        },
    }


def _raw_vector(*rows: dict) -> dict[str, list[dict]]:
    return {"collection": [], "essay": list(rows)}


def _config() -> dict:
    return {
        "common_pile_rights_authority": {
            "source_key": pdr.SOURCE_KEY,
            "project_review_status": "REVIEW_REQUIRED",
        },
        "source": {
            "dataset": pdr.SOURCE_DATASET,
            "revision": pdr.SOURCE_REVISION,
            "family_id": pdr.SOURCE_FAMILY,
            "source_value": pdr.SOURCE_VALUE,
            "origin_host": pdr.SOURCE_HOST,
            "expected_record_license": pdr.EXPECTED_LICENSE,
            "attribution_required": True,
        },
        "files": [
            {"path": path, "sha256": expected["sha256"]}
            for path, expected in sorted(pdr.EXPECTED_FILES.items())
        ],
        "truth_boundary": {
            "canonical_corpus_admitted": False,
            "training_authorized_bytes": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "model_training_executed": False,
            "final_test_payload_accessed": False,
            "paid_compute_used": False,
        },
    }


def _candidate_payload(rows: list[dict]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for row in rows
    )


def test_author_is_preserved_only_in_sidecar_and_report_is_text_free() -> None:
    candidate = _candidate()
    sidecar, report = pdr.build_sidecar([candidate], _raw_vector(_raw()))

    assert len(sidecar) == 1
    assert sidecar[0]["attribution"]["author"] == "Ada Example"
    assert sidecar[0]["attribution"]["publisher"] == pdr.PUBLISHER
    assert sidecar[0]["attribution_metadata_in_training_text"] is False
    assert candidate["text"] == "A stable candidate text."
    assert report["training_text_modified"] is False
    assert report["attributable_record_count"] == 1
    assert report["excluded_record_count"] == 0
    assert report["attribution_coverage_complete"] is True
    durable = json.dumps(report, ensure_ascii=False, sort_keys=True)
    assert "Ada Example" not in durable
    assert report["training_authorized_bytes"] == 0
    assert report["authorized_optimized_target_exposure"] == 0


def test_sidecar_is_deterministic_and_preserves_candidate_order() -> None:
    candidates = [_candidate("essay-b", "Text B"), _candidate("essay-a", "Text A")]
    raw = _raw_vector(_raw("essay-a"), _raw("essay-b"))

    first = pdr.build_sidecar(copy.deepcopy(candidates), copy.deepcopy(raw))
    second = pdr.build_sidecar(copy.deepcopy(candidates), copy.deepcopy(raw))

    assert first == second
    assert [row["record_id"] for row in first[0]] == ["essay-b", "essay-a"]


def test_unselected_missing_author_does_not_block_retained_candidate() -> None:
    raw = _raw_vector(
        _raw("essay-unselected", author=None),
        _raw("essay-a", author="Ada Example"),
    )

    sidecar, report = pdr.build_sidecar([_candidate("essay-a")], raw)

    assert [row["record_id"] for row in sidecar] == ["essay-a"]
    assert sidecar[0]["attribution"]["author"] == "Ada Example"
    assert report["selected_record_count"] == 1
    assert report["attributable_record_count"] == 1
    assert report["excluded_record_count"] == 0
    assert report["training_authorized_bytes"] == 0


def test_retained_missing_author_is_excluded_fail_closed_without_id_leak() -> None:
    candidates = [_candidate("essay-missing", "Text missing"), _candidate("essay-a", "Text A")]
    raw = _raw_vector(
        _raw("essay-missing", author="   "),
        _raw("essay-a", author="Ada Example"),
    )

    sidecar, report = pdr.build_sidecar(candidates, raw)

    assert [row["record_id"] for row in sidecar] == ["essay-a"]
    assert report["status"] == "ATTRIBUTION_PARTIAL_FAIL_CLOSED_ZERO_CREDIT"
    assert report["selected_record_count"] == 2
    assert report["attributable_record_count"] == 1
    assert report["excluded_record_count"] == 1
    assert report["excluded_reason_counts"] == {"missing": 1}
    assert report["attribution_coverage_complete"] is False
    assert report["canonical_corpus_admitted"] is False
    assert report["training_authorized_bytes"] == 0
    assert report["authorized_optimized_target_exposure"] == 0
    durable = json.dumps(report, ensure_ascii=False, sort_keys=True)
    assert "essay-missing" not in durable
    assert "Ada Example" not in durable


def test_exact_candidate_reader_binds_bytes_hash_count_and_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_candidate("essay-a", "Text A"), _candidate("essay-b", "Text B")]
    payload = _candidate_payload(rows)
    candidate_path = tmp_path / "candidate.jsonl"
    candidate_path.write_bytes(payload)
    projection = pdr._candidate_projection(rows)

    monkeypatch.setattr(pdr, "EXPECTED_CANDIDATE_JSONL_BYTES", len(payload))
    monkeypatch.setattr(pdr, "EXPECTED_CANDIDATE_JSONL_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(pdr, "EXPECTED_SELECTED_RECORD_COUNT", len(rows))
    monkeypatch.setattr(
        pdr,
        "EXPECTED_SELECTED_NORMALIZED_UTF8_BYTES",
        sum(row["normalized_utf8_bytes"] for row in rows),
    )
    monkeypatch.setattr(
        pdr,
        "EXPECTED_CANDIDATE_PROJECTION_SHA256",
        hashlib.sha256(pdr._canonical_bytes(projection)).hexdigest(),
    )

    assert pdr._read_exact_candidate(candidate_path) == rows


def test_exact_candidate_reader_rejects_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_candidate()]
    payload = _candidate_payload(rows)
    candidate_path = tmp_path / "candidate.jsonl"
    candidate_path.write_bytes(payload)

    monkeypatch.setattr(pdr, "EXPECTED_CANDIDATE_JSONL_BYTES", len(payload))
    monkeypatch.setattr(pdr, "EXPECTED_CANDIDATE_JSONL_SHA256", "f" * 64)

    with pytest.raises(pdr.AttributionError, match="candidate SHA-256 mismatch"):
        pdr._read_exact_candidate(candidate_path)


@pytest.mark.parametrize("author", [None, "", "   ", 123, "x\x00y", "a" * 513])
def test_invalid_retained_author_never_enters_sidecar(author: object) -> None:
    sidecar, report = pdr.build_sidecar([_candidate()], _raw_vector(_raw(author=author)))

    assert sidecar == []
    assert report["selected_record_count"] == 1
    assert report["attributable_record_count"] == 0
    assert report["excluded_record_count"] == 1
    assert report["attribution_coverage_complete"] is False
    assert report["canonical_corpus_admitted"] is False
    assert report["training_authorized_bytes"] == 0
    assert report["authorized_optimized_target_exposure"] == 0


def test_raw_metadata_schema_drift_fails_closed() -> None:
    raw = _raw()
    raw["metadata"]["unexpected"] = "value"

    with pytest.raises(pdr.AttributionError, match="metadata schema drift"):
        pdr.build_sidecar([_candidate()], _raw_vector(raw))


def test_candidate_without_exact_raw_attribution_record_fails_closed() -> None:
    with pytest.raises(pdr.AttributionError, match="no exact raw attribution record"):
        pdr.build_sidecar([_candidate("essay-a")], _raw_vector(_raw("essay-b")))


def test_duplicate_raw_identity_fails_closed() -> None:
    duplicate = _raw()
    with pytest.raises(pdr.AttributionError, match="duplicate raw PDR record identity"):
        pdr.build_sidecar([_candidate()], _raw_vector(duplicate, copy.deepcopy(duplicate)))


def test_candidate_text_identity_tamper_fails_closed() -> None:
    candidate = _candidate()
    candidate["text"] = "Different text"

    with pytest.raises(pdr.AttributionError, match="candidate text SHA mismatch"):
        pdr.build_sidecar([candidate], _raw_vector(_raw()))


def test_candidate_training_promotion_fails_closed() -> None:
    candidate = _candidate()
    candidate["training_eligible"] = True

    with pytest.raises(pdr.AttributionError, match="candidate training gate drift"):
        pdr.build_sidecar([candidate], _raw_vector(_raw()))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("common_pile_rights_authority", "project_review_status"), "PASS", "REVIEW_REQUIRED"),
        (("truth_boundary", "training_authorized_bytes"), 1, "training credit drift"),
        (("truth_boundary", "training_authorized_bytes"), False, "training credit drift"),
        (("truth_boundary", "authorized_optimized_target_exposure"), 1, "exposure drift"),
        (("truth_boundary", "authorized_optimized_target_exposure"), False, "exposure drift"),
        (("source", "revision"), "0" * 40, "source revision drift"),
    ],
)
def test_config_cannot_promote_or_rebind_authority(
    path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    config = _config()
    config[path[0]][path[1]] = value

    with pytest.raises(pdr.AttributionError, match=message):
        pdr._validate_config(config)


def test_valid_config_keeps_zero_credit_boundary() -> None:
    config = _config()
    pdr._validate_config(config)
    assert config["truth_boundary"]["training_authorized_bytes"] == 0
    assert config["truth_boundary"]["model_training_executed"] is False
