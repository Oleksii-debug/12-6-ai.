from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from tools.filter_d03_rada_bulk_quality_privacy import (
    QualityPrivacyError,
    materialize_quality_privacy_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/data/d03_rada_bulk_quality_privacy_v1.json").read_text(
        encoding="utf-8"
    )
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parent_record(
    record_id: str,
    text: str,
    *,
    source_encoding: str = "utf-8",
) -> dict[str, object]:
    normalized = text.encode("utf-8")
    raw_encoding = "cp1251" if source_encoding == "windows-1251" else "utf-8"
    raw = ("<p>" + text + "</p>").encode(raw_encoding)
    return {
        "record_id": record_id,
        "source_path": record_id.rsplit(".", 1)[-1] + ".htm",
        "source_encoding": source_encoding,
        "raw_bytes": len(raw),
        "raw_sha256": _sha256(raw),
        "normalized_bytes": len(normalized),
        "normalized_sha256": _sha256(normalized),
        "text": text,
    }


def _parent_payload(
    texts: list[str],
    *,
    source_encodings: list[str] | None = None,
) -> tuple[bytes, dict[str, object], str]:
    encodings = source_encodings or ["utf-8"] * len(texts)
    assert len(encodings) == len(texts)
    rows = [
        _parent_record(
            f"ua.rada.open-data.laws-texts.d{100 + index}",
            text,
            source_encoding=encodings[index],
        )
        for index, text in enumerate(texts)
    ]
    jsonl = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    metadata = [
        {key: value for key, value in row.items() if key != "text"}
        for row in rows
    ]
    inventory_hasher = hashlib.sha256()
    for row in metadata:
        inventory_hasher.update(_canonical_bytes(row))
        inventory_hasher.update(b"\n")
    source_encoding_counts = Counter(str(row["source_encoding"]) for row in rows)
    manifest: dict[str, object] = {
        "schema_version": "12-6.d03-rada-bulk-normalization-manifest.v1",
        "worker_id": "D03-RADA-BULK-NORMALIZATION-20260826",
        "local_free_only": True,
        "parent_probe": {
            "pr": 618,
            "head_sha": "a" * 40,
            "probe_config_identity_sha256": "b" * 64,
            "probe_report_sha256": "c" * 64,
            "archive_sha256": "d" * 64,
            "entry_identity_sha256": "e" * 64,
        },
        "normalization": {
            "name": "RADA_VISIBLE_TEXT_HTML_UTF8_CP1251_NFKC_V1",
            "record_count": len(rows),
            "nonempty_record_count": sum(bool(row["normalized_bytes"]) for row in rows),
            "raw_bytes": sum(int(row["raw_bytes"]) for row in rows),
            "normalized_bytes_observed_not_credited": sum(
                int(row["normalized_bytes"]) for row in rows
            ),
            "source_encoding_counts": dict(sorted(source_encoding_counts.items())),
            "normalized_record_inventory_sha256": inventory_hasher.hexdigest(),
            "jsonl_sha256": _sha256(jsonl),
        },
        "records": metadata,
        "gates": {
            "exact_probe_inventory": "PASS",
            "canonical_normalization": "PASS",
            "quality": "NOT_RUN",
            "privacy": "NOT_RUN",
            "global_cross_source_dedup": "NOT_RUN",
            "evaluation_decontamination": "NOT_RUN",
            "balance_diversity": "NOT_RUN",
            "corpus_materialization": "NOT_RUN",
            "unique_loss_ledger": "NOT_RUN",
        },
        "training_authorized_bytes": 0,
        "normalized_capacity_credited": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "paid_compute_used": False,
        "research_corpus_v1_released": False,
        "safe_result": "NORMALIZED_RECORD_MATERIALIZATION_ONLY_DOWNSTREAM_GATES_REQUIRED",
    }
    manifest["manifest_identity_sha256"] = _sha256(_canonical_bytes(manifest))
    manifest_bytes = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    return jsonl, manifest, _sha256(manifest_bytes)


def _run(
    texts: list[str],
    *,
    source_encodings: list[str] | None = None,
) -> tuple[bytes, dict[str, object]]:
    jsonl, manifest, manifest_transport_sha = _parent_payload(
        texts,
        source_encodings=source_encodings,
    )
    return materialize_quality_privacy_candidate(
        jsonl,
        manifest,
        CONFIG,
        parent_manifest_sha256=manifest_transport_sha,
    )


def test_production_contract_tracks_current_mixed_encoding_parent() -> None:
    assert CONFIG["parent_normalization"]["head_sha"] == (
        "ae79b078f849513dc202bcb723a4145455309e35"
    )
    assert CONFIG["output_contract"]["preserve_source_encoding_provenance"] is True
    assert "source_encoding" in CONFIG["output_contract"]["accepted_jsonl_fields"]


def test_clean_candidate_is_deterministic_and_keeps_training_closed() -> None:
    text = (
        "Верховна Рада України постановляє встановити порядок застосування норм закону "
        "для органів державної влади та забезпечити відкритість нормативної інформації. "
    ) * 12
    first_jsonl, first_report = _run([text])
    second_jsonl, second_report = _run([text])

    assert first_jsonl == second_jsonl
    assert first_report == second_report
    assert first_report["filter_result"]["accepted_chunk_count"] > 0
    assert first_report["filter_result"]["rejected_chunk_count"] == 0
    assert first_report["filter_result"]["accepted_source_encoding_counts"] == {
        "utf-8": first_report["filter_result"]["accepted_chunk_count"]
    }
    assert first_report["training_authorized_bytes"] == 0
    assert first_report["canonical_capacity_credited"] == 0
    assert first_report["model_training_executed"] is False
    assert first_report["optimizer_updates"] == 0
    assert first_report["safe_result"] == (
        "QUALITY_PRIVACY_FILTERED_CANDIDATE_ONLY_DOWNSTREAM_GATES_REQUIRED"
    )

    identity = first_report["report_identity_sha256"]
    unsigned = copy.deepcopy(first_report)
    unsigned.pop("report_identity_sha256")
    assert identity == _sha256(_canonical_bytes(unsigned))


def test_windows_1251_parent_provenance_survives_accepted_chunking() -> None:
    utf8_text = (
        "Новий нормативний акт містить достатньо українського тексту для проходження "
        "перевірки якості та подальшого детермінованого опрацювання. "
    ) * 3
    cp1251_text = (
        "Старий нормативний акт містить достатньо українського тексту для проходження "
        "перевірки якості та збереження походження кодування. "
    ) * 3
    accepted_jsonl, report = _run(
        [utf8_text, cp1251_text],
        source_encodings=["utf-8", "windows-1251"],
    )
    rows = [json.loads(line) for line in accepted_jsonl.decode("utf-8").splitlines()]

    assert {row["source_encoding"] for row in rows} == {"utf-8", "windows-1251"}
    assert report["parent_normalization"]["source_encoding_counts"] == {
        "utf-8": 1,
        "windows-1251": 1,
    }
    assert report["filter_result"]["accepted_source_encoding_counts"]["utf-8"] > 0
    assert report["filter_result"]["accepted_source_encoding_counts"]["windows-1251"] > 0
    assert report["training_authorized_bytes"] == 0


def test_parent_encoding_count_mismatch_fails_closed() -> None:
    text = "Нормативний український текст достатньої довжини для перевірки походження. " * 5
    jsonl, manifest, _ = _parent_payload([text])
    tampered = copy.deepcopy(manifest)
    tampered["normalization"]["source_encoding_counts"] = {"windows-1251": 1}
    tampered.pop("manifest_identity_sha256")
    tampered["manifest_identity_sha256"] = _sha256(_canonical_bytes(tampered))
    manifest_bytes = json.dumps(
        tampered,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"

    with pytest.raises(QualityPrivacyError, match="counts do not match records"):
        materialize_quality_privacy_candidate(
            jsonl,
            tampered,
            CONFIG,
            parent_manifest_sha256=_sha256(manifest_bytes),
        )


def test_email_and_phone_chunks_are_rejected_without_sensitive_output() -> None:
    email = "contact.person@example.org"
    phone = "+380 50 123 45 67"
    email_text = (
        "Офіційний нормативний текст містить службову контактну інформацію для прикладу "
        f"та адресу {email}. Інший зміст документа залишається нормативним і читабельним."
    )
    phone_text = (
        "Офіційний нормативний текст містить контактний номер для прикладу "
        f"{phone}. Інший зміст документа залишається нормативним і читабельним для перевірки."
    )

    accepted_jsonl, report = _run([email_text, phone_text])
    assert accepted_jsonl == b""
    assert report["filter_result"]["accepted_chunk_count"] == 0
    assert report["filter_result"]["rejected_chunk_count"] == 2
    assert report["filter_result"]["rejection_reasons"] == {
        "pii_email": 1,
        "pii_phone": 1,
    }
    assert report["filter_result"]["rejected_text_emitted"] is False
    assert report["filter_result"]["rejected_hashes_emitted"] is False
    serialized = json.dumps(report, ensure_ascii=False)
    assert email not in serialized
    assert phone not in serialized


def test_parent_jsonl_transport_tamper_fails_closed() -> None:
    text = "Нормативний український текст для перевірки цілісності документа. " * 5
    jsonl, manifest, manifest_transport_sha = _parent_payload([text])
    tampered = jsonl.replace("український".encode(), "змінений".encode(), 1)

    with pytest.raises(QualityPrivacyError, match="JSONL SHA-256 mismatch"):
        materialize_quality_privacy_candidate(
            tampered,
            manifest,
            CONFIG,
            parent_manifest_sha256=manifest_transport_sha,
        )


def test_parent_manifest_self_hash_tamper_fails_closed() -> None:
    text = "Нормативний текст достатньої довжини для стабільної перевірки маніфесту. " * 5
    jsonl, manifest, manifest_transport_sha = _parent_payload([text])
    tampered = copy.deepcopy(manifest)
    tampered["normalization"]["record_count"] = 999

    with pytest.raises(QualityPrivacyError, match="source-encoding count total drift|manifest self-hash mismatch"):
        materialize_quality_privacy_candidate(
            jsonl,
            tampered,
            CONFIG,
            parent_manifest_sha256=manifest_transport_sha,
        )


def test_parent_record_metadata_mismatch_fails_closed() -> None:
    text = "Нормативний текст достатньої довжини для перевірки метаданих запису. " * 5
    jsonl, manifest, _ = _parent_payload([text])
    tampered = copy.deepcopy(manifest)
    tampered["records"][0]["normalized_bytes"] += 1
    tampered.pop("manifest_identity_sha256")
    tampered["manifest_identity_sha256"] = _sha256(_canonical_bytes(tampered))
    manifest_bytes = json.dumps(
        tampered,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"

    with pytest.raises(QualityPrivacyError, match="metadata mismatch"):
        materialize_quality_privacy_candidate(
            jsonl,
            tampered,
            CONFIG,
            parent_manifest_sha256=_sha256(manifest_bytes),
        )


def test_truth_boundary_mutation_fails_closed() -> None:
    text = "Нормативний текст достатньої довжини для перевірки меж повноважень. " * 5
    jsonl, manifest, manifest_transport_sha = _parent_payload([text])
    weakened = copy.deepcopy(CONFIG)
    weakened["claim_boundary"]["training_authorized_bytes"] = 1

    with pytest.raises(QualityPrivacyError, match="training bytes must remain zero"):
        materialize_quality_privacy_candidate(
            jsonl,
            manifest,
            weakened,
            parent_manifest_sha256=manifest_transport_sha,
        )


def test_duplicate_manifest_record_id_fails_closed() -> None:
    text = "Нормативний текст достатньої довжини для перевірки дубльованих записів. " * 5
    jsonl, manifest, _ = _parent_payload([text])
    tampered = copy.deepcopy(manifest)
    tampered["records"].append(copy.deepcopy(tampered["records"][0]))
    tampered.pop("manifest_identity_sha256")
    tampered["manifest_identity_sha256"] = _sha256(_canonical_bytes(tampered))
    manifest_bytes = json.dumps(
        tampered,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"

    with pytest.raises(QualityPrivacyError, match="duplicate manifest record_id"):
        materialize_quality_privacy_candidate(
            jsonl,
            tampered,
            CONFIG,
            parent_manifest_sha256=_sha256(manifest_bytes),
        )


def test_exact_duplicate_chunks_are_observed_but_left_for_global_dedup() -> None:
    text = "Нормативний текст однакової структури та достатньої довжини для перевірки. " * 5
    accepted_jsonl, report = _run([text, text])
    assert accepted_jsonl
    assert report["filter_result"]["exact_duplicate_accepted_hashes_observed_not_removed"] > 0
    assert report["gates"]["global_cross_source_dedup"] == "NOT_RUN"
    assert report["training_authorized_bytes"] == 0