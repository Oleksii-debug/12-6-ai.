from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from twelve_six.data import rada_current_snapshot_dedup_adapter_v1 as adapter
from twelve_six.data.rada_current_snapshot_qp_authority_v1 import (
    RadaAcceptedPayloadBinding,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _row(
    parent: str,
    chunk: int,
    text: str,
    *,
    encoding: str = "utf-8",
) -> dict[str, object]:
    payload = text.encode("utf-8")
    return {
        "record_id": f"{parent}.q{chunk:05d}",
        "parent_record_id": parent,
        "source_path": f"zak/perv/text/{parent}",
        "source_encoding": encoding,
        "chunk_index": chunk,
        "normalized_bytes": len(payload),
        "normalized_sha256": hashlib.sha256(payload).hexdigest(),
        "text": text,
    }


def _candidate_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _binding(rows: list[dict[str, object]], raw: bytes) -> RadaAcceptedPayloadBinding:
    inventory = hashlib.sha256()
    encodings: Counter[str] = Counter()
    payload_hashes: list[str] = []
    payload_bytes = 0
    for row in rows:
        metadata = {key: row[key] for key in row if key != "text"}
        inventory.update(_canonical(metadata) + b"\n")
        encodings[str(row["source_encoding"])] += 1
        payload_hashes.append(str(row["normalized_sha256"]))
        payload_bytes += int(row["normalized_bytes"])
    duplicates = len(payload_hashes) - len(set(payload_hashes))
    return RadaAcceptedPayloadBinding(
        source_family=adapter.SOURCE_FAMILY,
        source_archive_sha256="1" * 64,
        source_archive_bytes=123,
        rights_policy_git_blob_sha1="2" * 40,
        rights_policy_identity_sha256="3" * 64,
        rights_scope="ARTIFACT_RETENTION_AND_REPRODUCIBILITY_ONLY",
        bulk_corpus_admission_granted=False,
        training_authority_granted=False,
        rights_recheck_for_training_required=True,
        product_parent_sha="4" * 40,
        normalizer_git_blob_sha1="5" * 40,
        normalization_config_git_blob_sha1="6" * 40,
        qp_reference_commit="7" * 40,
        qp_tool_git_blob_sha1="8" * 40,
        qp_config_git_blob_sha1="9" * 40,
        accepted_chunk_count=len(rows),
        accepted_payload_utf8_bytes=payload_bytes,
        accepted_jsonl_file_bytes=len(raw),
        accepted_jsonl_sha256=hashlib.sha256(raw).hexdigest(),
        accepted_inventory_sha256=inventory.hexdigest(),
        accepted_source_encoding_counts=tuple(sorted(encodings.items())),
        exact_duplicate_payload_hashes_observed_not_removed=duplicates,
        execution_head_sha="a" * 40,
        workflow_run_id=1,
        artifact_digest_sha256="b" * 64,
    )


def _write_candidate(
    tmp_path: Path,
    rows: list[dict[str, object]],
) -> tuple[Path, RadaAcceptedPayloadBinding]:
    raw = _candidate_bytes(rows)
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_bytes(raw)
    return candidate, _binding(rows, raw)


def test_projection_preserves_every_chunk_and_exact_duplicate_alias(
    tmp_path: Path,
) -> None:
    rows = [
        _row("d100.htm", 0, "Однаковий текст закону."),
        _row("d200.htm", 0, "Однаковий текст закону."),
        _row("d200.htm", 1, "Інший текст закону.", encoding="windows-1251"),
    ]
    candidate, binding = _write_candidate(tmp_path, rows)

    projection = adapter._read_candidate(
        candidate,
        binding,
        retain_payloads=True,
    )

    assert projection.sources is not None
    assert projection.payloads is not None
    assert len(projection.sources) == 3
    assert len(projection.payloads) == 3
    first, second, third = projection.sources
    assert first["stable_object_id"] == second["stable_object_id"]
    assert first["stable_origin_id"] != second["stable_origin_id"]
    assert first["origin_key"] != second["origin_key"]
    assert first["source_id"] == "rada-laws-qp:d100.htm.q00000"
    assert third["source_id"] == "rada-laws-qp:d200.htm.q00001"
    assert projection.receipt["exact_duplicate_payload_hashes_preserved"] == 1
    assert projection.receipt["source_object_count"] == 3


def test_projection_receipt_requires_family_replacement_and_zero_credit(
    tmp_path: Path,
) -> None:
    rows = [_row("d100.htm", 0, "Український юридичний текст.")]
    candidate, binding = _write_candidate(tmp_path, rows)
    projection = adapter._read_candidate(
        candidate,
        binding,
        retain_payloads=False,
    )

    assert projection.sources is None
    assert projection.payloads is None
    receipt = projection.receipt
    assert receipt["requires_replace_existing_source_family"] is True
    assert receipt["must_not_append_to_existing_source_family"] is True
    assert receipt["consumer_gate"] == "CURRENT_GLOBAL_CROSS_SOURCE_DEDUP_ONLY"
    assert receipt["rights_scope"] == (
        "ARTIFACT_RETENTION_AND_REPRODUCIBILITY_ONLY"
    )
    assert receipt["bulk_corpus_admission_granted"] is False
    assert receipt["training_authority_granted"] is False
    assert receipt["rights_recheck_for_training_required"] is True
    assert receipt["canonical_capacity_credited"] == 0
    assert receipt["training_authorized_bytes"] == 0
    assert receipt["authorized_optimized_target_exposure"] == 0
    assert receipt["training_executed"] is False


def test_candidate_transport_hash_mismatch_returns_no_projection(
    tmp_path: Path,
) -> None:
    rows = [_row("d100.htm", 0, "Український юридичний текст.")]
    candidate, binding = _write_candidate(tmp_path, rows)
    binding = RadaAcceptedPayloadBinding(
        **{
            **binding.__dict__,
            "accepted_jsonl_sha256": "0" * 64,
        }
    )

    with pytest.raises(
        adapter.RadaCurrentSnapshotDedupAdapterError,
        match="transport SHA-256 drift",
    ):
        adapter._read_candidate(candidate, binding, retain_payloads=True)


def test_candidate_inventory_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    rows = [_row("d100.htm", 0, "Український юридичний текст.")]
    candidate, binding = _write_candidate(tmp_path, rows)
    binding = RadaAcceptedPayloadBinding(
        **{
            **binding.__dict__,
            "accepted_inventory_sha256": "0" * 64,
        }
    )

    with pytest.raises(
        adapter.RadaCurrentSnapshotDedupAdapterError,
        match="inventory SHA-256 drift",
    ):
        adapter._read_candidate(candidate, binding, retain_payloads=True)


def test_candidate_rejects_noncanonical_order_even_when_rebound(
    tmp_path: Path,
) -> None:
    rows = [
        _row("d200.htm", 0, "Другий документ."),
        _row("d100.htm", 0, "Перший документ."),
    ]
    candidate, binding = _write_candidate(tmp_path, rows)

    with pytest.raises(
        adapter.RadaCurrentSnapshotDedupAdapterError,
        match="canonical order drift",
    ):
        adapter._read_candidate(candidate, binding, retain_payloads=True)


@pytest.mark.parametrize(
    ("field", "replacement", "pattern"),
    [
        ("chunk_index", True, "chunk_index invalid"),
        ("normalized_bytes", True, "normalized byte drift"),
        ("source_encoding", "utf-16", "source_encoding invalid"),
        ("parent_record_id", "../d100.htm", "parent_record_id invalid"),
        ("source_path", "/tmp/d100.htm", "source_path invalid"),
    ],
)
def test_candidate_row_type_and_path_aliases_fail_closed(
    tmp_path: Path,
    field: str,
    replacement: object,
    pattern: str,
) -> None:
    row = _row("d100.htm", 0, "Український юридичний текст.")
    row[field] = replacement
    raw = _candidate_bytes([row])
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_bytes(raw)
    binding = _binding([row], raw)

    with pytest.raises(
        adapter.RadaCurrentSnapshotDedupAdapterError,
        match=pattern,
    ):
        adapter._read_candidate(candidate, binding, retain_payloads=True)


def test_candidate_text_hash_and_byte_binding_fail_closed(tmp_path: Path) -> None:
    row = _row("d100.htm", 0, "Український юридичний текст.")
    row["text"] = "Підмінений текст."
    raw = _candidate_bytes([row])
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_bytes(raw)
    binding = _binding([row], raw)

    with pytest.raises(
        adapter.RadaCurrentSnapshotDedupAdapterError,
        match="normalized byte drift",
    ):
        adapter._read_candidate(candidate, binding, retain_payloads=True)


def test_candidate_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    line = (
        b'{"chunk_index":0,"chunk_index":0,"normalized_bytes":1,'
        b'"normalized_sha256":"' + b"0" * 64 + b'",'
        b'"parent_record_id":"d1.htm","record_id":"d1.htm.q00000",'
        b'"source_encoding":"utf-8","source_path":"zak/perv/text/d1.htm",'
        b'"text":"x"}\n'
    )
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_bytes(line)
    binding = RadaAcceptedPayloadBinding(
        source_family=adapter.SOURCE_FAMILY,
        source_archive_sha256="1" * 64,
        source_archive_bytes=1,
        rights_policy_git_blob_sha1="2" * 40,
        rights_policy_identity_sha256="3" * 64,
        rights_scope="ARTIFACT_RETENTION_AND_REPRODUCIBILITY_ONLY",
        bulk_corpus_admission_granted=False,
        training_authority_granted=False,
        rights_recheck_for_training_required=True,
        product_parent_sha="4" * 40,
        normalizer_git_blob_sha1="5" * 40,
        normalization_config_git_blob_sha1="6" * 40,
        qp_reference_commit="7" * 40,
        qp_tool_git_blob_sha1="8" * 40,
        qp_config_git_blob_sha1="9" * 40,
        accepted_chunk_count=1,
        accepted_payload_utf8_bytes=1,
        accepted_jsonl_file_bytes=len(line),
        accepted_jsonl_sha256=hashlib.sha256(line).hexdigest(),
        accepted_inventory_sha256="a" * 64,
        accepted_source_encoding_counts=(("utf-8", 1),),
        exact_duplicate_payload_hashes_observed_not_removed=0,
        execution_head_sha="b" * 40,
        workflow_run_id=1,
        artifact_digest_sha256="c" * 64,
    )

    with pytest.raises(
        adapter.RadaCurrentSnapshotDedupAdapterError,
        match="duplicate candidate JSON key",
    ):
        adapter._read_candidate(candidate, binding, retain_payloads=True)


def test_candidate_is_opened_once_and_authenticated_from_same_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_row("d100.htm", 0, "Український юридичний текст.")]
    candidate, binding = _write_candidate(tmp_path, rows)
    real_open = Path.open
    open_count = 0

    def counted_open(path: Path, *args: object, **kwargs: object):
        nonlocal open_count
        if path == candidate:
            open_count += 1
            if open_count > 1:
                raise AssertionError("candidate JSONL reopened")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    adapter._read_candidate(candidate, binding, retain_payloads=True)
    assert open_count == 1


def test_candidate_path_swap_before_descriptor_lock_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_row("d100.htm", 0, "Український юридичний текст.")]
    candidate, binding = _write_candidate(tmp_path, rows)
    real_lstat = os.lstat

    def substituted_lstat(path: Path):
        observed = real_lstat(path)
        if path == candidate:
            return SimpleNamespace(
                st_mode=observed.st_mode,
                st_dev=observed.st_dev + 1,
                st_ino=observed.st_ino,
            )
        return observed

    monkeypatch.setattr(os, "lstat", substituted_lstat)
    with pytest.raises(
        adapter.RadaCurrentSnapshotDedupAdapterError,
        match="path changed before descriptor lock",
    ):
        adapter._read_candidate(candidate, binding, retain_payloads=True)
