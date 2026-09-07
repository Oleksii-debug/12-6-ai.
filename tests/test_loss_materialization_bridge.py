from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from twelve_six.packing import (
    DEFAULT_SEQUENCE_LENGTH,
    MATERIALIZATION_SCHEMA,
    PACKING_CONFIG_HASH,
    LossMaterializationDocument,
    LossMaterializationError,
    build_postpack_loss_materialization,
    tokenizer_identity_sha256,
)
from twelve_six.tokenization import BYTE_TOKENIZER_VERSION, ByteTokenizer


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _document(
    document_id: str,
    text: str,
    *,
    split: str = "train",
    evaluation_reserved: bool = False,
    reserved_target_ranges: tuple[tuple[int, int], ...] = (),
) -> LossMaterializationDocument:
    payload = text.encode("utf-8")
    return LossMaterializationDocument(
        document_id=document_id,
        text=text,
        source_id=f"source:{document_id}",
        language="en",
        modality="text",
        family_id="family.en",
        normalized_payload_sha256=hashlib.sha256(payload).hexdigest(),
        source_bytes=len(payload),
        split=split,
        dedup_cluster_id=f"cluster:{document_id}",
        evaluation_reserved=evaluation_reserved,
        reserved_target_ranges=reserved_target_ranges,
    )


def _bindings() -> dict[str, str]:
    return {
        "normalization": _sha("normalization"),
        "evaluation_reservations": _sha("evaluation-reservations"),
        "dedup": _sha("dedup"),
        "split": _sha("split"),
        "packing": _sha("packing-stage"),
    }


def _build(documents: list[LossMaterializationDocument]) -> dict:
    return build_postpack_loss_materialization(
        documents,
        ByteTokenizer(),
        terminal_corpus_authority_identity_sha256=_sha("terminal-corpus"),
        stage_bindings=_bindings(),
    )


def test_bridge_emits_complete_v2_spans_from_canonical_overlap_packing() -> None:
    text = "x" * (DEFAULT_SEQUENCE_LENGTH + 2)
    materialization = _build([_document("doc", text)])

    assert materialization["schema_version"] == MATERIALIZATION_SCHEMA
    assert materialization["tokenizer"]["name"] == BYTE_TOKENIZER_VERSION
    assert materialization["tokenizer"]["identity_sha256"] == tokenizer_identity_sha256(
        ByteTokenizer()
    )
    assert materialization["tokenizer"]["source_bytes_are_loss_positions"] is False
    assert materialization["packing"]["identity_sha256"] == PACKING_CONFIG_HASH
    assert materialization["packing"]["complete_one_pass"] is True

    [document] = materialization["documents"]
    assert document["token_count"] == len(text)
    assert document["eligible_target_ranges"] == [[1, len(text)]]

    first, second = materialization["packing"]["packs"]
    assert first["token_count"] == DEFAULT_SEQUENCE_LENGTH
    assert first["loss_spans"] == [
        {
            "document_id": "doc",
            "target_start": 1,
            "target_end": DEFAULT_SEQUENCE_LENGTH,
            "pack_target_start": 1,
        }
    ]
    assert second["loss_spans"] == [
        {
            "document_id": "doc",
            "target_start": DEFAULT_SEQUENCE_LENGTH,
            "target_end": len(text),
            "pack_target_start": 1,
        }
    ]
    assert sum(
        span["target_end"] - span["target_start"]
        for pack in materialization["packing"]["packs"]
        for span in pack["loss_spans"]
    ) == len(text) - 1


def test_bridge_is_order_independent_and_keeps_heldout_documents_unpacked() -> None:
    train = _document("train", "abcdef")
    validation = _document(
        "validation",
        "reserved",
        split="validation",
        evaluation_reserved=True,
    )
    forward = _build([train, validation])
    reverse = _build([validation, train])
    assert forward == reverse
    assert [item["document_id"] for item in forward["documents"]] == [
        "train",
        "validation",
    ]
    validation_row = forward["documents"][1]
    assert validation_row["eligible_target_ranges"] == []
    assert all(
        span["document_id"] == "train"
        for pack in forward["packing"]["packs"]
        for span in pack["loss_spans"]
    )


def test_bridge_fails_closed_on_in_document_training_reservations() -> None:
    document = _document(
        "doc",
        "abcdef",
        reserved_target_ranges=((2, 4),),
    )
    with pytest.raises(LossMaterializationError, match="no in-document reservation masking"):
        _build([document])


def test_bridge_allows_heldout_reservation_metadata_without_training_exposure() -> None:
    document = _document(
        "validation",
        "abcdef",
        split="validation",
        evaluation_reserved=True,
        reserved_target_ranges=((2, 4),),
    )
    materialization = _build([document])
    [row] = materialization["documents"]
    assert row["reserved_target_ranges"] == [[2, 4]]
    assert row["eligible_target_ranges"] == []
    assert materialization["packing"]["packs"] == []


def test_bridge_rejects_payload_or_stage_identity_drift() -> None:
    document = _document("doc", "abcdef")
    with pytest.raises(LossMaterializationError, match="payload hash"):
        replace(document, normalized_payload_sha256=_sha("wrong"))

    bindings = _bindings()
    bindings["packing"] = "bad"
    with pytest.raises(LossMaterializationError, match="stage_bindings.packing"):
        build_postpack_loss_materialization(
            [document],
            ByteTokenizer(),
            terminal_corpus_authority_identity_sha256=_sha("terminal-corpus"),
            stage_bindings=bindings,
        )


def test_materialization_identity_changes_on_policy_binding_change() -> None:
    document = _document("doc", "abcdef")
    first = _build([document])
    changed_bindings = _bindings()
    changed_bindings["split"] = _sha("different-split")
    second = build_postpack_loss_materialization(
        [document],
        ByteTokenizer(),
        terminal_corpus_authority_identity_sha256=_sha("terminal-corpus"),
        stage_bindings=changed_bindings,
    )
    assert first["materialization_identity_sha256"] != second[
        "materialization_identity_sha256"
    ]
    assert first["packing"] == second["packing"]
