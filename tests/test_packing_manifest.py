from __future__ import annotations

import hashlib
import json

import pytest

from twelve_six.packing import TextRecord, measure_packed_split
from twelve_six.packing.measure import main, measure_d03_packaged_split
from twelve_six.tokenization import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH, ByteTokenizer


def _dataset_manifest(
    *,
    output_name: str,
    source_hash: str,
    assignments: list[tuple[str, str]],
) -> dict[str, object]:
    return {
        "dataset_id": "fixture-v1",
        "dataset_identity_sha256": "c" * 64,
        "outputs": {output_name: source_hash},
        "document_assignments": [
            {"id": record_id, "split": split} for record_id, split in assignments
        ],
    }


def test_measure_packed_split_binds_exact_identities_and_counts() -> None:
    manifest = measure_packed_split(
        [TextRecord("a", "abc", "train"), TextRecord("b", "Ж", "train")],
        ByteTokenizer(),
        dataset_id="fixture-v1",
        dataset_identity_sha256="a" * 64,
        source_jsonl_sha256="b" * 64,
        split="train",
    )

    assert manifest.schema_version == 2
    assert manifest.dataset_id == "fixture-v1"
    assert manifest.tokenizer_config_sha256 == BYTE_TOKENIZER_HASH
    assert manifest.tokenizer_vocab_sha256 == BYTE_VOCAB_HASH
    assert manifest.sequence_length == 128
    assert manifest.document_count == 2
    assert manifest.codepoint_count == 4
    assert manifest.utf8_byte_count == 5
    assert manifest.token_count == 5
    assert manifest.causal_loss_token_count == 3
    assert manifest.packed_example_count == 2
    assert manifest.packed_input_token_count == 5
    assert manifest.packed_capacity_token_count == 256
    assert manifest.masked_fill_position_count == 251
    assert manifest.documents_without_causal_pair == 0
    assert manifest.fertility_ratio == (5, 4)
    assert manifest.packed_input_utilization_ratio == (5, 256)
    assert len(manifest.manifest_sha256) == 64


def test_measure_packed_split_rejects_noncanonical_sequence_length() -> None:
    with pytest.raises(ValueError, match="sequence_length must be 128"):
        measure_packed_split(
            [TextRecord("a", "abc", "train")],
            ByteTokenizer(),
            dataset_id="fixture-v1",
            dataset_identity_sha256="a" * 64,
            source_jsonl_sha256="b" * 64,
            split="train",
            sequence_length=4,
        )


def test_measure_packed_split_reports_documents_without_causal_pair() -> None:
    manifest = measure_packed_split(
        [TextRecord("empty", "", "validation"), TextRecord("one", "A", "validation")],
        ByteTokenizer(),
        dataset_id="fixture-v1",
        dataset_identity_sha256="a" * 64,
        source_jsonl_sha256="b" * 64,
        split="validation",
    )
    assert manifest.document_count == 2
    assert manifest.documents_without_causal_pair == 2
    assert manifest.packed_example_count == 0
    assert manifest.causal_loss_token_count == 0


def test_d03_measurement_verifies_source_hash_and_preserves_split(tmp_path) -> None:
    split_path = tmp_path / "train.jsonl"
    split_text = "\n".join(
        [
            json.dumps({"id": "d1", "text": "alpha"}, separators=(",", ":")),
            json.dumps({"id": "d2", "text": "βeta"}, separators=(",", ":")),
        ]
    ) + "\n"
    split_path.write_text(split_text, encoding="utf-8")
    source_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()

    dataset_manifest_path = tmp_path / "manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            _dataset_manifest(
                output_name="train.jsonl",
                source_hash=source_hash,
                assignments=[("d1", "train"), ("held", "validation"), ("d2", "train")],
            )
        ),
        encoding="utf-8",
    )

    measured = measure_d03_packaged_split(
        dataset_manifest_path,
        split_path,
        split="train",
    )
    assert measured.dataset_id == "fixture-v1"
    assert measured.source_jsonl_sha256 == source_hash
    assert measured.tokenizer_vocab_sha256 == BYTE_VOCAB_HASH
    assert measured.document_count == 2
    assert measured.split == "train"


@pytest.mark.parametrize(
    ("filename", "requested_split"),
    [("train.jsonl", "validation"), ("validation.jsonl", "train")],
)
def test_d03_measurement_rejects_cross_labeled_split(
    tmp_path,
    filename: str,
    requested_split: str,
) -> None:
    split_path = tmp_path / filename
    split_path.write_text('{"id":"d1","text":"held out"}\n', encoding="utf-8")
    source_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    dataset_manifest_path = tmp_path / "manifest.json"
    actual_split = filename.removesuffix(".jsonl")
    dataset_manifest_path.write_text(
        json.dumps(
            _dataset_manifest(
                output_name=filename,
                source_hash=source_hash,
                assignments=[("d1", actual_split)],
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="split/output mismatch"):
        measure_d03_packaged_split(
            dataset_manifest_path,
            split_path,
            split=requested_split,
        )


def test_d03_measurement_rejects_record_assignment_mismatch(tmp_path) -> None:
    split_path = tmp_path / "train.jsonl"
    split_path.write_text('{"id":"d1","text":"train"}\n', encoding="utf-8")
    source_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    dataset_manifest_path = tmp_path / "manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            _dataset_manifest(
                output_name="train.jsonl",
                source_hash=source_hash,
                assignments=[("different-id", "train")],
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="record assignment mismatch"):
        measure_d03_packaged_split(
            dataset_manifest_path,
            split_path,
            split="train",
        )


def test_d03_measurement_fails_closed_on_source_hash_mismatch(tmp_path) -> None:
    split_path = tmp_path / "validation.jsonl"
    split_path.write_text('{"id":"d1","text":"valid"}\n', encoding="utf-8")
    dataset_manifest_path = tmp_path / "manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            _dataset_manifest(
                output_name="validation.jsonl",
                source_hash="0" * 64,
                assignments=[("d1", "validation")],
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        measure_d03_packaged_split(
            dataset_manifest_path,
            split_path,
            split="validation",
        )


def test_measure_cli_prints_machine_readable_manifest(tmp_path, capsys) -> None:
    split_path = tmp_path / "train.jsonl"
    split_path.write_text('{"id":"d1","text":"abcdef"}\n', encoding="utf-8")
    source_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    dataset_manifest_path = tmp_path / "manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            _dataset_manifest(
                output_name="train.jsonl",
                source_hash=source_hash,
                assignments=[("d1", "train")],
            )
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--dataset-manifest",
                str(dataset_manifest_path),
                "--jsonl",
                str(split_path),
                "--split",
                "train",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 2
    assert payload["dataset_id"] == "fixture-v1"
    assert payload["tokenizer_config_sha256"] == BYTE_TOKENIZER_HASH
    assert payload["tokenizer_vocab_sha256"] == BYTE_VOCAB_HASH
    assert len(payload["manifest_sha256"]) == 64
