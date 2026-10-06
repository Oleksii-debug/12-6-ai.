from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_d03_rada_current_postdata232_g05_g06_v1 as target


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _decision(label: str) -> str:
    return _sha(label.encode("utf-8"))


def _partial_authority(
    *,
    record_id: str,
    text: str,
    split: int,
) -> dict[str, object]:
    left = text[:split]
    right = text[split:]
    left_raw = left.encode("utf-8")
    right_raw = right.encode("utf-8")
    full_raw = text.encode("utf-8")
    return {
        "records": [
            {
                "record_id": record_id,
                "mode": "uk",
                "payload_sha256": _sha(full_raw),
                "utf8_bytes": len(full_raw),
                "authoritative_unit": "BOUNDED_NATURAL_LANGUAGE_WINDOW",
                "status": "RETAIN_PARTIAL",
                "retained_utf8_bytes": len(left_raw),
                "rejected_utf8_bytes": len(right_raw),
                "units": [
                    {
                        "unit_id": f"{record_id}#quality-window-0000",
                        "start_char": 0,
                        "end_char": split,
                        "payload_sha256": _sha(left_raw),
                        "utf8_bytes": len(left_raw),
                        "accepted": True,
                        "decision_sha256": _decision("accept"),
                    },
                    {
                        "unit_id": f"{record_id}#quality-window-0001",
                        "start_char": split,
                        "end_char": len(text),
                        "payload_sha256": _sha(right_raw),
                        "utf8_bytes": len(right_raw),
                        "accepted": False,
                        "decision_sha256": _decision("reject"),
                    },
                ],
            }
        ]
    }


def test_partial_window_materialization_is_exact_and_text_preserving() -> None:
    record_id = "rada.synthetic.1"
    text = "абвгДЕЖЗ"
    split = 4
    inputs = [{"id": record_id, "text": text, "mode": "uk"}]
    metadata = {
        record_id: {
            "source_id": "rada-source-1",
            "family": "rada-current",
            "mode": "uk",
        }
    }
    quality = _partial_authority(
        record_id=record_id,
        text=text,
        split=split,
    )

    output, stats, detail = target._materialize_quality_survivors_with_partial(
        inputs,
        metadata,
        quality,
    )

    retained = text[:split]
    retained_raw = retained.encode("utf-8")
    expected_id = (
        f"{record_id}#quality-window-0000:"
        f"{_sha(retained_raw)}"
    )
    assert output == [
        {
            "record_id": expected_id,
            "source_id": "rada-source-1",
            "family": "rada-current",
            "modality": "uk",
            "normalized_payload": retained,
        }
    ]
    assert stats == {
        "g05_reject_documents": 0,
        "g05_partial_documents": 1,
        "g05_rejected_units": 1,
        "g05_rejected_utf8_bytes": len(text[split:].encode("utf-8")),
    }
    assert detail["partial_documents"] == 1
    assert detail["partial_emitted_units"] == 1
    assert detail["partial_rejected_units"] == 1
    assert detail["input_utf8_bytes"] == len(text.encode("utf-8"))
    assert (
        detail["input_utf8_bytes"]
        == detail["retained_utf8_bytes"] + detail["rejected_utf8_bytes"]
    )
    assert detail["partial_unit_projection_count"] == 1


def test_partial_window_materialization_rejects_partition_gap() -> None:
    record_id = "rada.synthetic.2"
    text = "abcdefgh"
    quality = _partial_authority(
        record_id=record_id,
        text=text,
        split=4,
    )
    units = quality["records"][0]["units"]
    units[1]["start_char"] = 5

    with pytest.raises(
        target.RadaPostData232Error,
        match="unit partition drift",
    ):
        target._materialize_quality_survivors_with_partial(
            [{"id": record_id, "text": text, "mode": "uk"}],
            {
                record_id: {
                    "source_id": "rada-source-2",
                    "family": "rada-current",
                    "mode": "uk",
                }
            },
            quality,
        )


def test_partial_window_materialization_rejects_payload_hash_drift() -> None:
    record_id = "rada.synthetic.3"
    text = "abcdefgh"
    quality = _partial_authority(
        record_id=record_id,
        text=text,
        split=4,
    )
    quality["records"][0]["units"][0]["payload_sha256"] = "0" * 64

    with pytest.raises(
        target.RadaPostData232Error,
        match="unit payload drift",
    ):
        target._materialize_quality_survivors_with_partial(
            [{"id": record_id, "text": text, "mode": "uk"}],
            {
                record_id: {
                    "source_id": "rada-source-3",
                    "family": "rada-current",
                    "mode": "uk",
                }
            },
            quality,
        )


def test_strict_json_rejects_duplicate_members_and_nonfinite_values() -> None:
    with pytest.raises(
        target.RadaPostData232Error,
        match="duplicate JSON member",
    ):
        target.strict_loads(
            b'{"a":1,"a":2}',
            "duplicate",
        )

    with pytest.raises(
        target.RadaPostData232Error,
        match="non-finite JSON constant",
    ):
        target.strict_loads(
            b'{"a":NaN}',
            "nonfinite",
        )


def test_self_hash_rejects_caller_selected_reseal_against_external_pin() -> None:
    core = {
        "schema_version": "synthetic.v1",
        "value": 1,
    }
    document = {
        **core,
        "identity_sha256": _sha(target.canonical(core)),
    }
    target.verify_self_hash(
        document,
        "identity_sha256",
        label="synthetic",
        expected=document["identity_sha256"],
    )

    mutated = {
        "schema_version": "synthetic.v1",
        "value": 2,
    }
    resealed = {
        **mutated,
        "identity_sha256": _sha(target.canonical(mutated)),
    }
    with pytest.raises(
        target.RadaPostData232Error,
        match="not independently expected",
    ):
        target.verify_self_hash(
            resealed,
            "identity_sha256",
            label="synthetic",
            expected=document["identity_sha256"],
        )


def test_canonical_serialization_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        json.loads(target.canonical({"x": float("nan")}).decode("utf-8"))
