from __future__ import annotations

import ast
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


def test_bootstrap_has_no_eager_twelve_six_imports() -> None:
    source = Path(target.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    eager: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "twelve_six" or module.startswith("twelve_six."):
                eager.append(module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "twelve_six" or alias.name.startswith("twelve_six."):
                    eager.append(alias.name)
    assert eager == []


def test_bootstrap_pins_behavior_import_closure_before_runtime_load() -> None:
    expected = {
        "src/twelve_six/__init__.py":
            "5433166c507bc845bd12d8d5c4145f1fbedda204",
        "src/twelve_six/data/_data232_decontamination_matching.py":
            "afa70511f82dc81d9c9f85e3d0b67eba343004f9",
        "src/twelve_six/data/decontamination_authority_v2.py":
            "3ca8f21945c02f692c130a015e036673fa24e7af",
        "src/twelve_six/data/document_quality.py":
            "b1461263034b4fb9510479b20c9697e22faa5f97",
        "src/twelve_six/data/quality_granularity.py":
            "513523b86824c423cad97352b3abb3d1241531b9",
        "src/twelve_six/data/eval647_future_training_exclusion_v1.py":
            "5516577a0720150a7ec12c1bf8898972968e6970",
        "src/twelve_six/data/eval647_reserved_decontamination_v1.py":
            "ce33771c9fb4a6cc421e2f8e1f6f232c119bec71",
    }
    for path, blob in expected.items():
        assert target.EXPECTED_DEPENDENCY_BLOBS[path] == blob

    execute_source = ast.parse(
        Path(target.__file__).read_text(encoding="utf-8")
    )
    execute_node = next(
        node
        for node in execute_source.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute"
    )
    calls = [
        node.func.id
        for node in ast.walk(execute_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.index("verify_local_authority") < calls.index("load_bound_runtime")



def test_immutable_writer_resumes_identical_and_rejects_divergence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evidence.json"
    payload = b"{\"ok\":true}\n"
    target.write_immutable_bytes(output, payload, label="synthetic")
    assert output.read_bytes() == payload

    target.write_immutable_bytes(output, payload, label="synthetic")
    assert output.read_bytes() == payload

    with pytest.raises(
        target.RadaPostData232Error,
        match="refusing to overwrite divergent durable evidence",
    ):
        target.write_immutable_bytes(
            output,
            b"{\"ok\":false}\n",
            label="synthetic",
        )


def test_immutable_writer_recovers_matching_interrupted_temp(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evidence.json"
    payload = b"{\"resume\":true}\n"
    temp = output.with_name(output.name + ".tmp")
    temp.write_bytes(payload)

    target.write_immutable_bytes(output, payload, label="synthetic")

    assert output.read_bytes() == payload
    assert not temp.exists()


def test_immutable_writer_rejects_divergent_interrupted_temp(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evidence.json"
    temp = output.with_name(output.name + ".tmp")
    temp.write_bytes(b"stale\n")

    with pytest.raises(
        target.RadaPostData232Error,
        match="divergent interrupted temp evidence",
    ):
        target.write_immutable_bytes(
            output,
            b"fresh\n",
            label="synthetic",
        )



def _zero_credit_truth() -> dict[str, object]:
    return {
        "canonical_capacity_credited": 0,
        "authorized_optimized_target_exposure": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_training_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "scale_promotion_authorized": False,
        "model_architecture_or_hyperparameters_selected": False,
    }


def test_zero_credit_truth_accepts_exact_parent_boundary() -> None:
    target.verify_zero_credit_truth(
        _zero_credit_truth(),
        label="synthetic",
        require_model_selection_false=True,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canonical_capacity_credited", 1),
        ("authorized_optimized_target_exposure", 1),
        ("authorized_unique_loss_positions", 1),
        ("authorized_training_exposure", 1),
        ("tokenizer_fit_authorized", True),
        ("training_executed", True),
        ("learned_weights_created", True),
        ("final_test_outcomes_read", True),
        ("paid_compute_used", True),
        ("scale_promotion_authorized", True),
        ("model_architecture_or_hyperparameters_selected", True),
    ],
)
def test_zero_credit_truth_rejects_every_authority_widening(
    field: str,
    value: object,
) -> None:
    truth = _zero_credit_truth()
    truth[field] = value
    with pytest.raises(target.RadaPostData232Error):
        target.verify_zero_credit_truth(
            truth,
            label="synthetic",
            require_model_selection_false=True,
        )


def test_zero_credit_truth_rejects_boolean_zero_impostor() -> None:
    truth = _zero_credit_truth()
    truth["authorized_training_exposure"] = False
    with pytest.raises(
        target.RadaPostData232Error,
        match="authorized_training_exposure",
    ):
        target.verify_zero_credit_truth(
            truth,
            label="synthetic",
            require_model_selection_false=True,
        )
