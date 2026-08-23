import pytest

from twelve_six.posttraining.contracts import (
    DatasetRecord,
    RecordKind,
    Split,
    SyntheticProvenance,
)
from twelve_six.posttraining.dataset_adapters import (
    TRL_TRAINER_RECORD_KINDS,
    DatasetSchemaError,
    to_trl_example,
    validate_trl_compatible_record,
)

HEX_A = "a" * 64


def _provenance() -> SyntheticProvenance:
    return SyntheticProvenance(source_id="human-1", content_sha256=HEX_A)


def _record(kind: RecordKind, payload: dict[str, object], split: Split = Split.TRAIN) -> DatasetRecord:
    return DatasetRecord(
        record_id=f"record-{kind.value}",
        kind=kind,
        split=split,
        payload=payload,
        provenance=_provenance(),
    )


def test_current_trainer_kind_mapping_is_explicit() -> None:
    assert TRL_TRAINER_RECORD_KINDS["SFTTrainer"] == (
        RecordKind.LANGUAGE_MODELING,
        RecordKind.PROMPT_COMPLETION,
    )
    assert TRL_TRAINER_RECORD_KINDS["DPOTrainer"] == (RecordKind.PREFERENCE,)
    assert TRL_TRAINER_RECORD_KINDS["GRPOTrainer"] == (RecordKind.PROMPT_ONLY,)
    assert TRL_TRAINER_RECORD_KINDS["PRMTrainer"] == (RecordKind.STEPWISE_SUPERVISION,)


def test_prompt_completion_standard_shape_round_trips_without_runtime_dependency() -> None:
    record = _record(
        RecordKind.PROMPT_COMPLETION,
        {"prompt": "2 + 2 =", "completion": " 4"},
    )
    assert to_trl_example(record, for_training=True) == record.payload


def test_conversational_prompt_completion_is_supported() -> None:
    record = _record(
        RecordKind.PROMPT_COMPLETION,
        {
            "prompt": [{"role": "user", "content": "2 + 2?"}],
            "completion": [{"role": "assistant", "content": "4"}],
        },
    )
    validate_trl_compatible_record(record)


def test_preference_requires_distinct_candidates() -> None:
    record = _record(
        RecordKind.PREFERENCE,
        {"prompt": "answer", "chosen": "same", "rejected": "same"},
    )
    with pytest.raises(DatasetSchemaError, match="must differ"):
        validate_trl_compatible_record(record)


def test_stepwise_supervision_requires_aligned_boolean_labels() -> None:
    record = _record(
        RecordKind.STEPWISE_SUPERVISION,
        {
            "prompt": "reason",
            "completions": ["step 1", "step 2"],
            "labels": [True],
        },
    )
    with pytest.raises(DatasetSchemaError, match="one-to-one"):
        validate_trl_compatible_record(record)


def test_held_out_record_cannot_be_converted_as_training_input() -> None:
    record = _record(
        RecordKind.PROMPT_ONLY,
        {"prompt": "held out"},
        split=Split.TEST,
    )
    with pytest.raises(DatasetSchemaError, match="split=train"):
        to_trl_example(record, for_training=True)


def test_d09_verifier_record_is_not_silently_treated_as_trainer_input() -> None:
    record = _record(RecordKind.VERIFIER_TASK, {"prompt": "x", "reference": "y"})
    with pytest.raises(DatasetSchemaError, match="not a TRL trainer input"):
        validate_trl_compatible_record(record)
