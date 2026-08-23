"""Dependency-light validation/adaptation for mature post-training dataset formats.

The functions in this module intentionally do not import TRL. They validate the
stable 12-6 record contract against the task shapes expected by current TRL-style
trainers, so a future thin runtime adapter can remain replaceable and auditable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .contracts import DatasetRecord, RecordKind, Split


class DatasetSchemaError(ValueError):
    """Raised when a post-training record does not match its declared task kind."""


TRL_TRAINER_RECORD_KINDS: Mapping[str, tuple[RecordKind, ...]] = {
    "DPOTrainer": (RecordKind.PREFERENCE,),
    "GRPOTrainer": (RecordKind.PROMPT_ONLY,),
    "PRMTrainer": (RecordKind.STEPWISE_SUPERVISION,),
    "RewardTrainer": (RecordKind.PREFERENCE,),
    "SFTTrainer": (RecordKind.LANGUAGE_MODELING, RecordKind.PROMPT_COMPLETION),
}


def _require(payload: Mapping[str, object], *keys: str) -> None:
    missing = tuple(key for key in keys if key not in payload)
    if missing:
        raise DatasetSchemaError(f"missing required payload keys: {', '.join(missing)}")


def _validate_messages(value: object, field_name: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise DatasetSchemaError(f"{field_name} must be a non-empty message sequence")
    for message in value:
        if not isinstance(message, Mapping):
            raise DatasetSchemaError(f"{field_name} messages must be mappings")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise DatasetSchemaError(f"{field_name} message role must be non-empty text")
        if not isinstance(content, str) or not content:
            raise DatasetSchemaError(f"{field_name} message content must be non-empty text")


def _validate_text_or_messages(value: object, field_name: str) -> None:
    if isinstance(value, str):
        if not value:
            raise DatasetSchemaError(f"{field_name} must be non-empty")
        return
    _validate_messages(value, field_name)


def validate_trl_compatible_record(record: DatasetRecord) -> None:
    """Validate task shape without importing or invoking TRL.

    The supported shapes mirror current TRL dataset categories: language modeling,
    prompt-only, prompt-completion, paired preference, unpaired preference, and
    stepwise supervision. D09-specific verifier/candidate records are intentionally
    not silently treated as trainer inputs.
    """

    payload = record.payload
    kind = record.kind

    if kind is RecordKind.LANGUAGE_MODELING:
        has_text = "text" in payload
        has_messages = "messages" in payload
        if has_text == has_messages:
            raise DatasetSchemaError(
                "language-modeling record requires exactly one of text or messages"
            )
        if has_text:
            _validate_text_or_messages(payload["text"], "text")
        else:
            _validate_messages(payload["messages"], "messages")
        return

    if kind is RecordKind.PROMPT_ONLY:
        _require(payload, "prompt")
        _validate_text_or_messages(payload["prompt"], "prompt")
        return

    if kind is RecordKind.PROMPT_COMPLETION:
        _require(payload, "prompt", "completion")
        _validate_text_or_messages(payload["prompt"], "prompt")
        _validate_text_or_messages(payload["completion"], "completion")
        return

    if kind is RecordKind.PREFERENCE:
        _require(payload, "chosen", "rejected")
        if "prompt" in payload:
            _validate_text_or_messages(payload["prompt"], "prompt")
        _validate_text_or_messages(payload["chosen"], "chosen")
        _validate_text_or_messages(payload["rejected"], "rejected")
        if payload["chosen"] == payload["rejected"]:
            raise DatasetSchemaError("chosen and rejected preference values must differ")
        return

    if kind is RecordKind.UNPAIRED_PREFERENCE:
        _require(payload, "prompt", "completion", "label")
        _validate_text_or_messages(payload["prompt"], "prompt")
        _validate_text_or_messages(payload["completion"], "completion")
        if not isinstance(payload["label"], bool):
            raise DatasetSchemaError("unpaired preference label must be boolean")
        return

    if kind is RecordKind.STEPWISE_SUPERVISION:
        _require(payload, "prompt", "completions", "labels")
        _validate_text_or_messages(payload["prompt"], "prompt")
        completions = payload["completions"]
        labels = payload["labels"]
        if (
            not isinstance(completions, Sequence)
            or isinstance(completions, (str, bytes))
            or not completions
        ):
            raise DatasetSchemaError("stepwise completions must be a non-empty sequence")
        if not all(isinstance(item, str) and item for item in completions):
            raise DatasetSchemaError("stepwise completions must contain non-empty text")
        if (
            not isinstance(labels, Sequence)
            or isinstance(labels, (str, bytes))
            or len(labels) != len(completions)
        ):
            raise DatasetSchemaError("stepwise labels must align one-to-one with completions")
        if not all(isinstance(label, bool) for label in labels):
            raise DatasetSchemaError("stepwise labels must be boolean")
        return

    raise DatasetSchemaError(f"record kind is not a TRL trainer input: {kind.value}")


def to_trl_example(record: DatasetRecord, *, for_training: bool = False) -> dict[str, object]:
    """Return only trainer payload fields after fail-closed validation.

    ``for_training=True`` additionally requires the immutable record split to be
    TRAIN. This prevents validation/test records from becoming training rows merely
    because they can be serialized into a trainer-compatible shape.
    """

    validate_trl_compatible_record(record)
    if for_training and record.split is not Split.TRAIN:
        raise DatasetSchemaError(
            f"training conversion requires split=train, got split={record.split.value}"
        )
    return dict(record.payload)
