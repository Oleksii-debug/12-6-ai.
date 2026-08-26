"""Fail-closed contract for post-Base user/assistant communication data.

No training, Base-corpus admission, tokenizer mutation, or external model call occurs here.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from twelve_six.post_base.contract import TokenizerCompatibility
from twelve_six.posttraining.contracts import DatasetRecord, RecordKind, Split, SyntheticProvenance

DATASET_SCHEMA = "12-6.post-base.communication-dataset.v1"
DATASET_CLASSIFICATION = "POSTBASE_COMMUNICATION_ONLY"
SOURCE_REGISTRY_ID = "project:postbase352-manual-v1"
BYTE_TOKENIZER_ID = "s0-byte-v1"
BYTE_TOKENIZER_VOCAB_SIZE = 256
FORMATTER_ID = "postbase352.plain-role-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CommunicationDataError(ValueError):
    pass


class CommunicationSplit(StrEnum):
    TRAIN = "train"
    SELECTION = "selection"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class CommunicationRecord:
    record_id: str
    family_id: str
    split: CommunicationSplit
    language: str
    skill: str
    messages: tuple[tuple[str, str], ...]
    source_id: str
    rights: str
    foreign_model_output: bool
    synthetic_authority_id: str | None
    content_sha256: str


@dataclass(frozen=True, slots=True)
class SyntheticDataAuthority:
    authority_id: str
    authority_sha256: str
    allowed_source_ids: tuple[str, ...]
    purpose: str = "post_base_communication_data"
    owner_approved: bool = False

    def __post_init__(self) -> None:
        _text(self.authority_id, "authority_id")
        _sha(self.authority_sha256, "authority_sha256")
        if self.purpose != "post_base_communication_data" or not self.owner_approved:
            raise CommunicationDataError("synthetic authority must be owner-approved for communication data")
        if not self.allowed_source_ids:
            raise CommunicationDataError("synthetic authority must name allowed source IDs")


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    dataset_id: str
    manifest_sha256: str
    split_sha256: Mapping[str, str]
    record_counts: Mapping[str, int]
    foreign_model_records: int
    max_sft_example_bytes_observed: int


def _canon(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: str, field: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise CommunicationDataError(f"{field} must be lowercase 64-hex SHA-256")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CommunicationDataError(f"{field} must be non-empty text")
    if value != unicodedata.normalize("NFC", value) or "\r" in value or "\x00" in value:
        raise CommunicationDataError(f"{field} must be NFC LF-only text without NUL")
    if any(unicodedata.category(ch) == "Cc" and ch != "\n" for ch in value):
        raise CommunicationDataError(f"{field} contains a forbidden control character")
    return value


def _parse_record(row: Mapping[str, object]) -> CommunicationRecord:
    keys = {"record_id", "family_id", "split", "language", "skill", "messages", "provenance", "quality"}
    if set(row) != keys:
        raise CommunicationDataError("record keys do not exactly match v1")
    try:
        split = CommunicationSplit(str(row["split"]))
    except ValueError as exc:
        raise CommunicationDataError("split must be train, selection, or final") from exc
    raw_messages = row["messages"]
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
        raise CommunicationDataError("messages must be a sequence")
    messages: list[tuple[str, str]] = []
    for index, message in enumerate(raw_messages):
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise CommunicationDataError("messages require exactly role/content")
        role = _text(message["role"], f"messages[{index}].role")
        content = _text(message["content"], f"messages[{index}].content")
        if role not in {"user", "assistant"}:
            raise CommunicationDataError("only user/assistant roles are permitted")
        messages.append((role, content))
    if len(messages) < 2 or messages[0][0] != "user" or messages[-1][0] != "assistant":
        raise CommunicationDataError("dialogue must start with user and end with assistant")
    for index, (role, _) in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if role != expected:
            raise CommunicationDataError("roles must alternate user/assistant")

    provenance = row["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "origin_kind", "source_id", "rights", "foreign_model_output", "synthetic_authority_id", "content_sha256"
    }:
        raise CommunicationDataError("provenance keys do not exactly match v1")
    foreign = provenance["foreign_model_output"]
    if not isinstance(foreign, bool):
        raise CommunicationDataError("foreign_model_output must be boolean")
    origin = _text(provenance["origin_kind"], "origin_kind")
    if origin not in {"project_authored", "foreign_model_output"}:
        raise CommunicationDataError("unsupported origin_kind")
    if foreign != (origin == "foreign_model_output"):
        raise CommunicationDataError("origin_kind/foreign_model_output mismatch")
    authority_id = provenance["synthetic_authority_id"]
    if authority_id is not None:
        authority_id = _text(authority_id, "synthetic_authority_id")
    content_sha = _sha(str(provenance["content_sha256"]), "content_sha256")
    message_payload = [{"role": role, "content": content} for role, content in messages]
    if content_sha != hashlib.sha256(_canon(message_payload).encode()).hexdigest():
        raise CommunicationDataError("content_sha256 does not match messages")

    quality = row["quality"]
    if not isinstance(quality, Mapping) or set(quality) != {
        "answer_verified", "relevance_review", "language_review", "pii_review", "secret_review", "copyright_review", "no_hidden_reasoning"
    }:
        raise CommunicationDataError("quality keys do not exactly match v1")
    if quality["answer_verified"] is not True or quality["no_hidden_reasoning"] is not True:
        raise CommunicationDataError("answer verification and no-hidden-reasoning gates must pass")
    for name in ("relevance_review", "language_review", "pii_review", "secret_review", "copyright_review"):
        if quality[name] != "pass":
            raise CommunicationDataError(f"quality.{name} must be pass")

    return CommunicationRecord(
        record_id=_text(row["record_id"], "record_id"),
        family_id=_text(row["family_id"], "family_id"),
        split=split,
        language=_text(row["language"], "language"),
        skill=_text(row["skill"], "skill"),
        messages=tuple(messages),
        source_id=_text(provenance["source_id"], "source_id"),
        rights=_text(provenance["rights"], "rights"),
        foreign_model_output=foreign,
        synthetic_authority_id=authority_id,
        content_sha256=content_sha,
    )


def _validate_provenance(record: CommunicationRecord, authority: SyntheticDataAuthority | None) -> None:
    if not record.foreign_model_output:
        if record.source_id != SOURCE_REGISTRY_ID or record.rights != "project_owned" or record.synthetic_authority_id:
            raise CommunicationDataError("project-authored rows require project-owned provenance")
        return
    if not record.synthetic_authority_id or authority is None:
        raise CommunicationDataError("foreign model output requires explicit later synthetic-data authority")
    if record.synthetic_authority_id != authority.authority_id or record.source_id not in authority.allowed_source_ids:
        raise CommunicationDataError("foreign model output is outside the active synthetic authority")


def format_sft_turns(record: CommunicationRecord) -> tuple[tuple[str, str], ...]:
    history: list[str] = []
    result: list[tuple[str, str]] = []
    for role, content in record.messages:
        if role == "user":
            history.append(f"User: {content}")
        else:
            result.append(("\n".join(history) + "\nAssistant:", content))
            history.append(f"Assistant: {content}")
    return tuple(result)


def require_logical_tokenizer_compatibility(tokenizer: TokenizerCompatibility) -> None:
    if tokenizer.tokenizer_id != BYTE_TOKENIZER_ID or tokenizer.vocab_size != BYTE_TOKENIZER_VOCAB_SIZE:
        raise CommunicationDataError("v1 requires the s0-byte-v1 256-byte tokenizer profile")


def require_exact_base_tokenizer(expected: TokenizerCompatibility, candidate: TokenizerCompatibility) -> None:
    require_logical_tokenizer_compatibility(expected)
    expected.require_exact_match(candidate)


def to_posttraining_records(
    record: CommunicationRecord,
    *,
    for_training: bool = False,
    synthetic_authority: SyntheticDataAuthority | None = None,
) -> tuple[DatasetRecord, ...]:
    _validate_provenance(record, synthetic_authority)
    if for_training and record.split is not CommunicationSplit.TRAIN:
        raise CommunicationDataError("only train split may be converted for training")
    split_map = {CommunicationSplit.TRAIN: Split.TRAIN, CommunicationSplit.SELECTION: Split.VALIDATION, CommunicationSplit.FINAL: Split.TEST}
    rows: list[DatasetRecord] = []
    for turn, (prompt, completion) in enumerate(format_sft_turns(record), 1):
        digest = hashlib.sha256(_canon({"prompt": prompt, "completion": completion}).encode()).hexdigest()
        provenance = SyntheticProvenance(
            source_id=record.source_id,
            content_sha256=digest,
            synthetic=record.foreign_model_output,
            generator_id=record.source_id if record.foreign_model_output else None,
            external_generator=record.foreign_model_output,
            owner_policy_ref=record.synthetic_authority_id,
            parent_sha256=(record.content_sha256,),
            metadata={"classification": DATASET_CLASSIFICATION, "family_id": record.family_id, "language": record.language, "skill": record.skill, "formatter_id": FORMATTER_ID, "base_corpus_evidence": "false", "canonical_base_training_eligible": "false"},
        )
        rows.append(DatasetRecord(record_id=f"{record.record_id}.turn-{turn}", kind=RecordKind.PROMPT_COMPLETION, split=split_map[record.split], payload={"prompt": prompt, "completion": completion}, provenance=provenance))
    return tuple(rows)


def _load_jsonl(path: Path, expected_split: CommunicationSplit) -> tuple[CommunicationRecord, ...]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CommunicationDataError("split files must be UTF-8") from exc
    if not text.endswith("\n"):
        raise CommunicationDataError("canonical JSONL must end with LF")
    records: list[CommunicationRecord] = []
    for line in text.splitlines():
        if not line:
            raise CommunicationDataError("blank JSONL lines are forbidden")
        row = json.loads(line)
        if not isinstance(row, Mapping) or line != _canon(row):
            raise CommunicationDataError("each row must be canonical JSON")
        record = _parse_record(row)
        if record.split is not expected_split:
            raise CommunicationDataError("row split does not match split file")
        records.append(record)
    if not records:
        raise CommunicationDataError("splits must be non-empty")
    return tuple(records)


def _normalized(record: CommunicationRecord) -> str:
    return " ".join(" ".join(f"{role} {content}" for role, content in record.messages).casefold().split())


def _shingles(text: str, width: int = 5) -> set[str]:
    return {text} if len(text) <= width else {text[i:i + width] for i in range(len(text) - width + 1)}


def _validate_split_isolation(records: Sequence[CommunicationRecord], threshold: float) -> None:
    ids: set[str] = set()
    families: dict[str, CommunicationSplit] = {}
    exact: dict[str, str] = {}
    for record in records:
        if record.record_id in ids:
            raise CommunicationDataError("duplicate record_id")
        ids.add(record.record_id)
        prior = families.setdefault(record.family_id, record.split)
        if prior is not record.split:
            raise CommunicationDataError("family_id crosses splits")
        digest = hashlib.sha256(_normalized(record).encode()).hexdigest()
        if digest in exact:
            raise CommunicationDataError("exact normalized duplicate")
        exact[digest] = record.record_id
    for index, left in enumerate(records):
        left_set = _shingles(_normalized(left))
        for right in records[index + 1:]:
            if left.split is right.split:
                continue
            right_set = _shingles(_normalized(right))
            score = len(left_set & right_set) / len(left_set | right_set)
            if score >= threshold:
                raise CommunicationDataError("cross-split near duplicate")


def validate_dataset(root: Path, manifest_path: Path, *, base_tokenizer: TokenizerCompatibility | None = None, synthetic_authority: SyntheticDataAuthority | None = None) -> DatasetAudit:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, Mapping) or manifest_bytes != (_canon(manifest) + "\n").encode():
        raise CommunicationDataError("manifest must be canonical UTF-8 JSON plus final LF")
    required = {"schema", "dataset_id", "classification", "base_corpus_evidence", "canonical_base_training_eligible", "training_authorized", "source_registry_id", "foreign_model_records", "synthetic_data_authority", "tokenizer_profile", "formatter", "split_files", "split_sha256", "record_counts", "required_train_skills", "required_languages", "near_duplicate_threshold", "max_sft_example_bytes", "selection_for_training", "final_for_training", "final_for_selection"}
    if set(manifest) != required or manifest["schema"] != DATASET_SCHEMA:
        raise CommunicationDataError("manifest schema/keys drift")
    if manifest["classification"] != DATASET_CLASSIFICATION or manifest["source_registry_id"] != SOURCE_REGISTRY_ID:
        raise CommunicationDataError("post-Base classification/provenance drift")
    for name in ("base_corpus_evidence", "canonical_base_training_eligible", "training_authorized", "selection_for_training", "final_for_training", "final_for_selection"):
        if manifest[name] is not False:
            raise CommunicationDataError(f"manifest.{name} must remain false")
    if manifest["synthetic_data_authority"] is not None:
        raise CommunicationDataError("seed manifest carries no synthetic-data authority")
    expected_profile = {"tokenizer_id": BYTE_TOKENIZER_ID, "vocab_size": 256, "encoding": "utf-8-bytes", "adds_special_tokens": False, "installs_base_chat_template": False, "exact_base_hash_binding_required_before_training": True}
    if manifest["tokenizer_profile"] != expected_profile:
        raise CommunicationDataError("tokenizer profile drift")
    if manifest["formatter"] != {"formatter_id": FORMATTER_ID, "roles": ["user", "assistant"], "special_tokens": []}:
        raise CommunicationDataError("formatter drift")
    if base_tokenizer is not None:
        require_logical_tokenizer_compatibility(base_tokenizer)

    all_records: list[CommunicationRecord] = []
    split_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    max_observed = 0
    max_bytes = manifest["max_sft_example_bytes"]
    threshold = float(manifest["near_duplicate_threshold"])
    if not isinstance(max_bytes, int) or max_bytes <= 0 or not 0 < threshold <= 1:
        raise CommunicationDataError("invalid byte/near-duplicate gate")
    for split in CommunicationSplit:
        relative = manifest["split_files"].get(split.value)
        path = (root / relative).resolve() if isinstance(relative, str) else root
        if root.resolve() not in path.parents:
            raise CommunicationDataError("split path escapes dataset root")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != manifest["split_sha256"].get(split.value):
            raise CommunicationDataError("split SHA-256 mismatch")
        records = _load_jsonl(path, split)
        if len(records) != manifest["record_counts"].get(split.value):
            raise CommunicationDataError("split record count mismatch")
        split_hashes[split.value] = digest
        counts[split.value] = len(records)
        for record in records:
            _validate_provenance(record, synthetic_authority)
            for prompt, completion in format_sft_turns(record):
                observed = len((prompt + completion).encode("utf-8"))
                max_observed = max(max_observed, observed)
                if observed > max_bytes:
                    raise CommunicationDataError("SFT example exceeds byte-tokenizer/context contract")
        all_records.extend(records)
    _validate_split_isolation(all_records, threshold)

    train_skills = {r.skill for r in all_records if r.split is CommunicationSplit.TRAIN}
    if set(manifest["required_train_skills"]) - train_skills:
        raise CommunicationDataError("required train skill coverage is incomplete")
    for split in CommunicationSplit:
        languages = {r.language for r in all_records if r.split is split}
        if set(manifest["required_languages"]) - languages:
            raise CommunicationDataError("required language coverage is incomplete")
    foreign_count = sum(r.foreign_model_output for r in all_records)
    if foreign_count != manifest["foreign_model_records"] or (foreign_count and synthetic_authority is None):
        raise CommunicationDataError("foreign-model provenance gate failed")
    return DatasetAudit(_text(manifest["dataset_id"], "dataset_id"), hashlib.sha256(manifest_bytes).hexdigest(), split_hashes, counts, foreign_count, max_observed)
