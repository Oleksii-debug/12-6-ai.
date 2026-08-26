"""Independent frozen-seed audit for POSTBASE-352 communication data.

This module does not train a model, authorize training, mutate the dataset, or call any
external model. It intentionally rechecks the frozen seed with an authority boundary
separate from ``data_contract.validate_dataset``.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from twelve_six.post_base.contract import TokenizerCompatibility
from twelve_six.post_base.data_contract import (
    BYTE_TOKENIZER_ID,
    BYTE_TOKENIZER_VOCAB_SIZE,
    DATASET_CLASSIFICATION,
    FORMATTER_ID,
    SOURCE_REGISTRY_ID,
    require_exact_base_tokenizer,
    validate_dataset,
)

AUDIT_SCHEMA = "12-6.post-base.communication-data-independent-audit.v1"
TARGET_POSTBASE352_HEAD = "d83fe9f7227112615da1f8f6e7a10f56531dbb35"
EXPECTED_MANIFEST_SHA256 = "51a927c40b4274f8b8f992b8dd83b4dbddac1e925a45834832a79ee6be18d3d6"
EXPECTED_SPLIT_SHA256 = {
    "train": "ddafe61ce3255dd30d207ec1ee811efa59a2da37288368a9bbc3fa0602cb2ba7",
    "selection": "e36f7c560c44fd2812935b5382dd628fabbd7af0e79c080c295d1332de13309f",
    "final": "f50262994089f276fb7d3f4c644180d0854273b51d7f7920aca1d2048031d039",
}
EXPECTED_RECORD_COUNTS = {"train": 12, "selection": 4, "final": 4}
EXPECTED_LANGUAGES = {"en", "uk"}
EXPECTED_NEAR_DUPLICATE_THRESHOLD = 0.85
EXPECTED_MAX_SFT_EXAMPLE_BYTES = 256
EXPECTED_TRAIN_SKILLS = {
    "direct_answer",
    "transformation",
    "summarization",
    "structured_response",
    "clarification",
    "exact_reasoning",
    "context_carryover",
}
EXPECTED_DATASET_FILES = {"manifest.json", "train.jsonl", "selection.jsonl", "final.jsonl"}
BASE_FIREWALL_FIELDS = (
    "base_corpus_evidence",
    "canonical_base_training_eligible",
    "training_authorized",
    "selection_for_training",
    "final_for_training",
    "final_for_selection",
)
_FORBIDDEN_REASONING_FIELDS = {
    "analysis",
    "reasoning",
    "chain_of_thought",
    "hidden_reasoning",
    "scratchpad",
}
_QUALITY_KEYS = {
    "answer_verified",
    "relevance_review",
    "language_review",
    "pii_review",
    "secret_review",
    "copyright_review",
    "no_hidden_reasoning",
}
_PROVENANCE_KEYS = {
    "origin_kind",
    "source_id",
    "rights",
    "foreign_model_output",
    "synthetic_authority_id",
    "content_sha256",
}


class CommunicationDatasetAuditError(ValueError):
    """Raised when the frozen POSTBASE-352 seed fails the independent audit."""


@dataclass(frozen=True, slots=True)
class IndependentCommunicationDatasetAudit:
    schema: str
    target_postbase352_head: str
    manifest_sha256: str
    split_sha256: Mapping[str, str]
    record_counts: Mapping[str, int]
    unique_family_count: int
    foreign_model_records: int
    languages_by_split: Mapping[str, tuple[str, ...]]
    max_near_duplicate_score: float
    near_duplicate_threshold: float
    answer_quality_reviewed_records: int
    formatting_reviewed_records: int
    exact_base_tokenizer_checked: bool
    base_training_firewall: bool
    dataset_mutated: bool = False
    training_authorized: bool = False
    training_executed: bool = False
    external_model_calls: int = 0


def _canon(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl_independent(path: Path) -> list[dict[str, object]]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CommunicationDatasetAuditError(f"{path.name} is not UTF-8") from exc
    if not text.endswith("\n"):
        raise CommunicationDatasetAuditError(f"{path.name} must end in LF")
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        if not line:
            raise CommunicationDatasetAuditError(f"{path.name} contains a blank JSONL line")
        row = json.loads(line)
        if not isinstance(row, dict) or line != _canon(row):
            raise CommunicationDatasetAuditError(f"{path.name} is not canonical JSONL")
        rows.append(row)
    return rows


def _scan_forbidden_reasoning_fields(value: object) -> None:
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_REASONING_FIELDS & set(value)
        if forbidden:
            raise CommunicationDatasetAuditError(
                f"hidden-reasoning field present: {sorted(forbidden)[0]}"
            )
        for nested in value.values():
            _scan_forbidden_reasoning_fields(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _scan_forbidden_reasoning_fields(nested)


def _language_script_counts(text: str) -> tuple[int, int]:
    latin = 0
    cyrillic = 0
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        latin += "LATIN" in name
        cyrillic += "CYRILLIC" in name
    return latin, cyrillic


def _check_language(language: str, messages: Sequence[Mapping[str, object]]) -> None:
    text = "\n".join(str(message["content"]) for message in messages)
    latin, cyrillic = _language_script_counts(text)
    if language == "en":
        if latin == 0 or cyrillic != 0:
            raise CommunicationDatasetAuditError("English row has incompatible script content")
        return
    if language == "uk":
        if cyrillic == 0 or cyrillic <= latin:
            raise CommunicationDatasetAuditError("Ukrainian row has incompatible script content")
        return
    raise CommunicationDatasetAuditError(f"unsupported language tag: {language}")


def _normalized_dialogue(row: Mapping[str, object]) -> str:
    messages = row["messages"]
    assert isinstance(messages, Sequence)
    parts: list[str] = []
    for message in messages:
        assert isinstance(message, Mapping)
        parts.append(f"{message['role']} {message['content']}")
    return " ".join(" ".join(parts).casefold().split())


def _shingles(text: str, width: int = 5) -> set[str]:
    if len(text) <= width:
        return {text}
    return {text[index : index + width] for index in range(len(text) - width + 1)}


def _max_near_duplicate_score(rows: Sequence[Mapping[str, object]]) -> float:
    maximum = 0.0
    for left_index, left in enumerate(rows):
        left_set = _shingles(_normalized_dialogue(left))
        for right in rows[left_index + 1 :]:
            right_set = _shingles(_normalized_dialogue(right))
            score = len(left_set & right_set) / len(left_set | right_set)
            maximum = max(maximum, score)
            if score >= EXPECTED_NEAR_DUPLICATE_THRESHOLD:
                raise CommunicationDatasetAuditError("near-duplicate rows violate frozen 0.85 gate")
    return maximum


def _audit_rows(rows_by_split: Mapping[str, Sequence[Mapping[str, object]]]) -> tuple[int, int]:
    record_ids: set[str] = set()
    family_ids: set[str] = set()
    foreign_count = 0

    for split, rows in rows_by_split.items():
        for row in rows:
            _scan_forbidden_reasoning_fields(row)
            if row.get("split") != split:
                raise CommunicationDatasetAuditError("row split does not match physical split")

            record_id = row.get("record_id")
            family_id = row.get("family_id")
            if not isinstance(record_id, str) or not record_id:
                raise CommunicationDatasetAuditError("record_id must be non-empty text")
            if not isinstance(family_id, str) or not family_id:
                raise CommunicationDatasetAuditError("family_id must be non-empty text")
            if record_id in record_ids:
                raise CommunicationDatasetAuditError("duplicate record_id")
            if family_id in family_ids:
                raise CommunicationDatasetAuditError("duplicate family_id")
            record_ids.add(record_id)
            family_ids.add(family_id)

            messages = row.get("messages")
            if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
                raise CommunicationDatasetAuditError("messages must be a sequence")
            if len(messages) < 2:
                raise CommunicationDatasetAuditError("dialogue must contain at least two messages")
            for index, message in enumerate(messages):
                if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
                    raise CommunicationDatasetAuditError("messages require exactly role/content")
                expected_role = "user" if index % 2 == 0 else "assistant"
                if message["role"] != expected_role:
                    raise CommunicationDatasetAuditError("roles do not alternate user/assistant")
                content = message["content"]
                if not isinstance(content, str) or not content.strip():
                    raise CommunicationDatasetAuditError("message content must be non-empty text")
                if content != content.strip():
                    raise CommunicationDatasetAuditError("message content has edge whitespace")
                for line in content.splitlines():
                    marker = line.lstrip().casefold()
                    if marker.startswith(("user:", "assistant:", "system:")):
                        raise CommunicationDatasetAuditError("message content injects a role prefix")
            if messages[-1]["role"] != "assistant":
                raise CommunicationDatasetAuditError("dialogue must end with assistant")

            language = row.get("language")
            if not isinstance(language, str):
                raise CommunicationDatasetAuditError("language must be text")
            _check_language(language, messages)

            quality = row.get("quality")
            if not isinstance(quality, Mapping) or set(quality) != _QUALITY_KEYS:
                raise CommunicationDatasetAuditError("quality schema drift")
            if quality["answer_verified"] is not True:
                raise CommunicationDatasetAuditError("answer quality is not verified")
            if quality["no_hidden_reasoning"] is not True:
                raise CommunicationDatasetAuditError("hidden-reasoning gate is not true")
            for key in (
                "relevance_review",
                "language_review",
                "pii_review",
                "secret_review",
                "copyright_review",
            ):
                if quality[key] != "pass":
                    raise CommunicationDatasetAuditError(f"quality.{key} is not pass")

            provenance = row.get("provenance")
            if not isinstance(provenance, Mapping) or set(provenance) != _PROVENANCE_KEYS:
                raise CommunicationDatasetAuditError("provenance schema drift")
            if (
                provenance["origin_kind"] != "project_authored"
                or provenance["source_id"] != SOURCE_REGISTRY_ID
                or provenance["rights"] != "project_owned"
                or provenance["foreign_model_output"] is not False
                or provenance["synthetic_authority_id"] is not None
            ):
                raise CommunicationDatasetAuditError("seed provenance is not project-owned")
            message_payload = [
                {"role": message["role"], "content": message["content"]} for message in messages
            ]
            content_sha = hashlib.sha256(_canon(message_payload).encode("utf-8")).hexdigest()
            if provenance["content_sha256"] != content_sha:
                raise CommunicationDatasetAuditError("provenance content SHA mismatch")
            foreign_count += int(provenance["foreign_model_output"] is True)

    return len(family_ids), foreign_count


def audit_postbase352_seed(
    root: Path,
    manifest_path: Path,
    *,
    expected_base_tokenizer: TokenizerCompatibility | None = None,
    candidate_base_tokenizer: TokenizerCompatibility | None = None,
) -> IndependentCommunicationDatasetAudit:
    """Audit the exact reviewed POSTBASE-352 seed without mutating it.

    Exact manifest/split hashes are the semantic-review authority. Any content change,
    including a coordinated answer/language edit with recomputed row and split hashes,
    requires a new audit identity rather than silently inheriting this PASS.
    """

    root = root.resolve()
    manifest_path_raw = manifest_path
    if manifest_path_raw.is_symlink():
        raise CommunicationDatasetAuditError("manifest must not be a symlink")
    if manifest_path_raw.resolve() != root / "manifest.json":
        raise CommunicationDatasetAuditError("audit requires the canonical manifest path")
    if not manifest_path_raw.is_file():
        raise CommunicationDatasetAuditError("manifest must be a regular file")

    actual_names = {path.name for path in root.iterdir()}
    if actual_names != EXPECTED_DATASET_FILES:
        raise CommunicationDatasetAuditError("dataset root file set drift")

    manifest_sha = _sha256(manifest_path_raw)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise CommunicationDatasetAuditError("frozen manifest identity mismatch")

    manifest = json.loads(manifest_path_raw.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise CommunicationDatasetAuditError("manifest must be an object")
    if manifest["classification"] != DATASET_CLASSIFICATION:
        raise CommunicationDatasetAuditError("dataset classification drift")
    if manifest["source_registry_id"] != SOURCE_REGISTRY_ID:
        raise CommunicationDatasetAuditError("source registry drift")
    if manifest["near_duplicate_threshold"] != EXPECTED_NEAR_DUPLICATE_THRESHOLD:
        raise CommunicationDatasetAuditError("near-duplicate threshold is not frozen at 0.85")
    if manifest["max_sft_example_bytes"] != EXPECTED_MAX_SFT_EXAMPLE_BYTES:
        raise CommunicationDatasetAuditError("SFT byte limit drift")
    if set(manifest["required_languages"]) != EXPECTED_LANGUAGES:
        raise CommunicationDatasetAuditError("required language set drift")
    if set(manifest["required_train_skills"]) != EXPECTED_TRAIN_SKILLS:
        raise CommunicationDatasetAuditError("required train skill set drift")
    if manifest["tokenizer_profile"] != {
        "tokenizer_id": BYTE_TOKENIZER_ID,
        "vocab_size": BYTE_TOKENIZER_VOCAB_SIZE,
        "encoding": "utf-8-bytes",
        "adds_special_tokens": False,
        "installs_base_chat_template": False,
        "exact_base_hash_binding_required_before_training": True,
    }:
        raise CommunicationDatasetAuditError("tokenizer profile drift")
    if manifest["formatter"] != {
        "formatter_id": FORMATTER_ID,
        "roles": ["user", "assistant"],
        "special_tokens": [],
    }:
        raise CommunicationDatasetAuditError("formatter drift")
    for field in BASE_FIREWALL_FIELDS:
        if manifest[field] is not False:
            raise CommunicationDatasetAuditError(f"Base-training firewall opened at {field}")

    core_audit = validate_dataset(root, manifest_path_raw)
    if dict(core_audit.record_counts) != EXPECTED_RECORD_COUNTS:
        raise CommunicationDatasetAuditError("record count drift")
    if dict(core_audit.split_sha256) != EXPECTED_SPLIT_SHA256:
        raise CommunicationDatasetAuditError("core split identity drift")
    if core_audit.foreign_model_records != 0:
        raise CommunicationDatasetAuditError("seed contains foreign-model output")

    rows_by_split: dict[str, list[dict[str, object]]] = {}
    file_identities: set[tuple[int, int]] = set()
    for split in ("train", "selection", "final"):
        path = root / f"{split}.jsonl"
        if path.is_symlink() or not path.is_file():
            raise CommunicationDatasetAuditError("split must be a regular non-symlink file")
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in file_identities:
            raise CommunicationDatasetAuditError("split files alias the same filesystem object")
        file_identities.add(identity)
        digest = _sha256(path)
        if digest != EXPECTED_SPLIT_SHA256[split]:
            raise CommunicationDatasetAuditError(f"{split} split identity mismatch")
        rows = _load_jsonl_independent(path)
        if len(rows) != EXPECTED_RECORD_COUNTS[split]:
            raise CommunicationDatasetAuditError(f"{split} record count mismatch")
        rows_by_split[split] = rows

    unique_family_count, foreign_count = _audit_rows(rows_by_split)
    all_rows = [row for split in ("train", "selection", "final") for row in rows_by_split[split]]
    max_near_duplicate_score = _max_near_duplicate_score(all_rows)

    languages_by_split: dict[str, tuple[str, ...]] = {}
    for split, rows in rows_by_split.items():
        languages = tuple(sorted({str(row["language"]) for row in rows}))
        if set(languages) != EXPECTED_LANGUAGES:
            raise CommunicationDatasetAuditError(f"{split} does not contain both frozen languages")
        languages_by_split[split] = languages

    train_skills = {str(row["skill"]) for row in rows_by_split["train"]}
    if not EXPECTED_TRAIN_SKILLS.issubset(train_skills):
        raise CommunicationDatasetAuditError("train skill coverage is incomplete")

    if (expected_base_tokenizer is None) != (candidate_base_tokenizer is None):
        raise CommunicationDatasetAuditError(
            "exact tokenizer audit requires both expected and candidate identities"
        )
    exact_base_tokenizer_checked = expected_base_tokenizer is not None
    if expected_base_tokenizer is not None and candidate_base_tokenizer is not None:
        try:
            require_exact_base_tokenizer(expected_base_tokenizer, candidate_base_tokenizer)
        except ValueError as exc:
            raise CommunicationDatasetAuditError("exact Base tokenizer identity mismatch") from exc

    return IndependentCommunicationDatasetAudit(
        schema=AUDIT_SCHEMA,
        target_postbase352_head=TARGET_POSTBASE352_HEAD,
        manifest_sha256=manifest_sha,
        split_sha256=dict(core_audit.split_sha256),
        record_counts=dict(core_audit.record_counts),
        unique_family_count=unique_family_count,
        foreign_model_records=foreign_count,
        languages_by_split=languages_by_split,
        max_near_duplicate_score=max_near_duplicate_score,
        near_duplicate_threshold=EXPECTED_NEAR_DUPLICATE_THRESHOLD,
        answer_quality_reviewed_records=len(all_rows),
        formatting_reviewed_records=len(all_rows),
        exact_base_tokenizer_checked=exact_base_tokenizer_checked,
        base_training_firewall=True,
    )
