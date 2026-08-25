from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twelve_six.data.ukrainian_normalization import (
    NORMALIZATION_SCHEMA,
    NormalizationError,
    normalize_document,
)

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
WORD_RE = re.compile(r"\w+", re.UNICODE)


class DataContractError(ValueError):
    """Raised when source/provenance/data invariants are violated."""


@dataclass(frozen=True)
class PipelineConfig:
    split_seed: str = "12-6-ai-s0-v1"
    validation_fraction: float = 0.17
    min_chars: int = 60
    max_chars: int = 4096
    min_alpha_ratio: float = 0.50
    near_duplicate_threshold: float = 0.92
    near_duplicate_shingle_words: int = 5
    tiny_near_dedup_max_documents: int = 5000

    def as_dict(self) -> dict[str, Any]:
        return {
            "split_seed": self.split_seed,
            "validation_fraction": self.validation_fraction,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "min_alpha_ratio": self.min_alpha_ratio,
            "near_duplicate_threshold": self.near_duplicate_threshold,
            "near_duplicate_shingle_words": self.near_duplicate_shingle_words,
            "tiny_near_dedup_max_documents": self.tiny_near_dedup_max_documents,
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str, *, modality: str = "natural") -> str:
    """D03 compatibility wrapper over the DATA-27 modality-aware normalizer."""
    if not isinstance(text, str):
        raise DataContractError("document text must be a string")
    try:
        return normalize_document(text, modality=modality).text  # type: ignore[arg-type]
    except (NormalizationError, TypeError) as exc:
        raise DataContractError(str(exc)) from exc


def language_id(text: str) -> str:
    latin = 0
    cyrillic = 0
    ukrainian_specific = 0
    for char in text:
        name = unicodedata.name(char, "")
        if "LATIN" in name and char.isalpha():
            latin += 1
        elif "CYRILLIC" in name and char.isalpha():
            cyrillic += 1
            if char.casefold() in {"і", "ї", "є", "ґ"}:
                ukrainian_specific += 1
    if cyrillic > latin and (ukrainian_specific > 0 or cyrillic >= 20):
        return "uk"
    if latin > cyrillic and latin >= 20:
        return "en"
    return "und"


def _quality_reason(text: str, config: PipelineConfig) -> str | None:
    if len(text) < config.min_chars:
        return "too_short"
    if len(text) > config.max_chars:
        return "too_long"
    if any(unicodedata.category(ch) == "Cc" and ch not in "\n\t" for ch in text):
        return "control_character"
    if EMAIL_RE.search(text):
        return "pii_email"
    if PHONE_RE.search(text):
        return "pii_phone"
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return "empty"
    alpha_ratio = sum(ch.isalpha() for ch in visible) / len(visible)
    if alpha_ratio < config.min_alpha_ratio:
        return "low_alpha_ratio"
    return None


def _word_shingles(text: str, width: int) -> frozenset[str]:
    words = [token.casefold() for token in WORD_RE.findall(text)]
    if len(words) < width:
        return frozenset({" ".join(words)}) if words else frozenset()
    return frozenset(" ".join(words[i : i + width]) for i in range(len(words) - width + 1))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _load_sources(source_registry_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    registry_bytes = source_registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    if registry.get("schema_version") != 1:
        raise DataContractError("unsupported source registry schema")
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise DataContractError("source registry must contain sources")
    return registry, sources


def _validate_source(source: dict[str, Any]) -> None:
    source_id = source.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise DataContractError("source_id is required")
    provenance = source.get("provenance")
    if not isinstance(provenance, dict):
        raise DataContractError(f"{source_id}: provenance is required")
    synthetic = provenance.get("synthetic")
    if synthetic not in {True, False}:
        raise DataContractError(f"{source_id}: synthetic provenance tag must be explicit")
    if synthetic and not provenance.get("synthetic_kind"):
        raise DataContractError(f"{source_id}: synthetic_kind is required for synthetic sources")
    purpose = source.get("purpose")
    if purpose in {"benchmark", "evaluation_test", "heldout_test"}:
        raise DataContractError(f"{source_id}: benchmark/evaluation sources cannot enter pretraining")
    license_meta = source.get("license")
    if not isinstance(license_meta, dict) or not license_meta.get("status"):
        raise DataContractError(f"{source_id}: license/provenance review status is required")


def build_dataset(
    source_registry_path: Path,
    contamination_registry_path: Path,
    output_dir: Path,
    *,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    config = config or PipelineConfig()
    if not 0.0 < config.validation_fraction < 1.0:
        raise DataContractError("validation_fraction must be between 0 and 1")

    registry_bytes = source_registry_path.read_bytes()
    source_registry, sources = _load_sources(source_registry_path)
    contamination_bytes = contamination_registry_path.read_bytes()
    contamination_registry = json.loads(contamination_bytes)
    if contamination_registry.get("schema_version") != 1:
        raise DataContractError("unsupported contamination registry schema")
    forbidden_hashes = set(contamination_registry.get("forbidden_normalized_sha256", []))
    forbidden_purposes = set(
        contamination_registry.get(
            "forbidden_source_purposes", ["benchmark", "evaluation_test", "heldout_test"]
        )
    )

    accepted: list[dict[str, Any]] = []
    normalization_reasons: Counter[str] = Counter()
    stats = {
        "input_documents": 0,
        "quality_rejected": 0,
        "contamination_rejected": 0,
        "exact_duplicates_removed": 0,
        "near_duplicates_removed": 0,
        "normalization_changed_documents": 0,
        "normalization_raw_codepoints": 0,
        "normalization_output_codepoints": 0,
        "normalization_raw_byte_tokens": 0,
        "normalization_output_byte_tokens": 0,
    }

    for source in sorted(sources, key=lambda item: item["source_id"]):
        _validate_source(source)
        if source.get("purpose") in forbidden_purposes:
            raise DataContractError(f"{source['source_id']}: forbidden source purpose")
        raw_rel = source.get("raw_path")
        if not isinstance(raw_rel, str):
            raise DataContractError(f"{source['source_id']}: raw_path is required")
        raw_path = source_registry_path.parent / raw_rel
        raw_bytes = raw_path.read_bytes()
        if _sha256_bytes(raw_bytes) != source.get("content_sha256"):
            raise DataContractError(f"{source['source_id']}: immutable source hash mismatch")
        provenance = source["provenance"]
        source_version = str(
            source.get("source_version") or source.get("version") or source["source_id"]
        )
        for line_number, line in enumerate(raw_bytes.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            stats["input_documents"] += 1
            record = json.loads(line)
            doc_id = record.get("document_id")
            if not isinstance(doc_id, str) or not doc_id:
                raise DataContractError(f"{source['source_id']}:{line_number}: document_id is required")
            modality = record.get("modality", "natural")
            try:
                normalization = normalize_document(
                    record.get("text"),
                    modality=modality,
                    source_id=source["source_id"],
                    source_version=source_version,
                    raw_document_id=doc_id,
                    raw_source_sha256=source.get("content_sha256"),
                )
            except (NormalizationError, TypeError) as exc:
                raise DataContractError(
                    f"{source['source_id']}:{doc_id}: normalization failed: {exc}"
                ) from exc
            normalized = normalization.text
            trace = normalization.trace
            normalization_reasons.update(trace.reason_counts)
            stats["normalization_raw_codepoints"] += trace.raw_codepoints
            stats["normalization_output_codepoints"] += trace.normalized_codepoints
            stats["normalization_raw_byte_tokens"] += trace.raw_utf8_bytes
            stats["normalization_output_byte_tokens"] += trace.normalized_utf8_bytes
            if trace.raw_text_sha256 != trace.normalized_text_sha256:
                stats["normalization_changed_documents"] += 1

            reason = _quality_reason(normalized, config)
            if reason is not None:
                stats["quality_rejected"] += 1
                continue
            detected_language = language_id(normalized)
            hint = record.get("language_hint")
            if hint and detected_language not in {hint, "und"}:
                raise DataContractError(
                    f"{source['source_id']}:{doc_id}: language mismatch hint={hint} detected={detected_language}"
                )
            content_sha = trace.normalized_text_sha256
            if content_sha in forbidden_hashes:
                stats["contamination_rejected"] += 1
                continue
            accepted.append(
                {
                    "id": f"{source['source_id']}::{doc_id}",
                    "text": normalized,
                    "language": detected_language if detected_language != "und" else hint or "und",
                    "source_id": source["source_id"],
                    "content_sha256": content_sha,
                    "raw_text_sha256": trace.raw_text_sha256,
                    "normalization": trace.as_dict(),
                    "synthetic": provenance["synthetic"],
                    "synthetic_kind": provenance.get("synthetic_kind"),
                }
            )

    stats["normalization_codepoint_delta"] = (
        stats["normalization_output_codepoints"] - stats["normalization_raw_codepoints"]
    )
    stats["normalization_byte_token_delta"] = (
        stats["normalization_output_byte_tokens"] - stats["normalization_raw_byte_tokens"]
    )
    stats["normalization_reason_counts"] = dict(sorted(normalization_reasons.items()))

    exact_seen: set[str] = set()
    exact_deduped: list[dict[str, Any]] = []
    for record in sorted(accepted, key=lambda item: item["id"]):
        if record["content_sha256"] in exact_seen:
            stats["exact_duplicates_removed"] += 1
            continue
        exact_seen.add(record["content_sha256"])
        exact_deduped.append(record)

    if len(exact_deduped) > config.tiny_near_dedup_max_documents:
        raise DataContractError(
            "tiny near-dedup implementation limit exceeded; use DataTrove/MinHash backend"
        )
    near_deduped: list[dict[str, Any]] = []
    shingles: list[frozenset[str]] = []
    for record in exact_deduped:
        candidate = _word_shingles(record["text"], config.near_duplicate_shingle_words)
        if any(_jaccard(candidate, prior) >= config.near_duplicate_threshold for prior in shingles):
            stats["near_duplicates_removed"] += 1
            continue
        shingles.append(candidate)
        near_deduped.append(record)

    if len(near_deduped) < 2:
        raise DataContractError("dataset must contain at least two accepted documents")
    validation_count = max(1, round(len(near_deduped) * config.validation_fraction))
    validation_count = min(validation_count, len(near_deduped) - 1)
    ranked = sorted(
        near_deduped,
        key=lambda item: hashlib.sha256(
            f"{config.split_seed}\0{item['id']}".encode()
        ).hexdigest(),
    )
    validation_ids = {item["id"] for item in ranked[:validation_count]}
    train = sorted((item for item in near_deduped if item["id"] not in validation_ids), key=lambda x: x["id"])
    validation = sorted((item for item in near_deduped if item["id"] in validation_ids), key=lambda x: x["id"])

    output_dir.mkdir(parents=True, exist_ok=True)
    train_bytes = b"".join(_canonical_json_bytes(item) for item in train)
    validation_bytes = b"".join(_canonical_json_bytes(item) for item in validation)
    (output_dir / "train.jsonl").write_bytes(train_bytes)
    (output_dir / "validation.jsonl").write_bytes(validation_bytes)

    stats["accepted_documents"] = len(near_deduped)
    stats["train_documents"] = len(train)
    stats["validation_documents"] = len(validation)
    stats["accepted_utf8_bytes"] = sum(len(item["text"].encode("utf-8")) for item in near_deduped)
    stats["train_text_utf8_bytes"] = sum(len(item["text"].encode("utf-8")) for item in train)
    stats["validation_text_utf8_bytes"] = sum(
        len(item["text"].encode("utf-8")) for item in validation
    )

    identity_core = {
        "schema_version": 1,
        "dataset_id": source_registry["dataset_id"],
        "normalization_schema": NORMALIZATION_SCHEMA,
        "source_registry_sha256": _sha256_bytes(registry_bytes),
        "contamination_registry_sha256": _sha256_bytes(contamination_bytes),
        "pipeline_config": config.as_dict(),
        "document_assignments": [
            {
                "id": item["id"],
                "raw_text_sha256": item["raw_text_sha256"],
                "content_sha256": item["content_sha256"],
                "split": "validation" if item["id"] in validation_ids else "train",
            }
            for item in sorted(near_deduped, key=lambda x: x["id"])
        ],
        "outputs": {
            "train.jsonl": _sha256_bytes(train_bytes),
            "validation.jsonl": _sha256_bytes(validation_bytes),
        },
    }
    manifest = {
        **identity_core,
        "dataset_identity_sha256": _sha256_bytes(_canonical_json_bytes(identity_core)),
        "stats": stats,
        "dedup": {
            "exact": "normalized_text_sha256",
            "near": "tiny_word_shingle_jaccard_v1",
            "near_threshold": config.near_duplicate_threshold,
            "scale_backend_required_above_documents": config.tiny_near_dedup_max_documents,
            "recommended_scale_backend": "DataTrove MinHash/dedup",
        },
        "contamination_state": {
            "policy": "reject_forbidden_source_purpose_and_registered_normalized_hash",
            "claim": "controlled S0 sources only; not a universal benchmark-clean claim",
        },
        "license_provenance_state": [
            {
                "source_id": source["source_id"],
                "license": source["license"],
                "provenance": source["provenance"],
            }
            for source in sorted(sources, key=lambda item: item["source_id"])
        ],
    }
    (output_dir / "manifest.json").write_bytes(_canonical_json_bytes(manifest))
    return manifest


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the deterministic S0 D03 dataset package.")
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--contamination-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = build_dataset(args.source_registry, args.contamination_registry, args.output_dir)
    print(json.dumps(manifest["stats"], sort_keys=True))
    print(f"dataset_identity_sha256={manifest['dataset_identity_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())