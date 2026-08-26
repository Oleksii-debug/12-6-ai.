"""Executable pretraining data factory composed over D03/D09 provenance contracts.

Rights approval, byte acquisition, corpus processing, and tokenizer selection remain
separate authorities. Production near deduplication and Parquet materialization are
delegated to DataTrove 0.10.0 when that optional runtime is present.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .corpus_foundation import SQLiteExactDedupIndex
from .external_sources import (
    validate_external_source_registry,
    validate_reserved_fingerprint_registry,
)
from .pipeline import EMAIL_RE, PHONE_RE, language_id, normalize_text

FACTORY_PLAN_SCHEMA = "12-6.pretraining-data-factory-plan.v1"
FACTORY_STAGE_SCHEMA = "12-6.pretraining-data-factory-stage.v1"
TOKEN_TARGET_SCHEMA = "12-6.pretraining-token-targets.v1"
SUPPORTED_SOURCE_KINDS = frozenset({"jsonl", "jsonl_text_v1"})
_ALLOWED_URI_SCHEMES = frozenset({"file", "s3", "gs", "hf", "az", "r2"})


class PretrainingFactoryError(ValueError):
    """Raised when factory provenance, execution, or artifact invariants fail."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PretrainingFactoryError(f"{field} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field)
    valid = len(text) == 64 and text == text.lower()
    if not valid or any(char not in "0123456789abcdef" for char in text):
        raise PretrainingFactoryError(f"{field} must be lowercase SHA-256 hex")
    return text


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PretrainingFactoryError(f"{field} must be a positive integer")
    return value


def _validate_stable_uri(uri: str, field: str) -> str:
    text = _require_text(uri, field)
    parsed = urlsplit(text)
    if parsed.scheme not in _ALLOWED_URI_SCHEMES:
        raise PretrainingFactoryError(f"{field} uses unsupported URI scheme {parsed.scheme!r}")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise PretrainingFactoryError(f"{field} must be stable and credential-free")
    return text


def _local_path(uri: str, field: str) -> Path:
    parsed = urlsplit(_validate_stable_uri(uri, field))
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise PretrainingFactoryError(
            f"{field} must be local file:// URI for LOCAL_FREE execution"
        )
    return Path(unquote(parsed.path)).resolve()


@dataclass(frozen=True)
class FactoryPlan:
    """Content-addressed execution plan for one reviewed corpus slice."""

    source_registry_sha256: str
    retrieval_inventory_sha256: str
    reserved_registry_sha256: str
    output_uri: str
    split_seed: str = "12-6-pretraining-v1"
    validation_per_10k: int = 100
    shard_count: int = 16
    max_records_in_memory: int = 1024
    min_chars: int = 40
    max_chars: int = 200_000
    min_alpha_ratio: float = 0.20
    allowed_languages: tuple[str, ...] = ("en", "uk")
    datatrove_version: str = "0.10.0"
    minhash_num_buckets: int = 14
    minhash_hashes_per_bucket: int = 8
    minhash_n_grams: int = 5
    local_fixture_near_dedup_limit: int = 5_000

    def __post_init__(self) -> None:
        for field in (
            "source_registry_sha256",
            "retrieval_inventory_sha256",
            "reserved_registry_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        _validate_stable_uri(self.output_uri, "output_uri")
        _require_text(self.split_seed, "split_seed")
        for field in (
            "shard_count",
            "max_records_in_memory",
            "min_chars",
            "max_chars",
            "minhash_num_buckets",
            "minhash_hashes_per_bucket",
            "minhash_n_grams",
            "local_fixture_near_dedup_limit",
        ):
            _require_positive_int(getattr(self, field), field)
        if not 1 <= self.validation_per_10k <= 5_000:
            raise PretrainingFactoryError("validation_per_10k must be in [1, 5000]")
        if self.max_chars <= self.min_chars:
            raise PretrainingFactoryError("max_chars must exceed min_chars")
        if not 0.0 <= self.min_alpha_ratio <= 1.0:
            raise PretrainingFactoryError("min_alpha_ratio must be in [0,1]")
        if not self.allowed_languages or any(not item for item in self.allowed_languages):
            raise PretrainingFactoryError("allowed_languages must be non-empty")
        if self.datatrove_version != "0.10.0":
            raise PretrainingFactoryError("DataTrove must remain 0.10.0 until revalidated")

    def manifest(self) -> dict[str, Any]:
        core = {"schema_version": FACTORY_PLAN_SCHEMA, **asdict(self)}
        core["allowed_languages"] = list(self.allowed_languages)
        return {**core, "plan_sha256": _sha256_bytes(_canonical_json_bytes(core))}


@dataclass(frozen=True)
class ProcessedRecord:
    id: str
    text: str
    language: str
    source_id: str
    source_version: str
    content_sha256: str

    def as_mapping(self) -> dict[str, str]:
        return asdict(self)


def build_token_targets(parameter_count: int) -> dict[str, Any]:
    """Return planning targets without pretending tokenizer choice is frozen."""

    params = _require_positive_int(parameter_count, "parameter_count")
    tiers = {"mechanics_gate": 2, "serious_ablation": 10, "scratch_baseline": 20}
    targets = {name: params * ratio for name, ratio in tiers.items()}
    return {
        "schema_version": TOKEN_TARGET_SCHEMA,
        "parameter_count": params,
        "targets_in_selected_experiment_tokenizer_tokens": targets,
        "raw_candidate_tokens_at_70pct_yield": {
            name: math.ceil(tokens / 0.70) for name, tokens in targets.items()
        },
        "canonical_tokenizer_frozen": False,
        "note": "S0 byte-token counts are compatibility evidence, not future corpus authority.",
    }


def _validate_retrieval_inventory(
    source_registry: Mapping[str, Any],
    retrieval_inventory: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    registry_identity = _require_sha256(
        source_registry.get("registry_identity_sha256"), "registry_identity_sha256"
    )
    if retrieval_inventory.get("source_registry_identity_sha256") != registry_identity:
        raise PretrainingFactoryError("retrieval inventory uses another source registry")
    core = dict(retrieval_inventory)
    claimed = core.pop("inventory_sha256", None)
    if claimed != _sha256_bytes(_canonical_json_bytes(core)):
        raise PretrainingFactoryError("retrieval inventory identity/content mismatch")
    receipts = retrieval_inventory.get("receipts")
    if not isinstance(receipts, list) or not receipts:
        raise PretrainingFactoryError("retrieval inventory must contain verified receipts")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise PretrainingFactoryError("retrieval receipt must be an object")
        if receipt.get("verification") != "PASS":
            raise PretrainingFactoryError("retrieval receipt is not verified PASS")
        if receipt.get("training_eligibility_evaluated") is not False:
            raise PretrainingFactoryError("retrieval receipt must not impersonate rights approval")
        if receipt.get("expected_sha256") != receipt.get("verified_sha256"):
            raise PretrainingFactoryError("retrieval receipt expected/verified hash mismatch")
        observed_registry = receipt.get("source_registry_identity_sha256")
        if observed_registry is not None and observed_registry != registry_identity:
            raise PretrainingFactoryError("retrieval receipt source-registry identity drift")
        key = (
            _require_text(receipt.get("source_id"), "source_id"),
            _require_text(receipt.get("source_version"), "source_version"),
        )
        if key in result:
            raise PretrainingFactoryError("duplicate source version in retrieval inventory")
        result[key] = receipt
    return result


def _eligible_verified_sources(
    source_registry: Mapping[str, Any],
    retrieval_inventory: Mapping[str, Any],
) -> list[tuple[Any, Mapping[str, Any]]]:
    sources = validate_external_source_registry(source_registry)
    source_map = {(item.source_id, item.source_version): item for item in sources}
    receipts = _validate_retrieval_inventory(source_registry, retrieval_inventory)
    selected: list[tuple[Any, Mapping[str, Any]]] = []
    for key in sorted(receipts):
        source = source_map.get(key)
        if source is None:
            raise PretrainingFactoryError("retrieval inventory references unregistered source")
        source.assert_training_eligible()
        if source.source_kind not in SUPPORTED_SOURCE_KINDS:
            raise PretrainingFactoryError(
                f"unsupported source_kind for v1 extraction: {source.source_kind}"
            )
        receipt = receipts[key]
        if receipt.get("expected_sha256") != source.snapshot.sha256:
            raise PretrainingFactoryError("receipt/source snapshot SHA-256 drift")
        if receipt.get("verified_size_bytes") != source.snapshot.size_bytes:
            raise PretrainingFactoryError("receipt/source snapshot size drift")
        selected.append((source, receipt))
    if not selected:
        raise PretrainingFactoryError("no train-eligible verified source versions")
    return selected


def _iter_jsonl_snapshot(
    source: Any,
    receipt: Mapping[str, Any],
) -> Iterator[tuple[str, str, str | None]]:
    path = _local_path(_require_text(receipt.get("destination_uri"), "destination_uri"),
                       "destination_uri")
    if not path.is_file():
        raise PretrainingFactoryError(f"verified snapshot is missing: {path}")
    if path.stat().st_size != source.snapshot.size_bytes:
        raise PretrainingFactoryError("verified snapshot size changed after retrieval")
    if _sha256_file(path) != source.snapshot.sha256:
        raise PretrainingFactoryError("verified snapshot hash changed after retrieval")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PretrainingFactoryError(
                    f"{source.source_id}:{line_number}: invalid JSONL"
                ) from exc
            if not isinstance(row, Mapping):
                raise PretrainingFactoryError("source row must be a JSON object")
            document_id = _require_text(row.get("document_id", row.get("id")), "document_id")
            text = row.get("text")
            if not isinstance(text, str):
                raise PretrainingFactoryError("document text must be a string")
            hint = row.get("language_hint")
            if hint is not None and not isinstance(hint, str):
                raise PretrainingFactoryError("language_hint must be a string")
            yield document_id, text, hint


def _policy_rejection(text: str, language: str, plan: FactoryPlan) -> str | None:
    if len(text) < plan.min_chars:
        return "too_short"
    if len(text) > plan.max_chars:
        return "too_long"
    if EMAIL_RE.search(text):
        return "pii_email"
    if PHONE_RE.search(text):
        return "pii_phone"
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return "empty"
    if sum(char.isalpha() for char in visible) / len(visible) < plan.min_alpha_ratio:
        return "low_alpha_ratio"
    if language not in set(plan.allowed_languages):
        return "language_not_allowed"
    return None


def _stage_marker(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    core = {"schema_version": FACTORY_STAGE_SCHEMA, **dict(payload)}
    complete = {**core, "stage_sha256": _sha256_bytes(_canonical_json_bytes(core))}
    path.write_bytes(_canonical_json_bytes(complete))
    return complete


def _read_stage_marker(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    core = dict(data)
    claimed = core.pop("stage_sha256", None)
    if claimed != _sha256_bytes(_canonical_json_bytes(core)):
        raise PretrainingFactoryError("stage marker identity/content mismatch")
    return data


def prepare_exact_stage(
    plan: FactoryPlan,
    source_registry: Mapping[str, Any],
    retrieval_inventory: Mapping[str, Any],
    reserved_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Stream verified sources through normalization/policy/exact dedup."""

    plan_manifest = plan.manifest()
    source_identity = _require_sha256(
        source_registry.get("registry_identity_sha256"), "registry_identity_sha256"
    )
    if source_identity != plan.source_registry_sha256:
        raise PretrainingFactoryError("factory plan/source registry identity mismatch")
    if retrieval_inventory.get("inventory_sha256") != plan.retrieval_inventory_sha256:
        raise PretrainingFactoryError("factory plan/retrieval inventory identity mismatch")
    reserved_sets = validate_reserved_fingerprint_registry(reserved_registry)
    reserved_identity = _require_sha256(
        reserved_registry.get("registry_identity_sha256"), "reserved_registry_identity_sha256"
    )
    if reserved_identity != plan.reserved_registry_sha256:
        raise PretrainingFactoryError("factory plan/reserved registry identity mismatch")
    reserved_sources = {item.source_id for item in reserved_sets}
    reserved_hashes = {digest for item in reserved_sets for digest in item.normalized_sha256}
    root = _local_path(plan.output_uri, "output_uri") / "01_exact"
    marker = root / "COMPLETE.json"
    records = root / "records.jsonl"
    database = root / "exact.sqlite3"
    if marker.exists():
        prior = _read_stage_marker(marker)
        if prior.get("plan_sha256") != plan_manifest["plan_sha256"]:
            raise PretrainingFactoryError("existing exact stage belongs to another plan")
        if not records.is_file() or prior.get("records_sha256") != _sha256_file(records):
            raise PretrainingFactoryError("existing exact-stage payload is missing/tampered")
        return {**prior, "resumed": True}
    if root.exists() and any(root.iterdir()):
        raise PretrainingFactoryError("incomplete exact stage requires explicit recovery")
    root.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    utf8_bytes = 0
    selected = _eligible_verified_sources(source_registry, retrieval_inventory)
    with SQLiteExactDedupIndex(database) as dedup, records.open("wb") as writer:
        for source, receipt in selected:
            for document_id, raw_text, hint in _iter_jsonl_snapshot(source, receipt):
                counters["input_documents"] += 1
                normalized = normalize_text(raw_text)
                detected = language_id(normalized)
                language = detected if detected != "und" else hint or "und"
                reason = _policy_rejection(normalized, language, plan)
                if reason is not None:
                    counters[f"rejected_{reason}"] += 1
                    continue
                content_sha = _sha256_bytes(normalized.encode("utf-8"))
                if source.source_id in reserved_sources or content_sha in reserved_hashes:
                    counters["benchmark_contamination_rejected"] += 1
                    continue
                if dedup.seen_or_add(content_sha):
                    counters["exact_duplicates_removed"] += 1
                    continue
                record = ProcessedRecord(
                    id=f"{source.source_id}@{source.source_version}::{document_id}",
                    text=normalized,
                    language=language,
                    source_id=source.source_id,
                    source_version=source.source_version,
                    content_sha256=content_sha,
                )
                writer.write(_canonical_json_bytes(record.as_mapping()))
                counters["accepted_exact_unique"] += 1
                languages[language] += 1
                utf8_bytes += len(normalized.encode("utf-8"))
        writer.flush()
        os.fsync(writer.fileno())
        dedup.commit()
    if counters["accepted_exact_unique"] < 2:
        raise PretrainingFactoryError("factory requires at least two accepted unique documents")
    payload = {
        "stage": "exact_normalized_policy_filtered",
        "plan_sha256": plan_manifest["plan_sha256"],
        "records_uri": records.as_uri(),
        "records_sha256": _sha256_file(records),
        "record_count": counters["accepted_exact_unique"],
        "normalized_utf8_bytes": utf8_bytes,
        "counters": dict(sorted(counters.items())),
        "languages": dict(sorted(languages.items())),
        "memory_contract": "streaming_one_record_plus_sqlite_exact_index",
        "copyright_policy": "source-level D03 rights gate; record filters cannot grant rights",
        "lid_backend": "project_heuristic_v1_compatibility_not_future_canonical",
        "reserved_registry_sha256": reserved_identity,
    }
    return {**_stage_marker(marker, payload), "resumed": False}


def validate_datatrove_runtime(expected: str = "0.10.0") -> str:
    """Fail closed unless the exact maintained production runtime is installed."""

    try:
        installed = version("datatrove")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "DataTrove optional runtime is required for production near dedup/Parquet"
        ) from exc
    if installed != expected:
        raise RuntimeError(f"DataTrove runtime mismatch: expected {expected}, got {installed}")
    return installed


def run_datatrove_minhash_local(
    plan: FactoryPlan,
    exact_records_uri: str,
    *,
    tasks: int = 1,
    workers: int = 1,
) -> dict[str, Any]:
    """Execute maintained DataTrove four-stage MinHash on local verified JSONL."""

    validate_datatrove_runtime(plan.datatrove_version)
    _require_positive_int(tasks, "tasks")
    _require_positive_int(workers, "workers")
    if workers > tasks:
        raise PretrainingFactoryError("workers cannot exceed tasks")
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import (
        MinhashConfig,
        MinhashDedupBuckets,
        MinhashDedupCluster,
        MinhashDedupFilter,
    )
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers import JsonlWriter
    from datatrove.utils.hashing import HashConfig

    exact = _local_path(exact_records_uri, "exact_records_uri")
    root = _local_path(plan.output_uri, "output_uri") / "02_minhash"
    signatures = root / "signatures"
    buckets = root / "buckets"
    remove_ids = root / "remove_ids"
    deduped = root / "deduplicated"
    logs = root / "logs"
    config = MinhashConfig(
        hash_config=HashConfig(precision=64),
        num_buckets=plan.minhash_num_buckets,
        hashes_per_bucket=plan.minhash_hashes_per_bucket,
        n_grams=plan.minhash_n_grams,
    )
    reader = JsonlReader(data_folder=str(exact.parent), glob_pattern=exact.name)
    LocalPipelineExecutor(
        pipeline=[reader, MinhashDedupSignature(output_folder=str(signatures), config=config)],
        tasks=tasks,
        workers=workers,
        logging_dir=str(logs / "signatures"),
    ).run()
    LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=str(signatures), output_folder=str(buckets), config=config
            )
        ],
        tasks=plan.minhash_num_buckets,
        workers=min(workers, plan.minhash_num_buckets),
        logging_dir=str(logs / "buckets"),
    ).run()
    LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=str(buckets), output_folder=str(remove_ids), config=config
            )
        ],
        tasks=1,
        workers=1,
        logging_dir=str(logs / "clusters"),
    ).run()
    LocalPipelineExecutor(
        pipeline=[
            reader,
            MinhashDedupFilter(input_folder=str(remove_ids)),
            JsonlWriter(output_folder=str(deduped)),
        ],
        tasks=tasks,
        workers=workers,
        logging_dir=str(logs / "filter"),
    ).run()
    return {
        "backend": "datatrove_minhash",
        "datatrove_version": plan.datatrove_version,
        "output_uri": deduped.as_uri(),
        "tasks": tasks,
        "workers": workers,
        "production_near_dedup_executed": True,
    }


def run_local_fixture_near_dedup(plan: FactoryPlan, exact_records_uri: str) -> dict[str, Any]:
    """Reuse bounded S0 Jaccard only for LOCAL_FREE fixture execution."""

    from .pipeline import _jaccard, _word_shingles

    exact = _local_path(exact_records_uri, "exact_records_uri")
    root = _local_path(plan.output_uri, "output_uri") / "02_near_fixture"
    root.mkdir(parents=True, exist_ok=True)
    output = root / "records.jsonl"
    rows: list[dict[str, Any]] = []
    with exact.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) > plan.local_fixture_near_dedup_limit:
                raise PretrainingFactoryError("fixture near-dedup limit exceeded; use DataTrove")
    kept: list[dict[str, Any]] = []
    signatures: list[frozenset[str]] = []
    removed = 0
    for row in rows:
        signature = _word_shingles(row["text"], 5)
        if any(_jaccard(signature, prior) >= 0.92 for prior in signatures):
            removed += 1
            continue
        signatures.append(signature)
        kept.append(row)
    with output.open("wb") as writer:
        for row in kept:
            writer.write(_canonical_json_bytes(row))
        writer.flush()
        os.fsync(writer.fileno())
    return {
        "backend": "s0_tiny_word_shingle_jaccard_v1",
        "authority": "LOCAL_FREE_SYNTHETIC_COMPATIBILITY_ONLY_NOT_PRODUCTION_NEAR_DEDUP",
        "input_records": len(rows),
        "removed": removed,
        "output_records": len(kept),
        "output_uri": output.as_uri(),
        "output_sha256": _sha256_file(output),
        "production_near_dedup_executed": False,
    }


def _split_for(record_id: str, plan: FactoryPlan) -> str:
    digest = hashlib.sha256(f"{plan.split_seed}\0{record_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    return "validation" if bucket < plan.validation_per_10k else "train"


def _shard_for(record_id: str, plan: FactoryPlan) -> int:
    payload = f"{plan.split_seed}\0shard\0{record_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % plan.shard_count


def finalize_jsonl_and_tokenizer_input(
    plan: FactoryPlan,
    deduped_records_uri: str,
    *,
    near_dedup_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministically split/shard a deduped stream and build tokenizer input."""

    source = _local_path(deduped_records_uri, "deduped_records_uri")
    root = _local_path(plan.output_uri, "output_uri") / "03_final"
    marker = root / "COMPLETE.json"
    if marker.exists():
        prior = _read_stage_marker(marker)
        if prior.get("plan_sha256") != plan.manifest()["plan_sha256"]:
            raise PretrainingFactoryError("existing final stage belongs to another plan")
        return {**prior, "resumed": True}
    if root.exists() and any(root.iterdir()):
        raise PretrainingFactoryError("incomplete final stage requires explicit recovery")
    shard_dir = root / "jsonl_shards"
    tokenizer_dir = root / "tokenizer_input"
    shard_dir.mkdir(parents=True)
    tokenizer_dir.mkdir()
    handles: dict[tuple[str, int], Any] = {}
    tokenizer_path = tokenizer_dir / "train.jsonl"
    tokenizer = tokenizer_path.open("wb")
    counters: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    split_text_bytes: Counter[str] = Counter()
    try:
        for split in ("train", "validation"):
            for shard in range(plan.shard_count):
                path = shard_dir / f"{split}-{shard:05d}.jsonl"
                handles[(split, shard)] = path.open("wb")
        with source.open("r", encoding="utf-8") as reader:
            for line in reader:
                if not line.strip():
                    continue
                row = json.loads(line)
                record_id = _require_text(row.get("id"), "id")
                text = _require_text(row.get("text"), "text")
                language = _require_text(row.get("language"), "language")
                split = _split_for(record_id, plan)
                shard = _shard_for(record_id, plan)
                handles[(split, shard)].write(_canonical_json_bytes(row))
                counters[f"{split}_documents"] += 1
                counters["documents"] += 1
                languages[language] += 1
                split_text_bytes[split] += len(text.encode("utf-8"))
                if split == "train":
                    tokenizer.write(_canonical_json_bytes({"id": record_id, "text": text}))
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
        tokenizer.flush()
        os.fsync(tokenizer.fileno())
    finally:
        for handle in handles.values():
            handle.close()
        tokenizer.close()
    if counters["train_documents"] == 0 or counters["validation_documents"] == 0:
        raise PretrainingFactoryError("deterministic split produced an empty split")
    artifacts = [
        {"uri": path.as_uri(), "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(shard_dir.glob("*.jsonl"))
    ]
    payload = {
        "stage": "final_jsonl_tokenizer_handoff",
        "plan_sha256": plan.manifest()["plan_sha256"],
        "input_sha256": _sha256_file(source),
        "near_dedup_evidence": dict(near_dedup_evidence),
        "documents": counters["documents"],
        "split_documents": {
            "train": counters["train_documents"],
            "validation": counters["validation_documents"],
        },
        "split_text_utf8_bytes": dict(sorted(split_text_bytes.items())),
        "languages": dict(sorted(languages.items())),
        "jsonl_shards": artifacts,
        "tokenizer_input": {
            "uri": tokenizer_path.as_uri(),
            "sha256": _sha256_file(tokenizer_path),
            "format": "jsonl_id_text_v1",
            "canonical_tokenizer_selected": False,
        },
        "parquet_status": "DATATROVE_0_10_0_OPTIONAL_RUNTIME_NOT_EXECUTED_IN_JSONL_STAGE",
        "memory_contract": f"streaming_one_record_plus_{2 * plan.shard_count}_shard_handles",
    }
    return {**_stage_marker(marker, payload), "resumed": False}


def convert_jsonl_shards_to_parquet(
    plan: FactoryPlan,
    final_stage: Mapping[str, Any],
) -> dict[str, Any]:
    """Use maintained DataTrove ParquetWriter over deterministic JSONL shards."""

    if final_stage.get("plan_sha256") != plan.manifest()["plan_sha256"]:
        raise PretrainingFactoryError("Parquet conversion/final-stage plan identity mismatch")
    validate_datatrove_runtime(plan.datatrove_version)
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers import ParquetWriter

    base = _local_path(plan.output_uri, "output_uri")
    input_dir = base / "03_final" / "jsonl_shards"
    root = base / "04_parquet"
    LocalPipelineExecutor(
        pipeline=[
            JsonlReader(data_folder=str(input_dir)),
            ParquetWriter(
                output_folder=str(root),
                output_filename="${rank}_${chunk_index}.parquet",
                expand_metadata=True,
                batch_size=plan.max_records_in_memory,
            ),
        ],
        tasks=plan.shard_count,
        workers=min(plan.shard_count, max(1, os.cpu_count() or 1)),
        logging_dir=str(root / "logs"),
    ).run()
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise PretrainingFactoryError("DataTrove Parquet execution produced no parquet files")
    return {
        "backend": "datatrove_parquet_writer",
        "datatrove_version": plan.datatrove_version,
        "files": [
            {"uri": path.as_uri(), "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
            for path in files
        ],
        "parquet_executed": True,
    }
