"""Scalable exact/near deduplication and benchmark decontamination execution seam.

This module extends the D03 corpus foundation. It does not own benchmark
semantics: D06's ``12-6.benchmark-registry.v1`` remains authoritative. It also
does not grant source rights; decontamination is only one fail-closed input to
training eligibility.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .corpus_foundation import CorpusFoundationError

DATATROVE_VERSION = "0.10.0"
DATATROVE_WHEEL_SHA256 = "c7bb75deed2c3e88fb5138f8ea075a170ee98d6c94fc263829609091ea9c2b5d"
SCALE_PLAN_SCHEMA = "12-6.dedup-scale-plan.v1"
OUTPUT_MANIFEST_SCHEMA = "12-6.dedup-output-manifest.v1"
ELIGIBILITY_SCHEMA = "12-6.training-eligibility-envelope.v1"

try:
    from datatrove.utils.word_tokenizers import WordTokenizer as _DataTroveWordTokenizer
except ImportError:  # DataTrove is optional for normal repo/unit-test environments.
    class _DataTroveWordTokenizer:
        def __init__(self, language: str | None = None) -> None:
            self.language = language


class UnicodeRegexWordTokenizer(_DataTroveWordTokenizer):
    """Deterministic dependency-light tokenizer for mixed EN/UK/code MinHash mechanics."""

    _word_re = re.compile(r"\w+", re.UNICODE)

    def __init__(self) -> None:
        super().__init__(None)

    def word_tokenize(self, text: str) -> list[str]:
        return self._word_re.findall(text)

    def sent_tokenize(self, text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if line.strip()] or [text]

    def span_tokenize(self, text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        spans: list[tuple[int, int]] = []
        offset = 0
        for line in text.splitlines(keepends=True):
            stripped = line.strip()
            if stripped:
                start = offset + len(line) - len(line.lstrip())
                end = offset + len(line.rstrip())
                spans.append((start, end))
            offset += len(line)
        return spans or [(0, len(text))]


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise CorpusFoundationError(f"{field} must be lowercase SHA-256 hex")
    if any(char not in "0123456789abcdef" for char in value):
        raise CorpusFoundationError(f"{field} must be lowercase SHA-256 hex")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CorpusFoundationError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True)
class DataTroveMinhashExecutionPlan:
    """Exact executable MinHash topology layered on the incumbent D03 plan.

    DataTrove 0.10.0 defines the signature as
    ``num_buckets * hashes_per_bucket``. The parent D03 plan carried a generic
    ``minhash_signature_size`` field; this execution plan makes that relationship
    explicit and records the stage-specific task topology required by DataTrove.
    """

    source_registry_sha256: str
    reserved_registry_sha256: str
    input_manifest_sha256: str
    workspace_uri: str
    candidate_shards: int = 8
    workers: int = 2
    n_grams: int = 5
    num_buckets: int = 14
    hashes_per_bucket: int = 8
    minhash_seed: int = 1
    hash_precision: int = 64
    datatrove_version: str = DATATROVE_VERSION
    datatrove_wheel_sha256: str = DATATROVE_WHEEL_SHA256

    def __post_init__(self) -> None:
        _require_sha256(self.source_registry_sha256, "source_registry_sha256")
        _require_sha256(self.reserved_registry_sha256, "reserved_registry_sha256")
        _require_sha256(self.input_manifest_sha256, "input_manifest_sha256")
        _require_sha256(self.datatrove_wheel_sha256, "datatrove_wheel_sha256")
        if not isinstance(self.workspace_uri, str) or not self.workspace_uri.strip():
            raise CorpusFoundationError("workspace_uri must be a non-empty string")
        if "?" in self.workspace_uri or "#" in self.workspace_uri:
            raise CorpusFoundationError("workspace_uri must not contain query/fragment")
        for field in (
            "candidate_shards",
            "workers",
            "n_grams",
            "num_buckets",
            "hashes_per_bucket",
            "minhash_seed",
            "hash_precision",
        ):
            _require_positive_int(getattr(self, field), field)
        if self.workers > max(self.candidate_shards, self.num_buckets):
            raise CorpusFoundationError("workers exceed every MinHash stage task count")
        if self.hash_precision not in {32, 64}:
            raise CorpusFoundationError("hash_precision must be 32 or 64")
        if self.datatrove_version != DATATROVE_VERSION:
            raise CorpusFoundationError(
                f"DataTrove must remain pinned to {DATATROVE_VERSION} until revalidated"
            )

    @property
    def signature_size(self) -> int:
        return self.num_buckets * self.hashes_per_bucket

    @property
    def lsh_similarity_at_50pct_detection(self) -> float:
        return (1.0 - 0.5 ** (1.0 / self.num_buckets)) ** (1.0 / self.hashes_per_bucket)

    def manifest(self) -> dict[str, Any]:
        core = {
            "schema_version": SCALE_PLAN_SCHEMA,
            **asdict(self),
            "signature_size": self.signature_size,
            "lsh_similarity_at_50pct_detection": self.lsh_similarity_at_50pct_detection,
            "topology": {
                "reserved_signature_tasks": 1,
                "reserved_index_bucket_tasks": self.num_buckets,
                "candidate_signature_tasks": self.candidate_shards,
                "candidate_bucket_tasks": self.num_buckets,
                "cluster_tasks": 1,
                "filter_tasks": self.candidate_shards,
                "skip_completed": True,
            },
            "benchmark_authority": "D06_12-6.benchmark-registry.v1",
            "semantic_cleanliness_claimed": False,
        }
        return {**core, "plan_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def validate_datatrove_runtime(plan: DataTroveMinhashExecutionPlan) -> dict[str, Any]:
    """Fail closed unless the exact validated DataTrove runtime/API is installed."""

    try:
        installed = importlib.metadata.version("datatrove")
    except importlib.metadata.PackageNotFoundError as exc:
        raise CorpusFoundationError("DataTrove is not installed") from exc
    if installed != plan.datatrove_version:
        raise CorpusFoundationError(
            f"DataTrove runtime mismatch: expected {plan.datatrove_version}, got {installed}"
        )

    try:
        from datatrove.executor.local import LocalPipelineExecutor  # noqa: F401
        from datatrove.pipeline.dedup import MinhashDedupSignature  # noqa: F401
        from datatrove.pipeline.dedup.minhash import (  # noqa: F401
            MinhashConfig,
            MinhashDedupBuckets,
            MinhashDedupCluster,
            MinhashDedupFilter,
        )
        from datatrove.pipeline.readers import JsonlReader  # noqa: F401
        from datatrove.pipeline.writers.jsonl import JsonlWriter  # noqa: F401
        from datatrove.utils.hashing import HashConfig  # noqa: F401
    except ImportError as exc:
        raise CorpusFoundationError("DataTrove 0.10.0 MinHash API is incomplete") from exc

    return {
        "datatrove_version": installed,
        "expected_wheel_sha256": plan.datatrove_wheel_sha256,
        "api_validation": "PASS",
    }


def datatrove_minhash_config(plan: DataTroveMinhashExecutionPlan):
    """Create the maintained DataTrove MinHash config lazily."""

    validate_datatrove_runtime(plan)
    from datatrove.pipeline.dedup.minhash import MinhashConfig
    from datatrove.utils.hashing import HashConfig

    return MinhashConfig(
        n_grams=plan.n_grams,
        num_buckets=plan.num_buckets,
        hashes_per_bucket=plan.hashes_per_bucket,
        seed=plan.minhash_seed,
        hash_config=HashConfig(precision=plan.hash_precision),
    )


def run_datatrove_reference_index(
    plan: DataTroveMinhashExecutionPlan,
    *,
    reference_input: str | Path,
    workspace: str | Path,
    index_name: str,
) -> dict[str, Any]:
    """Build a MinHash reference index from D06-reserved benchmark documents."""

    validate_datatrove_runtime(plan)
    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import MinhashDedupBuckets
    from datatrove.pipeline.readers import JsonlReader

    workspace = Path(workspace)
    signatures = workspace / "reserved_signatures"
    bucket_pairs = workspace / "reserved_bucket_pairs"
    index = workspace / "reserved_index"
    logs = workspace / "logs" / "reserved"
    config = datatrove_minhash_config(plan)

    stage1 = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(str(reference_input)),
            MinhashDedupSignature(
                output_folder=str(signatures),
                config=config,
                language=UnicodeRegexWordTokenizer(),
            ),
        ],
        tasks=1,
        workers=1,
        logging_dir=str(logs / "signatures"),
        skip_completed=True,
    )
    stage1.run()

    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=str(signatures),
                output_folder=str(bucket_pairs),
                index_folder=str(index),
                config=config,
                only_dedup_in_index=True,
                create_index_name=index_name,
            )
        ],
        tasks=plan.num_buckets,
        workers=min(plan.workers, plan.num_buckets),
        logging_dir=str(logs / "index"),
        skip_completed=True,
    )
    stage2.run()

    return {
        "reference_signatures": str(signatures),
        "reference_index": str(index),
        "index_name": index_name,
        "signature_tasks": 1,
        "bucket_tasks": plan.num_buckets,
    }


def run_datatrove_candidate_dedup(
    plan: DataTroveMinhashExecutionPlan,
    *,
    candidate_input: str | Path,
    workspace: str | Path,
    reference_index: str | Path,
    exercise_restart: bool = True,
) -> dict[str, Any]:
    """Run DataTrove MinHash internal dedup + reference-index decontamination."""

    validate_datatrove_runtime(plan)
    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import (
        MinhashDedupBuckets,
        MinhashDedupCluster,
        MinhashDedupFilter,
    )
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers.jsonl import JsonlWriter

    workspace = Path(workspace)
    signatures = workspace / "candidate_signatures"
    bucket_pairs = workspace / "candidate_bucket_pairs"
    remove_ids = workspace / "candidate_remove_ids"
    removed = workspace / "candidate_removed"
    output = workspace / "candidate_deduplicated"
    logs = workspace / "logs" / "candidate"
    config = datatrove_minhash_config(plan)

    restart: dict[str, Any] = {
        "exercised": bool(exercise_restart),
        "partial_tasks": 0,
        "resumed_tasks": 0,
        "skip_completed": True,
    }
    if exercise_restart and plan.candidate_shards > 1:
        partial = max(1, plan.candidate_shards // 2)
        partial_stage1 = LocalPipelineExecutor(
            pipeline=[
                JsonlReader(str(candidate_input)),
                MinhashDedupSignature(
                    output_folder=str(signatures),
                    config=config,
                    language=UnicodeRegexWordTokenizer(),
                ),
            ],
            tasks=plan.candidate_shards,
            workers=min(plan.workers, partial),
            logging_dir=str(logs / "signatures"),
            skip_completed=True,
            local_tasks=partial,
        )
        partial_stage1.run()
        restart["partial_tasks"] = partial
        restart["resumed_tasks"] = plan.candidate_shards - partial
        restart["partial_signature_files"] = len(list(signatures.rglob("*.minhash.sig")))

    stage1 = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(str(candidate_input)),
            MinhashDedupSignature(
                output_folder=str(signatures),
                config=config,
                language=UnicodeRegexWordTokenizer(),
            ),
        ],
        tasks=plan.candidate_shards,
        workers=min(plan.workers, plan.candidate_shards),
        logging_dir=str(logs / "signatures"),
        skip_completed=True,
    )
    stage1.run()
    restart["final_signature_files"] = len(list(signatures.rglob("*.minhash.sig")))
    restart["verified"] = (
        not exercise_restart
        or plan.candidate_shards <= 1
        or (
            restart.get("partial_signature_files") == restart["partial_tasks"] * plan.num_buckets
            and restart["final_signature_files"] == plan.candidate_shards * plan.num_buckets
        )
    )

    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=str(signatures),
                output_folder=str(bucket_pairs),
                index_folder=str(reference_index),
                config=config,
                only_dedup_in_index=False,
            )
        ],
        tasks=plan.num_buckets,
        workers=min(plan.workers, plan.num_buckets),
        logging_dir=str(logs / "buckets"),
        skip_completed=True,
    )
    stage2.run()

    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=str(bucket_pairs),
                output_folder=str(remove_ids),
                config=config,
            )
        ],
        tasks=1,
        workers=1,
        logging_dir=str(logs / "cluster"),
        skip_completed=True,
    )
    stage3.run()

    stage4 = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(str(candidate_input)),
            MinhashDedupFilter(
                input_folder=str(remove_ids),
                exclusion_writer=JsonlWriter(str(removed), compression=None),
            ),
            JsonlWriter(str(output), compression=None),
        ],
        tasks=plan.candidate_shards,
        workers=min(plan.workers, plan.candidate_shards),
        logging_dir=str(logs / "filter"),
        skip_completed=True,
    )
    stage4.run()

    return {
        "signatures": str(signatures),
        "bucket_pairs": str(bucket_pairs),
        "remove_ids": str(remove_ids),
        "removed": str(removed),
        "output": str(output),
        "restart": restart,
    }


def build_dedup_output_manifest(
    *,
    plan: DataTroveMinhashExecutionPlan,
    input_records: int,
    exact_survivors: int,
    final_survivors: int,
    output_files: Mapping[str, str],
    metrics_sha256: str,
) -> dict[str, Any]:
    """Bind dedup outputs and metrics to the exact plan/input identities."""

    for name, digest in output_files.items():
        _require_sha256(digest, f"output_files[{name}]")
    _require_sha256(metrics_sha256, "metrics_sha256")
    for field, value in {
        "input_records": input_records,
        "exact_survivors": exact_survivors,
        "final_survivors": final_survivors,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CorpusFoundationError(f"{field} must be a non-negative integer")
    if not (final_survivors <= exact_survivors <= input_records):
        raise CorpusFoundationError("dedup survivor counts are inconsistent")

    core = {
        "schema_version": OUTPUT_MANIFEST_SCHEMA,
        "plan_sha256": plan.manifest()["plan_sha256"],
        "source_registry_sha256": plan.source_registry_sha256,
        "reserved_registry_sha256": plan.reserved_registry_sha256,
        "input_manifest_sha256": plan.input_manifest_sha256,
        "input_records": input_records,
        "exact_survivors": exact_survivors,
        "final_survivors": final_survivors,
        "reduction_ratio": (
            (input_records - final_survivors) / input_records if input_records else 0.0
        ),
        "output_files": dict(sorted(output_files.items())),
        "metrics_sha256": metrics_sha256,
    }
    return {**core, "manifest_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def build_training_eligibility_envelope(
    *,
    output_manifest: Mapping[str, Any],
    source_rights_eligible: bool,
    record_policy_eligible: bool,
    exact_reserved_overlap_count: int,
    lexical_reserved_overlap_count: int,
    known_semantic_overlap_count: int,
    experiment_acceptance_pass: bool,
) -> dict[str, Any]:
    """Combine upstream rights/policy state with decontamination evidence.

    Any known residual benchmark relation blocks training, but zero known residuals
    never implies that all semantic contamination in the universe has been found.
    """

    if output_manifest.get("schema_version") != OUTPUT_MANIFEST_SCHEMA:
        raise CorpusFoundationError("unsupported dedup output manifest")
    _require_sha256(output_manifest.get("manifest_sha256"), "output_manifest.manifest_sha256")
    for field, value in {
        "exact_reserved_overlap_count": exact_reserved_overlap_count,
        "lexical_reserved_overlap_count": lexical_reserved_overlap_count,
        "known_semantic_overlap_count": known_semantic_overlap_count,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CorpusFoundationError(f"{field} must be a non-negative integer")
    for field, value in {
        "source_rights_eligible": source_rights_eligible,
        "record_policy_eligible": record_policy_eligible,
        "experiment_acceptance_pass": experiment_acceptance_pass,
    }.items():
        if type(value) is not bool:
            raise CorpusFoundationError(f"{field} must be exact boolean")

    decontamination_eligible = (
        experiment_acceptance_pass
        and exact_reserved_overlap_count == 0
        and lexical_reserved_overlap_count == 0
        and known_semantic_overlap_count == 0
    )
    training_eligible = (
        source_rights_eligible and record_policy_eligible and decontamination_eligible
    )
    core = {
        "schema_version": ELIGIBILITY_SCHEMA,
        "dedup_output_manifest_sha256": output_manifest["manifest_sha256"],
        "source_rights_eligible": source_rights_eligible,
        "record_policy_eligible": record_policy_eligible,
        "decontamination_eligible": decontamination_eligible,
        "training_eligible": training_eligible,
        "known_residuals": {
            "exact_reserved_overlap_count": exact_reserved_overlap_count,
            "lexical_reserved_overlap_count": lexical_reserved_overlap_count,
            "known_semantic_overlap_count": known_semantic_overlap_count,
        },
        "experiment_acceptance_pass": experiment_acceptance_pass,
        "semantic_universal_cleanliness_claimed": False,
        "scope_note": (
            "Eligibility means all registered/known exclusions and configured exact/MinHash "
            "checks passed for this manifest. It is not proof of universal semantic cleanliness."
        ),
    }
    return {**core, "envelope_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def minhash_lsh_detection_probability(
    similarity: float, *, num_buckets: int, hashes_per_bucket: int
) -> float:
    """Return theoretical LSH candidate probability for a Jaccard similarity."""

    if not isinstance(similarity, (int, float)) or isinstance(similarity, bool):
        raise CorpusFoundationError("similarity must be numeric")
    similarity = float(similarity)
    if not math.isfinite(similarity) or not 0.0 <= similarity <= 1.0:
        raise CorpusFoundationError("similarity must be in [0, 1]")
    _require_positive_int(num_buckets, "num_buckets")
    _require_positive_int(hashes_per_bucket, "hashes_per_bucket")
    return 1.0 - (1.0 - similarity**hashes_per_bucket) ** num_buckets
