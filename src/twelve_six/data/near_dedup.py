"""DATA-30 calibrated near-dedup policy layer over the D03/DataTrove incumbent.

This module does not implement a second deduplication engine. All candidate
matching, clustering, and filtering is executed by DataTrove 0.10.0 MinHash.
The code here owns calibration, modality policy selection, provenance reporting,
and deterministic corpus identity only.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .corpus_foundation import CorpusFoundationError
from .dedup_scale import (
    DATATROVE_VERSION,
    DATATROVE_WHEEL_SHA256,
    DataTroveMinhashExecutionPlan,
    UnicodeRegexWordTokenizer,
    datatrove_minhash_config,
    validate_datatrove_runtime,
)

POLICY_SCHEMA = "12-6.near-dedup-policy.v1"
CALIBRATION_SCHEMA = "12-6.near-dedup-calibration.v1"
REPORT_SCHEMA = "12-6.near-dedup-report.v1"
CORPUS_ID_SCHEMA = "12-6.near-dedup-surviving-corpus.v1"


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lsh_detection_probability(similarity: float, *, buckets: int, hashes_per_bucket: int) -> float:
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("similarity must be within [0, 1]")
    return 1.0 - (1.0 - similarity**hashes_per_bucket) ** buckets


@dataclass(frozen=True)
class NearDedupPolicy:
    name: str
    modality: str
    n_grams: int
    num_buckets: int
    hashes_per_bucket: int
    seed: int = 1
    hash_precision: int = 64
    normalize_numbers: bool = False

    def __post_init__(self) -> None:
        if self.modality not in {"natural", "code"}:
            raise ValueError("modality must be natural or code")
        if not self.name:
            raise ValueError("policy name must be non-empty")
        for field in ("n_grams", "num_buckets", "hashes_per_bucket", "seed"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.hash_precision not in {32, 64}:
            raise ValueError("hash_precision must be 32 or 64")
        if type(self.normalize_numbers) is not bool:
            raise ValueError("normalize_numbers must be boolean")

    @property
    def signature_size(self) -> int:
        return self.num_buckets * self.hashes_per_bucket

    @property
    def lsh_similarity_at_50pct_detection(self) -> float:
        return (1.0 - 0.5 ** (1.0 / self.num_buckets)) ** (1.0 / self.hashes_per_bucket)

    def manifest(self) -> dict[str, Any]:
        core = {
            "schema_version": POLICY_SCHEMA,
            **asdict(self),
            "signature_size": self.signature_size,
            "lsh_similarity_at_50pct_detection": self.lsh_similarity_at_50pct_detection,
            "datatrove_version": DATATROVE_VERSION,
            "datatrove_wheel_sha256": DATATROVE_WHEEL_SHA256,
            "lexical_minhash_only": True,
            "semantic_deduplication_claimed": False,
        }
        return {**core, "policy_sha256": sha256_bytes(canonical_json_bytes(core))}


def policy_candidates() -> dict[str, tuple[NearDedupPolicy, ...]]:
    """Small calibration grid centered on the DATA-12 incumbent.

    Natural text keeps the incumbent 9-gram/14x8 policy as the center point.
    Code evaluates a stricter LSH banding choice because boilerplate and forks
    can share large lexical regions without being interchangeable training docs.
    """

    return {
        "natural": (
            NearDedupPolicy("natural_conservative_9g_12x9", "natural", 9, 12, 9),
            NearDedupPolicy("natural_incumbent_9g_14x8", "natural", 9, 14, 8),
            NearDedupPolicy("natural_recall_9g_16x7", "natural", 9, 16, 7),
        ),
        "code": (
            NearDedupPolicy("code_strict_5g_10x10", "code", 5, 10, 10),
            NearDedupPolicy("code_middle_5g_12x9", "code", 5, 12, 9),
            NearDedupPolicy("code_incumbent_band_5g_14x8", "code", 5, 14, 8),
        ),
    }


def _plan_for(
    policy: NearDedupPolicy, input_identity: str, workspace: Path
) -> DataTroveMinhashExecutionPlan:
    zero = "0" * 64
    return DataTroveMinhashExecutionPlan(
        source_registry_sha256=zero,
        reserved_registry_sha256=zero,
        input_manifest_sha256=input_identity,
        workspace_uri=str(workspace),
        candidate_shards=1,
        workers=1,
        n_grams=policy.n_grams,
        num_buckets=policy.num_buckets,
        hashes_per_bucket=policy.hashes_per_bucket,
        minhash_seed=policy.seed,
        hash_precision=policy.hash_precision,
        normalize_numbers=policy.normalize_numbers,
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl_folder(folder: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(folder.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _binary_pairs(path: Path) -> list[tuple[int, int]]:
    payload = path.read_bytes()
    if len(payload) % 8:
        raise CorpusFoundationError(f"corrupt DataTrove metadata file: {path}")
    return [struct.unpack_from("<II", payload, offset) for offset in range(0, len(payload), 8)]


def _file_hashes(folder: Path, pattern: str) -> dict[str, str]:
    return {
        str(path.relative_to(folder)): sha256_file(path)
        for path in sorted(folder.rglob(pattern))
        if path.is_file()
    }


def _cluster_provenance(
    remove_ids: Path,
    ordered_records: Sequence[Mapping[str, Any]],
    survivor_ids: set[str],
) -> list[dict[str, Any]]:
    by_cluster: dict[int, list[int]] = {}
    size_by_doc: dict[int, int] = {}
    for path in sorted(remove_ids.glob("*.clusters")):
        file_id = int(path.name.split(".", 1)[0])
        if file_id != 0:
            raise CorpusFoundationError(
                "DATA-30 bounded runner expects one deterministic input shard"
            )
        for doc_idx, cluster_id in _binary_pairs(path):
            by_cluster.setdefault(cluster_id, []).append(doc_idx)
    for path in sorted(remove_ids.glob("*.sizes")):
        file_id = int(path.name.split(".", 1)[0])
        if file_id != 0:
            raise CorpusFoundationError(
                "DATA-30 bounded runner expects one deterministic input shard"
            )
        for doc_idx, size in _binary_pairs(path):
            size_by_doc[doc_idx] = size

    clusters: list[dict[str, Any]] = []
    for cluster_id, indices in sorted(by_cluster.items()):
        members = []
        for index in sorted(indices):
            if index >= len(ordered_records):
                raise CorpusFoundationError(
                    "DataTrove cluster doc index exceeds deterministic input"
                )
            record = ordered_records[index]
            members.append(
                {
                    "record_id": str(record["id"]),
                    "source_id": str(record.get("metadata", {}).get("source_id", "unknown")),
                    "raw_identity": str(
                        record.get("metadata", {}).get("raw_identity", record["id"])
                    ),
                }
            )
        member_ids = {member["record_id"] for member in members}
        representatives = sorted(member_ids & survivor_ids)
        if len(representatives) != 1:
            raise CorpusFoundationError(
                f"expected exactly one deterministic survivor for cluster {cluster_id}, "
                f"got {representatives}"
            )
        declared_sizes = {size_by_doc[index] for index in indices if index in size_by_doc}
        if declared_sizes and declared_sizes != {len(indices)}:
            raise CorpusFoundationError(
                "DataTrove cluster size metadata disagrees with cluster membership"
            )
        clusters.append(
            {
                "cluster_id": cluster_id,
                "representative_record_id": representatives[0],
                "member_count": len(members),
                "members": members,
            }
        )
    return clusters


def run_datatrove_policy(
    records: Sequence[Mapping[str, Any]],
    *,
    policy: NearDedupPolicy,
    workspace: Path,
    exercise_skip_completed: bool = True,
) -> dict[str, Any]:
    """Execute the maintained DataTrove four-stage MinHash pipeline.

    Records are sorted by stable record id before DataTrove sees them. The bounded
    DATA-30 executor deliberately uses one input shard so DataTrove's (file,doc)
    coordinates can be joined back to source provenance without ambiguity.
    """

    if not records:
        raise ValueError("records must not be empty")
    ordered = sorted((dict(record) for record in records), key=lambda item: str(item["id"]))
    ids = [str(record["id"]) for record in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("record ids must be unique")
    if any(not isinstance(record.get("text"), str) or not record["text"] for record in ordered):
        raise ValueError("every record must contain non-empty text")

    input_identity = sha256_bytes(canonical_json_bytes(ordered))
    plan = _plan_for(policy, input_identity, workspace)
    validate_datatrove_runtime(plan)
    config = datatrove_minhash_config(plan)

    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import (
        MinhashDedupBuckets,
        MinhashDedupCluster,
        MinhashDedupFilter,
    )
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers.jsonl import JsonlWriter

    workspace.mkdir(parents=True, exist_ok=True)
    input_path = workspace / "input" / "00000.jsonl"
    signatures = workspace / "signatures"
    pairs = workspace / "bucket_pairs"
    remove_ids = workspace / "remove_ids"
    removed = workspace / "removed"
    output = workspace / "survivors"
    logs = workspace / "logs"
    _write_jsonl(input_path, ordered)

    def stage1() -> None:
        LocalPipelineExecutor(
            pipeline=[
                JsonlReader(str(input_path)),
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
        ).run()

    stage1()
    first_signature_hashes = _file_hashes(signatures, "*.minhash.sig")
    if exercise_skip_completed:
        stage1()
    second_signature_hashes = _file_hashes(signatures, "*.minhash.sig")
    skip_completed_verified = (
        bool(first_signature_hashes) and first_signature_hashes == second_signature_hashes
    )

    LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=str(signatures),
                output_folder=str(pairs),
                config=config,
                only_dedup_in_index=False,
            )
        ],
        tasks=policy.num_buckets,
        workers=1,
        logging_dir=str(logs / "buckets"),
        skip_completed=True,
    ).run()

    LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=str(pairs),
                output_folder=str(remove_ids),
                config=config,
                save_cluster_id=True,
                save_cluster_size=True,
            )
        ],
        tasks=1,
        workers=1,
        logging_dir=str(logs / "clusters"),
        skip_completed=True,
    ).run()

    LocalPipelineExecutor(
        pipeline=[
            JsonlReader(str(input_path)),
            MinhashDedupFilter(
                input_folder=str(remove_ids),
                exclusion_writer=JsonlWriter(str(removed), compression=None),
            ),
            JsonlWriter(str(output), compression=None),
        ],
        tasks=1,
        workers=1,
        logging_dir=str(logs / "filter"),
        skip_completed=True,
    ).run()

    survivor_records = sorted(_read_jsonl_folder(output), key=lambda item: str(item["id"]))
    removed_records = sorted(_read_jsonl_folder(removed), key=lambda item: str(item["id"]))
    survivor_ids = {str(record["id"]) for record in survivor_records}
    removed_record_ids = {str(record["id"]) for record in removed_records}
    if survivor_ids & removed_record_ids or survivor_ids | removed_record_ids != set(ids):
        raise CorpusFoundationError("DataTrove survivor/removed partition is not exact")

    clusters = _cluster_provenance(remove_ids, ordered, survivor_ids)
    size_histogram: dict[str, int] = {}
    for cluster in clusters:
        key = str(cluster["member_count"])
        size_histogram[key] = size_histogram.get(key, 0) + 1

    input_bytes = sum(len(record["text"].encode("utf-8")) for record in ordered)
    survivor_bytes = sum(len(record["text"].encode("utf-8")) for record in survivor_records)
    return {
        "policy": policy.manifest(),
        "input_identity": input_identity,
        "input_records": len(ordered),
        "survivor_records": len(survivor_records),
        "removed_records": len(removed_records),
        "input_bytes": input_bytes,
        "survivor_bytes": survivor_bytes,
        "removed_bytes": input_bytes - survivor_bytes,
        "document_reduction_ratio": len(removed_records) / len(ordered),
        "byte_reduction_ratio": (
            0.0 if input_bytes == 0 else (input_bytes - survivor_bytes) / input_bytes
        ),
        "survivor_ids": sorted(survivor_ids),
        "removed_ids": sorted(removed_record_ids),
        "clusters": clusters,
        "cluster_statistics": {
            "cluster_count": len(clusters),
            "clustered_documents": sum(cluster["member_count"] for cluster in clusters),
            "largest_cluster": max((cluster["member_count"] for cluster in clusters), default=0),
            "size_histogram": dict(sorted(size_histogram.items())),
        },
        "restart": {
            "skip_completed": True,
            "signature_rerun_exercised": exercise_skip_completed,
            "signature_rerun_byte_identical": skip_completed_verified,
        },
        "engine": {
            "name": "DataTrove MinHash",
            "version": DATATROVE_VERSION,
            "wheel_sha256": DATATROVE_WHEEL_SHA256,
            "second_dedup_engine_created": False,
        },
    }


def load_calibration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CALIBRATION_SCHEMA:
        raise ValueError("unexpected calibration schema")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("calibration must contain pairs")
    seen_pair_ids: set[str] = set()
    for pair in pairs:
        pair_id = str(pair.get("pair_id", ""))
        if not pair_id or pair_id in seen_pair_ids:
            raise ValueError("calibration pair ids must be unique and non-empty")
        seen_pair_ids.add(pair_id)
        if pair.get("modality") not in {"natural", "code"}:
            raise ValueError("calibration pair modality must be natural or code")
        if pair.get("target") not in {"deduplicate", "preserve"}:
            raise ValueError("calibration target must be deduplicate or preserve")
        pair_records = pair.get("records")
        if not isinstance(pair_records, list) or len(pair_records) != 2:
            raise ValueError("each calibration pair must have exactly two records")
    return payload


def _pair_detected(pair: Mapping[str, Any], clusters: Sequence[Mapping[str, Any]]) -> bool:
    ids = {str(record["id"]) for record in pair["records"]}
    return any(
        ids <= {str(member["record_id"]) for member in cluster["members"]}
        for cluster in clusters
    )


def score_calibration(
    calibration: Mapping[str, Any],
    *,
    modality: str,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    pairs = [pair for pair in calibration["pairs"] if pair["modality"] == modality]
    positives = [pair for pair in pairs if pair["target"] == "deduplicate"]
    negatives = [pair for pair in pairs if pair["target"] == "preserve"]
    detected = {pair["pair_id"]: _pair_detected(pair, execution["clusters"]) for pair in pairs}
    true_positive = sum(bool(detected[pair["pair_id"]]) for pair in positives)
    false_positive = sum(bool(detected[pair["pair_id"]]) for pair in negatives)
    category: dict[str, dict[str, int]] = {}
    for pair in pairs:
        bucket = category.setdefault(pair["category"], {"pairs": 0, "detected": 0})
        bucket["pairs"] += 1
        bucket["detected"] += int(bool(detected[pair["pair_id"]]))
    review = []
    for pair in negatives:
        if detected[pair["pair_id"]]:
            review.append(
                {
                    "pair_id": pair["pair_id"],
                    "category": pair["category"],
                    "reason": pair.get(
                        "reason", "labeled preserve pair clustered by lexical MinHash"
                    ),
                    "record_ids": [record["id"] for record in pair["records"]],
                }
            )
    return {
        "modality": modality,
        "positive_pairs": len(positives),
        "preserve_pairs": len(negatives),
        "recall": 1.0 if not positives else true_positive / len(positives),
        "false_removal_risk": 0.0 if not negatives else false_positive / len(negatives),
        "true_positive_pairs": true_positive,
        "false_positive_pairs": false_positive,
        "pair_detection": detected,
        "category_detection": dict(sorted(category.items())),
        "false_positive_review_sample": review[:10],
    }


def select_policy(
    scored: Sequence[tuple[NearDedupPolicy, Mapping[str, Any]]],
    *,
    min_recall: float = 0.75,
    max_false_removal_risk: float = 0.25,
    preferred_policy_name: str | None = None,
) -> NearDedupPolicy:
    acceptable = [
        (policy, metrics)
        for policy, metrics in scored
        if float(metrics["recall"]) >= min_recall
        and float(metrics["false_removal_risk"]) <= max_false_removal_risk
    ]
    if not acceptable:
        raise CorpusFoundationError("no near-dedup policy satisfies calibration safety gates")
    if preferred_policy_name is not None:
        for policy, _metrics in acceptable:
            if policy.name == preferred_policy_name:
                return policy
    acceptable.sort(
        key=lambda item: (
            -float(item[1]["recall"]),
            float(item[1]["false_removal_risk"]),
            -item[0].lsh_similarity_at_50pct_detection,
            item[0].name,
        )
    )
    return acceptable[0][0]


def surviving_corpus_identity(
    survivor_records: Iterable[Mapping[str, Any]],
    *,
    selected_policies: Mapping[str, NearDedupPolicy],
    input_corpus_identity: str,
) -> dict[str, Any]:
    records = []
    for record in survivor_records:
        records.append(
            {
                "id": str(record["id"]),
                "source_id": str(record.get("metadata", {}).get("source_id", "unknown")),
                "raw_identity": str(
                    record.get("metadata", {}).get("raw_identity", record["id"])
                ),
                "text_sha256": sha256_bytes(str(record["text"]).encode("utf-8")),
            }
        )
    core = {
        "schema_version": CORPUS_ID_SCHEMA,
        "input_corpus_identity": input_corpus_identity,
        "policies": {
            name: policy.manifest()["policy_sha256"]
            for name, policy in sorted(selected_policies.items())
        },
        "survivors": sorted(records, key=lambda item: item["id"]),
    }
    return {**core, "surviving_corpus_identity": sha256_bytes(canonical_json_bytes(core))}


def calibration_records(calibration: Mapping[str, Any], modality: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair in calibration["pairs"]:
        if pair["modality"] != modality:
            continue
        for record in pair["records"]:
            item = dict(record)
            metadata = dict(item.get("metadata", {}))
            metadata.update(
                {
                    "pair_id": pair["pair_id"],
                    "category": pair["category"],
                    "source_id": "data30_calibration",
                }
            )
            item["metadata"] = metadata
            records.append(item)
    return records
